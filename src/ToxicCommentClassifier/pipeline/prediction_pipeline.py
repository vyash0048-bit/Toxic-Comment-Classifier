import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from src.ToxicCommentClassifier.config.configuration import ConfigurationManager
from src.ToxicCommentClassifier.components.data_preprocessing import clean_text
from src.ToxicCommentClassifier.logger import logger

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


class SimpleTokenizer:
    """Lightweight tokenizer that replicates Keras Tokenizer.texts_to_sequences
    without requiring TensorFlow. Loads from a JSON config file."""

    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.word_index = data["word_index"]
        self.num_words = data.get("num_words")
        self.oov_token = data.get("oov_token")
        self.lower = data.get("lower", True)
        self.char_level = data.get("char_level", False)

    def texts_to_sequences(self, texts: list) -> list:
        """Convert list of texts to list of integer sequences."""
        sequences = []
        for text in texts:
            if self.lower:
                text = text.lower()
            if self.char_level:
                words = list(text)
            else:
                words = text.split()
            seq = []
            for word in words:
                idx = self.word_index.get(word)
                if idx is not None:
                    if self.num_words is None or idx < self.num_words:
                        seq.append(idx)
            sequences.append(seq)
        return sequences

class PredictionPipeline:
    def __init__(self):
        config_manager = ConfigurationManager()

        # --- Load Logistic Regression artifacts (if available) ---
        self.tfidf_vectorizer = None
        self.lr_model = None

        try:
            preprocessing_config = config_manager.get_data_preprocessing_config()
            evaluation_config = config_manager.get_model_evaluation_config()

            vectorizer_path = str(preprocessing_config.tokenizer_path)
            model_path = str(evaluation_config.model_path)

            if os.path.exists(vectorizer_path) and os.path.exists(model_path):
                self.tfidf_vectorizer = joblib.load(vectorizer_path)
                self.lr_model = joblib.load(model_path)
                logger.info("Logistic Regression model loaded successfully.")
            else:
                logger.warning("LR model artifacts not found. LR predictions will be unavailable.")
        except Exception as e:
            logger.warning(f"Could not load LR model: {e}. LR predictions will be unavailable.")

        # --- Load BiLSTM artifacts (if available) ---
        self.bilstm_model = None
        self.keras_tokenizer = None
        self.bilstm_max_seq_len = None
        self.bilstm_available = False

        try:
            bilstm_config = config_manager.get_bilstm_training_config()
            bilstm_eval_config = config_manager.get_bilstm_evaluation_config()
            
            bilstm_model_path = str(bilstm_config.model_path)
            keras_tokenizer_path = str(bilstm_config.tokenizer_path)
            
            json_tokenizer_path = os.path.join(os.path.dirname(keras_tokenizer_path), "tokenizer_config.json")
            
            if os.environ.get("RENDER") == "true":
                logger.warning("Running on Render (512MB RAM). Skipping BiLSTM load to prevent OOM crash.")
            elif os.path.exists(bilstm_model_path) and (os.path.exists(keras_tokenizer_path) or os.path.exists(json_tokenizer_path)):
                self.bilstm_model_path = bilstm_model_path
                self.keras_tokenizer_path = json_tokenizer_path if os.path.exists(json_tokenizer_path) else keras_tokenizer_path
                self.bilstm_max_seq_len = bilstm_config.max_seq_len
                self.bilstm_available = True
                logger.info("BiLSTM artifacts found. Model will be loaded lazily on first prediction to prevent early CUDA initialization.")
            else:
                logger.warning("BiLSTM model artifacts not found. BiLSTM predictions will be unavailable.")
        except Exception as e:
            logger.warning(f"Could not load BiLSTM model artifacts: {e}. BiLSTM predictions will be unavailable.")

    def predict(self, text: str, model_type: str = "lr") -> dict:
        """
        Takes a raw string, preprocesses it, and returns a dictionary 
        of toxicity labels with their predicted probabilities.
        
        Args:
            text: The raw comment text.
            model_type: "lr" for Logistic Regression, "bilstm" for BiLSTM.
        """
        cleaned = clean_text(text)

        if model_type == "bilstm":
            return self._predict_bilstm(cleaned)
        else:
            return self._predict_lr(cleaned)

    def _predict_lr(self, cleaned_text: str) -> dict:
        """Predict using TF-IDF + Logistic Regression."""
        if self.lr_model is None or self.tfidf_vectorizer is None:
            raise ValueError("LR model is not available. Please train it first.")

        vectorized_text = self.tfidf_vectorizer.transform([cleaned_text])
        probabilities = self.lr_model.predict_proba(vectorized_text)[0]
        
        return {
            label: round(float(prob), 4) 
            for label, prob in zip(LABELS, probabilities)
        }

    def _predict_bilstm(self, cleaned_text: str) -> dict:
        """Predict using FastText + BiLSTM (TFLite Runtime for lightweight deployment)."""
        if not getattr(self, "bilstm_available", False) and self.bilstm_model is None:
            raise ValueError("BiLSTM model is not available. Please train it first.")

        if self.bilstm_model is None:
            # Load tokenizer (JSON preferred, fallback to joblib which needs keras)
            if self.keras_tokenizer_path.endswith(".json"):
                self.keras_tokenizer = SimpleTokenizer(self.keras_tokenizer_path)
            else:
                self.keras_tokenizer = joblib.load(self.keras_tokenizer_path)

            import tensorflow as tf
            self.bilstm_model = tf.keras.models.load_model(self.bilstm_model_path)
            logger.info("BiLSTM Keras model lazily loaded successfully.")

        # Tokenize and pad the input
        sequences = self.keras_tokenizer.texts_to_sequences([cleaned_text])
        padded = self._pad_sequences(
            sequences, maxlen=self.bilstm_max_seq_len
        ).astype(np.int32)

        # Run inference using TensorFlow CPU
        probabilities = self.bilstm_model.predict(padded, verbose=0)[0]
        return {
            label: round(float(prob), 4)
            for label, prob in zip(LABELS, probabilities)
        }

    @staticmethod
    def _pad_sequences(sequences, maxlen, padding="post", truncating="post", value=0):
        """Lightweight pad_sequences replacement (no TensorFlow dependency needed)."""
        result = np.full((len(sequences), maxlen), value, dtype=np.int32)
        for i, seq in enumerate(sequences):
            if truncating == "post":
                trunc = seq[:maxlen]
            else:
                trunc = seq[-maxlen:]
            if padding == "post":
                result[i, :len(trunc)] = trunc
            else:
                result[i, maxlen - len(trunc):] = trunc
        return result

    @property
    def available_models(self) -> list:
        """Return list of available model types."""
        models = []
        if self.lr_model is not None:
            models.append("lr")
        if getattr(self, "bilstm_available", False) or self.bilstm_model is not None:
            models.append("bilstm")
        return models


# Quick test logic if this script is run directly
if __name__ == "__main__":
    pipeline = PredictionPipeline()
    sample_text = "I love this project, it's so helpful!"
    print(f"Text: '{sample_text}'")
    print("LR Predictions:", pipeline.predict(sample_text, model_type="lr"))
    if "bilstm" in pipeline.available_models:
        print("BiLSTM Predictions:", pipeline.predict(sample_text, model_type="bilstm"))
