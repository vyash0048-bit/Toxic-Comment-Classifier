import os
import re
import joblib
import numpy as np
import pandas as pd
from src.ToxicCommentClassifier.config.configuration import ConfigurationManager
from src.ToxicCommentClassifier.components.data_preprocessing import clean_text
from src.ToxicCommentClassifier.logger import logger

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

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
            
            if os.environ.get("RENDER") == "true":
                logger.warning("Running on Render (512MB RAM). Skipping BiLSTM load to prevent OOM crash.")
            elif os.path.exists(bilstm_model_path) and os.path.exists(keras_tokenizer_path):
                self.bilstm_model_path = bilstm_model_path
                self.keras_tokenizer_path = keras_tokenizer_path
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
        """Predict using FastText + BiLSTM."""
        if not getattr(self, "bilstm_available", False) and self.bilstm_model is None:
            raise ValueError("BiLSTM model is not available. Please train it first.")

        if self.bilstm_model is None:
            import tensorflow as tf
            self.keras_tokenizer = joblib.load(self.keras_tokenizer_path)
            self.bilstm_model = tf.keras.models.load_model(self.bilstm_model_path)
            logger.info("BiLSTM model lazily loaded successfully.")

        from tensorflow.keras.preprocessing.sequence import pad_sequences

        sequences = self.keras_tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(
            sequences, maxlen=self.bilstm_max_seq_len,
            padding="post", truncating="post"
        )
        probabilities = self.bilstm_model.predict(padded, verbose=0)[0]

        return {
            label: round(float(prob), 4)
            for label, prob in zip(LABELS, probabilities)
        }

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
