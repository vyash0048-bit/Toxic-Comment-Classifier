import os
import sys
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, log_loss
from src.ToxicCommentClassifier.utils.common import save_json
from src.ToxicCommentClassifier.logger import logger
from src.ToxicCommentClassifier.exception import CustomException
from src.ToxicCommentClassifier.entity import BiLSTMEvaluationConfig
from pathlib import Path
import mlflow
import dagshub

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def clean_text(text):
    """Same cleaning function used by the TF-IDF pipeline."""
    text = "" if pd.isna(text) else str(text).lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    return re.sub(r"\s+", " ", text).strip() or "unknown"


class BiLSTMEvaluation:
    def __init__(self, config: BiLSTMEvaluationConfig):
        self.config = config

    def initiate_bilstm_evaluation(self):
        try:
            import tensorflow as tf
            from tensorflow.keras.preprocessing.sequence import pad_sequences

            # --- 1. Load test data ---
            logger.info(f"Loading test data from {self.config.test_data_path}")
            test_df = pd.read_csv(self.config.test_data_path)
            test_df["text_clean"] = test_df["comment_text"].map(clean_text)

            # Check if labels exist
            if not set(LABELS).issubset(test_df.columns):
                logger.warning("Test dataset has no labels. BiLSTM Evaluation aborted.")
                return

            # Filter out Kaggle's -1 padding labels
            valid_mask = (test_df[LABELS] != -1).all(axis=1)
            test_df = test_df[valid_mask]

            if len(test_df) == 0:
                logger.warning("No valid test samples after filtering. Evaluation aborted.")
                return

            # --- 2. Load tokenizer and model ---
            logger.info(f"Loading Keras tokenizer from {self.config.keras_tokenizer_path}")
            keras_tokenizer = joblib.load(self.config.keras_tokenizer_path)

            logger.info(f"Loading BiLSTM model from {self.config.model_path}")
            model = tf.keras.models.load_model(str(self.config.model_path))

            logger.info(f"Loading optimal thresholds from {self.config.optimal_thresholds_path}")
            optimal_thresholds = joblib.load(self.config.optimal_thresholds_path)

            # --- 3. Prepare features ---
            X_test = pad_sequences(
                keras_tokenizer.texts_to_sequences(test_df["text_clean"]),
                maxlen=self.config.max_seq_len, padding="post", truncating="post"
            )
            y_test = test_df[LABELS].values.astype(np.float32)

            logger.info(f"Test matrix shape: {X_test.shape}")

            # --- 4. Predict ---
            logger.info("Predicting on test set...")
            test_probabilities = model.predict(X_test, batch_size=256, verbose=1)
            
            # Apply per-label optimal thresholds
            test_predictions = np.zeros_like(test_probabilities, dtype=int)
            for i, label in enumerate(LABELS):
                thresh = optimal_thresholds.get(label, 0.5)
                test_predictions[:, i] = (test_probabilities[:, i] >= thresh).astype(int)

            # --- 5. Calculate metrics ---
            logger.info("Calculating metrics...")
            metrics = {}
            per_label_auc = {}

            for i, label in enumerate(LABELS):
                y_true = y_test[:, i]
                y_prob = test_probabilities[:, i]
                y_pred = test_predictions[:, i]

                auc = roc_auc_score(y_true, y_prob)
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                loss = log_loss(y_true, y_prob)

                per_label_auc[label] = float(auc)

                metrics[f"{label}_auc"] = float(auc)
                metrics[f"{label}_acc"] = float(acc)
                metrics[f"{label}_precision"] = float(prec)
                metrics[f"{label}_recall"] = float(rec)
                metrics[f"{label}_f1"] = float(f1)
                metrics[f"{label}_log_loss"] = float(loss)

            metrics["mean_auc"] = sum(per_label_auc.values()) / len(per_label_auc)

            logger.info(f"BiLSTM Mean ROC-AUC: {metrics['mean_auc']:.4f}")

            # --- 6. Save metrics ---
            logger.info(f"Saving metrics to {self.config.metric_file_name}")
            save_json(path=Path(self.config.metric_file_name), data=metrics)

            # --- 7. Log to MLflow ---
            logger.info("Initializing MLFlow via Dagshub...")
            dagshub.init(repo_owner='vyash0048', repo_name='Toxic-Comment-Classifier', mlflow=True)

            logger.info("Logging BiLSTM metrics to MLFlow...")
            with mlflow.start_run(run_name="BiLSTM"):
                mlflow.log_params({f"bilstm_{k}": v for k, v in self.config.all_params.get("BiLSTMTraining", {}).items()})
                mlflow.log_params({f"preprocess_{k}": v for k, v in self.config.all_params.get("DataPreprocessing", {}).items()})
                mlflow.log_metrics({f"bilstm_{k}": v for k, v in metrics.items()})

            logger.info("BiLSTM evaluation and MLflow logging completed successfully!")

        except Exception as e:
            raise CustomException(e, sys)
