import os
import sys
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, log_loss
from src.ToxicCommentClassifier.utils.common import save_json
from src.ToxicCommentClassifier.logger import logger
from src.ToxicCommentClassifier.exception import CustomException
from src.ToxicCommentClassifier.entity import ModelEvaluationConfig
from pathlib import Path
import mlflow
import dagshub

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def initiate_model_evaluation(self):
        try:
            logger.info(f"Loading test data from {self.config.test_data_path}")
            test_data = joblib.load(self.config.test_data_path)
            X_test = test_data["X"]
            y_test = test_data["y"]

            if y_test is None:
                logger.warning("Test dataset has no labels. Evaluation aborted.")
                return

            logger.info(f"Loading trained model from {self.config.model_path}")
            model = joblib.load(self.config.model_path)

            logger.info("Predicting probabilities and binary classes on test set...")
            test_probabilities = model.predict_proba(X_test)
            test_predictions = model.predict(X_test)

            logger.info("Calculating metrics...")
            metrics = {}
            per_label_auc = {}
            
            for i, label in enumerate(LABELS):
                y_true = y_test[label]
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
            
            logger.info(f"Mean ROC-AUC: {metrics['mean_auc']:.4f}")

            logger.info(f"Saving metrics to {self.config.metric_file_name}")
            save_json(path=Path(self.config.metric_file_name), data=metrics)

            logger.info("Initializing MLFlow via Dagshub...")
            dagshub.init(repo_owner='vyash0048', repo_name='Toxic-Comment-Classifier', mlflow=True)

            logger.info("Logging to MLFlow...")
            with mlflow.start_run():
                mlflow.log_params(self.config.all_params.get("ModelTraining", {}))
                mlflow.log_params(self.config.all_params.get("DataPreprocessing", {}))
                mlflow.log_metrics(metrics)

            logger.info("Model evaluation and MLflow logging completed successfully!")

        except Exception as e:
            raise CustomException(e, sys)
