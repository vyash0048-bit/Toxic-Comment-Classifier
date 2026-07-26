import os
import sys
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from src.ToxicCommentClassifier.logger import logger
from src.ToxicCommentClassifier.exception import CustomException
from src.ToxicCommentClassifier.entity import ModelTrainingConfig

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    def initiate_model_training(self):
        try:
            logger.info(f"Loading preprocessed training data from {self.config.train_data_path}")
            train_data = joblib.load(self.config.train_data_path)
            X_train = train_data["X"]
            y_train = train_data["y"]

            logger.info(f"Loading preprocessed validation data from {self.config.test_data_path}")
            val_data = joblib.load(self.config.test_data_path)
            X_validation = val_data["X"]
            y_validation = val_data["y"]

            logger.info("Initializing LogisticRegression wrapped in OneVsRestClassifier with params.yaml config")
            model = OneVsRestClassifier(LogisticRegression(
                solver=self.config.solver, 
                C=self.config.C, 
                max_iter=self.config.max_iter,
                n_jobs=self.config.n_jobs,
                verbose=self.config.verbose
            ), n_jobs=-1)

            logger.info("Training the model on sparse matrix... this may take some time.")
            model.fit(X_train, y_train)

            logger.info("Predicting probabilities for the validation set...")
            validation_probabilities = model.predict_proba(X_validation)

            logger.info("Calculating ROC-AUC scores for each label...")
            per_label_auc = pd.Series({
                label: roc_auc_score(y_validation[label], validation_probabilities[:, i])
                for i, label in enumerate(LABELS)
            }, name="ROC-AUC")
            
            logger.info(f"Per-label ROC-AUC:\n{per_label_auc.to_frame()}")
            logger.info(f"Mean ROC-AUC: {per_label_auc.mean():.4f}")

            model_save_path = os.path.join(self.config.root_dir, self.config.model_name)
            logger.info(f"Saving the trained model to {model_save_path}")
            joblib.dump(model, model_save_path)

            logger.info("Model training and validation completed successfully!")

        except Exception as e:
            raise CustomException(e, sys)
