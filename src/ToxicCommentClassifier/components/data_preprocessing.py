import os
import sys
import re
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from src.ToxicCommentClassifier.logger import logger
from src.ToxicCommentClassifier.exception import CustomException
from src.ToxicCommentClassifier.entity import DataPreprocessingConfig

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def clean_text(text):
    text = "" if pd.isna(text) else str(text).lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    return re.sub(r"\s+", " ", text).strip() or "unknown"

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    def make_vectorizer(self):
        return FeatureUnion([
            ("word", TfidfVectorizer(
                sublinear_tf=True, strip_accents="unicode", analyzer="word", 
                token_pattern=r"\w{1,}", stop_words="english", 
                ngram_range=tuple(self.config.word_ngram_range), 
                min_df=self.config.word_min_df, 
                max_features=self.config.word_max_features)),
            ("char", TfidfVectorizer(
                sublinear_tf=True, strip_accents="unicode", analyzer="char", 
                ngram_range=tuple(self.config.char_ngram_range), 
                min_df=self.config.char_min_df, 
                max_features=self.config.char_max_features)),
        ])

    def initiate_data_preprocessing(self):
        try:
            logger.info("Reading train and test data for preprocessing")
            train_df = pd.read_csv(self.config.train_data_path)
            test_df = pd.read_csv(self.config.test_data_path)

            logger.info("Cleaning comment text")
            for frame in (train_df, test_df):
                frame["text_clean"] = frame["comment_text"].map(clean_text)

            logger.info("Extracting meta features and validating labels")
            train_df["comment_length"] = train_df["comment_text"].fillna("").str.len()
            train_df["is_toxic"] = train_df[LABELS].any(axis=1).astype(int)
            train_df["label_count"] = train_df[LABELS].sum(axis=1)
            
            test_df["comment_length"] = test_df["comment_text"].fillna("").str.len()
            
            logger.info(f"Splitting train into train and validation sets (test_size={self.config.test_size}, random_state={self.config.random_state})")
            train_indices, validation_indices = train_test_split(
                train_df.index, 
                test_size=self.config.test_size, 
                random_state=self.config.random_state, 
                stratify=train_df["is_toxic"]
            )
            train_split, validation_split = train_df.loc[train_indices], train_df.loc[validation_indices]

            logger.info("Fitting and transforming TfidfVectorizer (Word + Char Level) with params from params.yaml")
            vectorizer = self.make_vectorizer()
            X_train = vectorizer.fit_transform(train_split["text_clean"])
            X_validation = vectorizer.transform(validation_split["text_clean"])
            
            # test_df transform
            X_test = vectorizer.transform(test_df["text_clean"])

            y_train = train_split[LABELS].astype(int)
            y_validation = validation_split[LABELS].astype(int)
            
            # test_df may not have labels, but extract if it does
            y_test = test_df[LABELS].astype(int) if set(LABELS).issubset(test_df.columns) else None

            logger.info(f"Training matrix: {X_train.shape}; validation matrix: {X_validation.shape}")

            logger.info(f"Saving vectorizer to {self.config.tokenizer_path}")
            joblib.dump(vectorizer, self.config.tokenizer_path)

            logger.info("Saving preprocessed matrices (sparse scipy matrices) to joblib files")
            joblib.dump({"X": X_train, "y": y_train}, self.config.preprocessed_train_data_path)
            joblib.dump({"X": X_validation, "y": y_validation}, self.config.preprocessed_test_data_path)
            
            # Save the transformed blind test set as well
            test_path = str(self.config.preprocessed_test_data_path).replace("preprocessed_test", "preprocessed_blind_test")
            joblib.dump({"X": X_test, "y": y_test}, test_path)

            logger.info("Data preprocessing completed successfully!")

        except Exception as e:
            raise CustomException(e, sys)
