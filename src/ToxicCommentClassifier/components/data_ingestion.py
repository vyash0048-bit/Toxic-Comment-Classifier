import os
import sys
from dotenv import load_dotenv
import pandas as pd
from pymongo import MongoClient
from src.ToxicCommentClassifier.logger import logger
from src.ToxicCommentClassifier.exception import CustomException
from src.ToxicCommentClassifier.entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        load_dotenv()  # Loads variables from .env

    def export_collection_as_dataframe(self, collection_name: str) -> pd.DataFrame:
        try:
            database_name = os.getenv("DATABASE_NAME")
            mongodb_url = os.getenv("MONGODB_URI")
            
            if not database_name or not mongodb_url:
                raise ValueError("MongoDB Environment variables not found. Make sure .env is populated.")

            client = MongoClient(mongodb_url)
            database = client[database_name]
            collection = database[collection_name]
            
            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns:
                df = df.drop(columns=["_id"])
                
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        logger.info("Entered the data ingestion method or component")
        try:
            train_collection = os.getenv("TRAIN_COLLECTION")
            test_collection = os.getenv("TEST_COLLECTION")
            
            logger.info(f"Exporting train collection: {train_collection} as dataframe")
            train_df = self.export_collection_as_dataframe(train_collection)
            
            logger.info(f"Exporting test collection: {test_collection} as dataframe")
            test_df = self.export_collection_as_dataframe(test_collection)
            
            logger.info(f"Saving train dataframe to {self.config.train_data_path}")
            train_df.to_csv(self.config.train_data_path, index=False, header=True)
            
            logger.info(f"Saving test dataframe to {self.config.test_data_path}")
            test_df.to_csv(self.config.test_data_path, index=False, header=True)
            
            logger.info("Data Ingestion is completed")

        except Exception as e:
            raise CustomException(e, sys)
