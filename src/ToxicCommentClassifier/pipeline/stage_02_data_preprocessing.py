import os
from src.ToxicCommentClassifier.config.configuration import ConfigurationManager
from src.ToxicCommentClassifier.components.data_preprocessing import DataPreprocessing
from src.ToxicCommentClassifier.logger import logger

STAGE_NAME = "Data Preprocessing stage"

class DataPreprocessingTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()
        
        if (os.path.exists(data_preprocessing_config.preprocessed_train_data_path) and 
            os.path.exists(data_preprocessing_config.preprocessed_test_data_path) and 
            os.path.exists(data_preprocessing_config.tokenizer_path)):
            logger.info("Data Preprocessing files already exist. Skipping stage.")
            return

        data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
        data_preprocessing.initiate_data_preprocessing()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreprocessingTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
