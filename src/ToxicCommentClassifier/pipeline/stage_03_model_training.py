import os
from src.ToxicCommentClassifier.config.configuration import ConfigurationManager
from src.ToxicCommentClassifier.components.model_training import ModelTraining
from src.ToxicCommentClassifier.logger import logger

STAGE_NAME = "Model Training stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        
        model_path = os.path.join(model_training_config.root_dir, model_training_config.model_name)
        if os.path.exists(model_path):
            logger.info("Trained model file already exists. Skipping stage.")
            return

        model_training = ModelTraining(config=model_training_config)
        model_training.initiate_model_training()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
