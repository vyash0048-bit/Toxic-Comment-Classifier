import os
from src.ToxicCommentClassifier.config.configuration import ConfigurationManager
from src.ToxicCommentClassifier.components.bilstm_training import BiLSTMTraining
from src.ToxicCommentClassifier.logger import logger

STAGE_NAME = "BiLSTM Training stage"

class BiLSTMTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        bilstm_training_config = config.get_bilstm_training_config()
        
        model_path = str(bilstm_training_config.model_path)
        if os.path.exists(model_path):
            logger.info("BiLSTM model file already exists. Skipping stage.")
            return

        bilstm_training = BiLSTMTraining(config=bilstm_training_config)
        bilstm_training.initiate_bilstm_training()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = BiLSTMTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
