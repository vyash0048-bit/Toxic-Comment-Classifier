import os
from src.ToxicCommentClassifier.config.configuration import ConfigurationManager
from src.ToxicCommentClassifier.components.bilstm_evaluation import BiLSTMEvaluation
from src.ToxicCommentClassifier.logger import logger

STAGE_NAME = "BiLSTM Evaluation stage"

class BiLSTMEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        bilstm_evaluation_config = config.get_bilstm_evaluation_config()
        
        bilstm_evaluation = BiLSTMEvaluation(config=bilstm_evaluation_config)
        bilstm_evaluation.initiate_bilstm_evaluation()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = BiLSTMEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
