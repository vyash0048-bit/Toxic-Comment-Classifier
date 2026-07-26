import os
from src.ToxicCommentClassifier.config.configuration import ConfigurationManager
from src.ToxicCommentClassifier.components.model_evaluation import ModelEvaluation
from src.ToxicCommentClassifier.logger import logger

STAGE_NAME = "Model Evaluation stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        
        if os.path.exists(model_evaluation_config.metric_file_name):
            logger.info("Metrics file already exists. Skipping evaluation stage.")
            return

        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.initiate_model_evaluation()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
