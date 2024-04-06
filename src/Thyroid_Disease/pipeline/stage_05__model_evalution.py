from Thyroid_Disease.config.configuration import ConfigurationManager
from Thyroid_Disease.components.model_evaluation import ModelEvaluation
from Thyroid_Disease import logger
from pathlib import Path


STAGE_NAME = "Model Evaluation stage"

class ModelEvalutionPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evalution_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.log_into_mlflow()

if __name__ == "__main__":
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvalutionPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx==============x")
    except Exception as e:
        logger.exception(e)
        raise e