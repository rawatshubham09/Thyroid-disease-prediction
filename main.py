from src.Thyroid_Disease import logger
from src.Thyroid_Disease.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline


STAGE_NAME = "Data  Ingestion Pipeline"

try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx==============x")
except Exception as e:
    logger.exception(e)
    raise e
