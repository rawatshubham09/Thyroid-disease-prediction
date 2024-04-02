from Thyroid_Disease.config.configuration import ConfigurationManager
from Thyroid_Disease.components.data_transformation import DataTransformation
from Thyroid_Disease import logger
from pathlib import Path



STAGE_NAME = "Data Transformation stage"
status_file_path = Path("artifacts/data_validation/status.txt")

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            with open(status_file_path,"r") as f:
                status = f.read().split(" ")[-1]
            if status == 'True':
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(data_transformation_config)
                data_transformation.run()
            else:
                raise Exception("Data Schema is not valid")
        except Exception as e:
            raise e
        
if __name__ == "__main__":
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx==============x")
    except Exception as e:
        logger.exception(e)
        raise e
    