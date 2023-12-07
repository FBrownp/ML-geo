from XGboost_for_slopes import logger
from XGboost_for_slopes.pipeline.stage01_data_ingestion import DataIngestionTrainingPipeline
from XGboost_for_slopes.pipeline.stage02_data_validation import DataValidationTrainingPipeline
from XGboost_for_slopes.pipeline.stage03_data_transformation import DataTransformationTrainingPipeline
from XGboost_for_slopes.pipeline.stage04_model_trainer import ModelTrainerPipeline


STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=======x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Validation Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataValidationTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=======x")
except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = "Data Transformation Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataTransformationTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=======x")
except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = "Model Trainer Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelTrainerPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=======x")
except Exception as e:
    logger.exception(e)
    raise e
