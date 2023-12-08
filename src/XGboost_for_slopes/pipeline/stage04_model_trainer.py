from XGboost_for_slopes.config.configuration import ConfigurationManager
from XGboost_for_slopes.components.model_trainer import ModelTrainer
from XGboost_for_slopes import logger
import sys
from XGboost_for_slopes.exception.exception import CustomException

STAGE_NAME = "Model Trainer Stage"

class ModelTrainerPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config = model_trainer_config)
        model_trainer.get_model_trainer_object()




if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=======x")
    except Exception as e:
        logger.exception(e)
        raise CustomException(e,sys)
