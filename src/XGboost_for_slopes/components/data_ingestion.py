import os
import urllib.request as request
from XGboost_for_slopes import logger
from XGboost_for_slopes.utils.common import get_size
from XGboost_for_slopes.entity.config_entity import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url= self.config.source_URL,
                filename= self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exist of size: {get_size(Path(self.config.local_data_file))}")