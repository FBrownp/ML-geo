import os 
from XGboost_for_slopes import logger
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from XGboost_for_slopes.utils.common import save_object
from XGboost_for_slopes.config.configuration import DataTransformationConfig


class DataTransformation():
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_data_transformer_object(self):

        """
        This function is responsible for data transformation
        """
        try:

            num_Pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="median") ),
                ("scaler", StandardScaler()),
                ("polynomial-scaler", PolynomialFeatures(degree=2))
                ])
            logger.info("Data Transformation file created")

            return num_Pipeline
        except Exception as e:
            raise e

    def get_train_test_data(self):
        data = pd.read_csv(self.config.data_path)
        train, test = train_test_split(data,test_size=0.3)
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index= False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index= False)

        logger.info("Splitted data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)
        
    def initiate_data_transformation(self):
        try:

            logger.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()
            save_object(self.config.transformation_path, preprocessing_obj)
            
        except:
            pass