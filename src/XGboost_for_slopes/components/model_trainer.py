import os 
from XGboost_for_slopes import logger
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from XGboost_for_slopes.entity.config_entity import ModelTrainerConfig
from XGboost_for_slopes.utils.common import save_json
from pathlib import Path

class ModelTrainer():
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def get_model_trainer_object(self):

        """
        This function is responsible for training the model
        """
        train_data_df      = pd.read_csv(self.config.train_data_path)
        test_data_df      = pd.read_csv(self.config.test_data_path)

        Transform_pipeline = joblib.load(self.config.transformation_path)

        
        y_data_train_1 = train_data_df[self.config.target_column_1]
        X_data_train_1 = train_data_df.drop([self.config.target_column_1, self.config.target_column_2], axis = 1)

        y_data_test_1 = test_data_df[self.config.target_column_1]
        X_data_test_1 = test_data_df.drop([self.config.target_column_1, self.config.target_column_2], axis = 1)

        X_data_train_1 = Transform_pipeline.fit_transform(X_data_train_1)
        X_data_test_1  = Transform_pipeline.fit_transform(X_data_test_1)


        params_xgb  =  {"reg_alpha": self.config.reg_alpha,
                        "reg_lambda": self.config.reg_lambda}
    
        xgb_Reg = xgb.XGBRegressor(**params_xgb, booster= "gblinear")
        xgb_Reg.fit(X_data_train_1,y_data_train_1)
        r2_value_model_1 = r2_score(y_data_test_1, xgb_Reg.predict(X_data_test_1))

        logger.info(f"Model_1 is fitted, r2 score for testing data is: {r2_value_model_1}")

        joblib.dump(xgb_Reg, os.path.join(self.config.root_dir,self.config.model_name_1))

        logger.info(f"Model_1 is saved in {os.path.join(self.config.root_dir,self.config.model_name_1)}")


        y_data_train_2 = train_data_df[self.config.target_column_2]
        X_data_train_2 = train_data_df.drop([self.config.target_column_1, self.config.target_column_2], axis = 1)

        y_data_test_2 = test_data_df[self.config.target_column_2]
        X_data_test_2 = test_data_df.drop([self.config.target_column_1, self.config.target_column_2], axis = 1)

        X_data_train_2 = Transform_pipeline.fit_transform(X_data_train_2)
        X_data_test_2  = Transform_pipeline.fit_transform(X_data_test_2)
    
        xgb_Reg = xgb.XGBRegressor(**params_xgb, booster= "gblinear")
        xgb_Reg.fit(X_data_train_2,y_data_train_2)
        r2_value_model_2 = r2_score(y_data_test_2, xgb_Reg.predict(X_data_test_2))

        logger.info(f"Model_2 is fitted, r2 score for testing data is: {r2_value_model_2}")

        joblib.dump(xgb_Reg, os.path.join(self.config.root_dir,self.config.model_name_2))

        logger.info(f"Model_2 is saved in {os.path.join(self.config.root_dir,self.config.model_name_2)}")

        scores = {"r2_score_model_1" : r2_value_model_1, "r2_score_model_2": r2_value_model_2}
        save_json(path= Path("scores.json"), data = scores)
        