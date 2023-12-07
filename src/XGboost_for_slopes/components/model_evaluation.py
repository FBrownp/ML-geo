from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import os 
from XGboost_for_slopes import logger
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from XGboost_for_slopes.utils.common import save_json
import mlflow
import mlflow.xgboost
from  urllib.parse import urlparse
from XGboost_for_slopes.entity.config_entity import ModelEvaluationConfig
from pathlib import Path

class ModelEvaluation():
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def get_model_evaluation_object(self):
        
        test_data_df      = pd.read_csv(self.config.test_data_path)
        
        Transform_pipeline = joblib.load(self.config.transformation_path)


        y_data_test_1 = test_data_df[self.config.target_column_1]
        X_data_test_1 = test_data_df.drop([self.config.target_column_1, self.config.target_column_2], axis = 1)
        X_data_test_1  = Transform_pipeline.fit_transform(X_data_test_1)


        y_data_test_2 = test_data_df[self.config.target_column_2]
        X_data_test_2 = test_data_df.drop([self.config.target_column_1, self.config.target_column_2], axis = 1)
        X_data_test_2  = Transform_pipeline.fit_transform(X_data_test_2)


        model_1 = joblib.load(os.path.join(self.config.model_path,self.config.model_name_1))
        model_2 = joblib.load(os.path.join(self.config.model_path,self.config.model_name_2))


        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():

            r2_value_model_1 = r2_score(y_data_test_1, model_1.predict(X_data_test_1))
            MSE_value_model_1 = mean_squared_error(y_data_test_1, model_1.predict(X_data_test_1))
            MAPE_value_model_1 = mean_absolute_percentage_error(y_data_test_1, model_1.predict(X_data_test_1))


            r2_value_model_2 = r2_score(y_data_test_2, model_2.predict(X_data_test_2))
            MSE_value_model_2 = mean_squared_error(y_data_test_2, model_2.predict(X_data_test_2))
            MAPE_value_model_2 = mean_absolute_percentage_error(y_data_test_2, model_2.predict(X_data_test_2))

            scores = {"r2_m1" : r2_value_model_1,
                    "MSE_m1": MSE_value_model_1,
                    "MAPE_m1": MAPE_value_model_1,
                    "r2_m2" : r2_value_model_2,
                    "MSE_m2": MSE_value_model_2,
                    "MAPE_m2": MAPE_value_model_2
                    }

            save_json(path= Path(os.path.join(self.config.root_dir,"scores.json")), data = scores)
            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("r2_m1",r2_value_model_1)
            mlflow.log_metric("r2_m2",r2_value_model_2)
            mlflow.log_metric("MSE_m1",MSE_value_model_1)
            mlflow.log_metric("MSE_m2",MSE_value_model_2)
            mlflow.log_metric("MAPE_m1",MAPE_value_model_1)
            mlflow.log_metric("MAPE_m2",MAPE_value_model_2)

            if tracking_url_type_store != "file":
                mlflow.xgboost.log_model(model_1, "model_1", registered_model_name="XGboost_FSs")
                mlflow.xgboost.log_model(model_2, "model_2", registered_model_name="XGboost_FSgmp")
            else:
                mlflow.xgboost.log_model(model_1, "model_1")
