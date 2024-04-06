import os
import pandas as pd
from sklearn.metrics import recall_score, confusion_matrix, f1_score, accuracy_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from Thyroid_Disease.config.configuration import ModulEvaluationConfig
from pathlib import Path
from Thyroid_Disease.utils.common import save_json


class ModelEvaluation:
    def __init__(self, config: ModulEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        recall = recall_score(actual,pred, average="weighted")
        #confusion_mat = confusion_matrix(actual, pred)
        f1_s = f1_score(actual,pred,average="weighted")
        accuracy = accuracy_score(actual,pred)

        return recall, f1_s, accuracy
    
    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column],axis=1)
        test_y = test_data[[self.config.target_column]]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme # https:

        with mlflow.start_run():
            pred_y = model.predict(test_x)

            (recall, f1_sc, accuracy) = self.eval_metrics(test_y, pred_y)

            # saving metrics as local
            scores = {"Recall_score": recall, "f1_score": f1_sc, "Accuracy":accuracy}
            save_json(path=Path(self.config.metric_file_name),data=scores)

            mlflow.log_params(self.config.all_params) #XGBParams
            mlflow.log_metric("Recall_Score", recall)
            mlflow.log_metric("F1_Score",f1_sc)
            mlflow.log_metric("Accuracy",accuracy)

            if tracking_url_type_store != "file":
                # check if it is a local directory or not
                mlflow.sklearn.log_model(model,"model",registered_model_name="XGBoost")
            else:
                mlflow.sklearn.log_model(model,"model")



