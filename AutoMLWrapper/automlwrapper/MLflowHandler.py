import mlflow, mlflow.sklearn, mlflow.keras, mlflow.pyfunc, mlflow.gluon
from typing import Dict, Any
from .ModelInfo import ModelInfo

class MLflowHandler:
    #---------------------------------------------------------------------------------------------#
    def __init__(self, model_info: ModelInfo, user_tags: Dict[str, Any], output_path: str):
        self.model_info = model_info
        self.user_tags = user_tags
        self.output_path = output_path

    #---------------------------------------------------------------------------------------------#
    def log_to_mlflow(self):
        with mlflow.start_run() as run:
            for key, value in self.user_tags.items():
                mlflow.set_tag(key, str(value))

            model_info_dict = self.model_info.to_dict()

            for key, value in model_info_dict['tags'].items():
                mlflow.set_tag(key, str(value))
            for key, value in model_info_dict['params'].items():
                mlflow.log_param(key, value)
            for key, value in model_info_dict['metrics'].items():
                mlflow.log_metric(key, value)

            if self.model_info.model_type == 'sklearn':
                mlflow.sklearn.log_model(self.model_info.model_object, "model")
            elif self.model_info.model_type == 'pyfunc':
                mlflow.pyfunc.log_model("model", python_model=self.model_info.model_object)
            elif self.model_info.model_type == 'keras':
                mlflow.keras.log_model(self.model_info.model_object, "model")
            elif self.model_info.model_type == 'gluon':
                mlflow.gluon.log_model(self.model_info.model_object, "model")
            else:
                raise ValueError(f"Unknown model type {self.model_info.model_type}")