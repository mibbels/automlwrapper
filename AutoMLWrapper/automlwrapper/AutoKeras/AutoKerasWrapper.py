import autokeras as ak

from datetime import datetime
import os
from ..AutoMLLibrary import AutoMLLibrary
from .AutoKerasConfig import AutoKerasConfig


class AutoKerasWrapper(AutoMLLibrary):
    __slots__ = []
    #---------------------------------------------------------------------------------------------#
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = AutoKerasConfig(os.path.join(os.path.dirname(__file__), 'AutoKerasConfig.yaml'))
        self.output_path = os.path.join(os.path.dirname(__file__),
                                         f'../output/autokeras/{datetime.timestamp(datetime.now())}')
        
    #---------------------------------------------------------------------------------------------#
    def data_preprocessing(self, data, target):

        if self.custom_data_preprocessing_func is not None:
            data = self.custom_data_preprocessing_func(data)
            return data
        
        if self.data_type == 'tabular' or self.data_type == 'timeseries':
            x, y = self.separate(data, target, type='pandas')

        elif self.data_type == 'image' or self.data_type == 'text':
            x,y = self.separate(data, target, type='numpy')
        
        return {'x': x, 'y': y}
    
    #---------------------------------------------------------------------------------------------#
    def _train_model(self, data, target_column, user_hyperparameters: dict = {}):
        
        self.config.map_hyperparameters(user_hyperparameters)

        if self.data_type == 'text':
            self._train_text(data, target_column, user_hyperparameters)
        elif self.data_type == 'tabular':
            self._train_structured(data, target_column, user_hyperparameters)
        elif self.data_type == 'image':
            self._train_image(data, target_column, user_hyperparameters)


    def _get_params(self, func_type: str, model_type: str):
        return self.config.get_params(func_type, model_type)
    
    #---------------------------------------------------------------------------------------------#
    def _train_structured(self, data, target_column, user_hyperparameters: dict = {}):
        if self.task_type == "classification":
            self.model = ak.StructuredDataClassifier(
                directory=self.output_path,
                **(self.config.get_params_constructor_by_key('StructuredDataClassifier') or {})
            )
            self.log_model_type = self.config._get_mlflow_details('StructuredDataClassifier').get('__log_model_type', {})

        elif self.task_type == "regression":
            self.model = ak.StructuredDataRegressor(
                directory=self.output_path,
                **(self.config.get_params_constructor_by_key('StructuredDataRegressor') or {})
            )
            self.log_model_type = self.config._get_mlflow_details('StructuredDataRegressor').get('__log_model_type', {})

        self.model.fit(
            **(self.data_preprocessing(data, target_column)),
            **(self.config.get_params_fit_by_key('StructuredDataClassifier' if self.task_type == 'classification' 
                                          else 'StructuredDataRegressor' if self.task_type == 'regression'
                                          else None) or {})
            )


    #---------------------------------------------------------------------------------------------#
    def _train_image(self, data, target_column, user_hyperparameters: dict = {}):
        if self.task_type == "classification":
            self.model = ak.ImageClassifier(
                directory=self.output_path,
                **(self.config.get_params_constructor_by_key('ImageClassifier') or {})
            )
            self.log_model_type = self.config._get_mlflow_details('ImageClassifier').get('__log_model_type', {})

        elif self.task_type == "regression":
            self.model = ak.ImageRegressor(
                directory=self.output_path,
                **(self.config.get_params_constructor_by_key('ImageRegressor') or {})
            )
            self.log_model_type = self.config._get_mlflow_details('ImageRegressor').get('__log_model_type', {})

        self.model.fit(
            **(self.data_preprocessing(data, target_column)),
            **(self.config.get_params_fit_by_key('ImageClassifier' if self.task_type == 'classification' 
                                          else 'ImageRegressor' if self.task_type == 'regression'
                                          else None) or {})
            )
                    
    #---------------------------------------------------------------------------------------------#
    def _train_text(self, data, target_column, user_hyperparameters: dict = {}):
        if self.task_type == "classification":
            self.model = ak.TextClassifier(
                directory=self.output_path,
                **(self.config.get_params_constructor_by_key('TextClassifier') or {})
            )
            self.log_model_type = self.config._get_mlflow_details('TextClassifier').get('__log_model_type', {})

        elif self.task_type == "regression":
            self.model = ak.TextRegressor(
                **(self.config.get_params_constructor_by_key('TextRegressor') or {})
            )
            self.log_model_type = self.config._get_mlflow_details('TextRegressor').get('__log_model_type', {})

        self.model.fit(
            **(self.data_preprocessing(data, target_column)),
            **(self.config.get_params_fit_by_key('TextClassifier' if self.task_type == 'classification' 
                                          else 'TextRegressor' if self.task_type == 'regression'
                                          else None) or {})
            )

    #---------------------------------------------------------------------------------------------#
    def _train_timeseries(self, data, target_column, user_hyperparameters: dict = {}):
        self.model = ak.TimeseriesForecaster(
            directory=self.output_path,
            **(self.config.get_params_constructor_by_key('TimeseriesForecaster') or {})
        )
        self.log_model_type = self.config._get_mlflow_details('TimeseriesForecaster').get('__log_model_type', {})

        self.model.fit(
            **(self.data_preprocessing(data, target_column)),
            **(self.config.get_params_fit_by_key('TimeseriesForecaster') or {})
            )
