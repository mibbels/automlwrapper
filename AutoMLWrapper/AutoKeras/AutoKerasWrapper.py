import autokeras as ak

import os
from ..AutoMLLibrary import AutoMLLibrary
from .AutoKerasConfig import AutoKerasConfig


class AutoKerasWrapper(AutoMLLibrary):
    #---------------------------------------------------------------------------------------------#
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = AutoKerasConfig(os.path.join(os.path.dirname(__file__), 'AutoKerasConfig.yaml'))
    
    #---------------------------------------------------------------------------------------------#
    def data_preprocessing(self, data, target):
        if self.data_type == 'tabular' or self.data_type == 'timeseries':
            x, y = self.seperate(data, target, type='pandas')

        elif self.data_type == 'image' or self.data_type == 'text':
            x,y = self.seperate(data, target, type='numpy')
        
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
                **(self.config.get_params_constructor_by_key('StructuredDataClassifier') or {})
            )
        elif self.task_type == "regression":
            self.model = ak.StructuredDataRegressor(
                **(self.config.get_params_constructor_by_key('StructuredDataRegressor') or {})
            )

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
                **(self.config.get_params_constructor_by_key('ImageClassifier') or {})
            )
        elif self.task_type == "regression":
            self.model = ak.ImageRegressor(
                **(self.config.get_params_constructor_by_key('ImageRegressor') or {})
            )

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
                **(self.config.get_params_constructor_by_key('TextClassifier') or {})
            )
        elif self.task_type == "regression":
            self.model = ak.TextRegressor(
                **(self.config.get_params_constructor_by_key('TextRegressor') or {})
            )

        self.model.fit(
            **(self.data_preprocessing(data, target_column)),
            **(self.config.get_params_fit_by_key('TextClassifier' if self.task_type == 'classification' 
                                          else 'TextRegressor' if self.task_type == 'regression'
                                          else None) or {})
            )

    #---------------------------------------------------------------------------------------------#
    def _train_timeseries(self, data, target_column, user_hyperparameters: dict = {}):
        self.model = ak.TimeseriesForecaster(
            **(self.config.get_params_constructor_by_key('TimeseriesForecaster') or {})
        )

        self.model.fit(
            **(self.data_preprocessing(data, target_column)),
            **(self.config.get_params_fit_by_key('TimeseriesForecaster') or {})
            )
