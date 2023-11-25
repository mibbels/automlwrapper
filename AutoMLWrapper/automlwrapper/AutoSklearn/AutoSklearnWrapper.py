from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor

import os
from datetime import datetime

from ..AutoMLLibrary import AutoMLLibrary
from .AutoSklearnConfig import AutoSklearnConfig

class AutoSklearnWrapper(AutoMLLibrary):
    __slots__ = []
    #---------------------------------------------------------------------------------------------#
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = AutoSklearnConfig(os.path.join(os.path.dirname(__file__), 'AutoSklearnConfig.yaml'))
        self.output_path = os.path.join(os.path.dirname(__file__),
                                         f'../output/autosklearn/{datetime.timestamp(datetime.now())}')

    #---------------------------------------------------------------------------------------------#
    def data_preprocessing(self, data, target_column):
        X_train, y_train, X_test, y_test = self.split_and_separate(data, target_column, ratio=0.2, type='pandas')
        return {'X': X_train, 'y': y_train, 'X_test': X_test, 'y_test': y_test}
    
    #---------------------------------------------------------------------------------------------#
    def _train_model(self, data, target_column : str, user_hyperparameters : dict = {}):
    
        self.config.map_hyperparameters(user_hyperparameters)

        if self.task_type == 'classification':
            self.model = AutoSklearnClassifier(
                tmp_folder=self.output_path,
                delete_tmp_folder_after_terminate=False,
                **(self.config.get_params_constructor_by_key('AutoSklearnClassifier') or {})
            )
        elif self.task_type == 'regression':
            self.model = AutoSklearnRegressor(
                tmp_folder=self.output_path,
                delete_tmp_folder_after_terminate=False,
                **(self.config.get_params_constructor_by_key('AutoSklearnRegressor') or {})
            )

        self.model.fit(
            **(self.data_preprocessing(data, target_column)),
            **(self.config.get_params_fit_by_key('AutoSklearnClassifier' if self._set_task_type == 'classification' 
                                          else 'AutoSklearnRegressor' if self._set_task_type == 'regression'
                                          else None) or {})
            )
    
    