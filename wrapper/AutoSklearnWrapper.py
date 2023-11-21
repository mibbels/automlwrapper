from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor

from AutoMLLibrary import AutoMLLibrary
from AutoSklearnConfig import AutoSklearnConfig

class AutoSklearnWrapper(AutoMLLibrary):
    #---------------------------------------------------------------------------------------------#
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
   
    #---------------------------------------------------------------------------------------------#
    def _initialize_auto_sklearn(self, user_hyperparameters: dict):
        
        self.config = AutoSklearnConfig()

        self.config.map_hyperparameters(user_hyperparameters)

        self.is_initialized = True
    
    #---------------------------------------------------------------------------------------------#
    def data_preprocessing(self, data, target_column):
        X_train, y_train, X_test, y_test = self.split_and_seperate(data, target_column, ratio=0.2, type='pandas')
        return {'X': X_train, 'y': y_train, 'X_test': X_test, 'y_test': y_test}
    
    #---------------------------------------------------------------------------------------------#
    def _train_model(self, data, target_column, user_hyperparameters=None):
        if not self.is_initialized:
            self._initialize_auto_sklearn(user_hyperparameters)

        if self.task_type == 'classification':
            self.model = AutoSklearnClassifier(
                **(self.config.get_params_constructor_by_key('AutoSklearnClassifier') or {})
            )
        elif self.task_type == 'regression':
            self.model = AutoSklearnRegressor(
                **(self.config.get_params_constructor_by_key('AutoSklearnRegressor') or {})
            )

        self.model.fit(
            **(self.data_preprocessing(data, target_column)),
            **(self.config.get_params_fit_by_key('AutoSklearnClassifier' if self._set_task_type == 'classification' 
                                          else 'AutoSklearnRegressor' if self._set_task_type == 'regression'
                                          else None) or {})
            )
    
    