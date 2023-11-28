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
    def _data_preprocessing(self, data, target_column):
        
        if self.custom_data_preprocessing_func is not None:
            data = self.custom_data_preprocessing_func(data)
            return data
        
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
            self.log_model_type = self.config._get_mlflow_details('AutoSklearnClassifier').get('__log_model_type', {})

        elif self.task_type == 'regression':
            self.model = AutoSklearnRegressor(
                tmp_folder=self.output_path,
                delete_tmp_folder_after_terminate=False,
                **(self.config.get_params_constructor_by_key('AutoSklearnRegressor') or {})
            )
            self.log_model_type = self.config._get_mlflow_details('AutoSklearnRegressor').get('__log_model_type', {})

        self.fit_output = self.model.fit(
            **(self._data_preprocessing(data, target_column)),
            **(self.config.get_params_fit_by_key('AutoSklearnClassifier' if self._set_task_type == 'classification' 
                                          else 'AutoSklearnRegressor' if self._set_task_type == 'regression'
                                          else None) or {})
            )
    
    
    #---------------------------------------------------------------------------------------------#
    def _mlflow_ready_output(self, n: int = 1):
        df_leaderboard = self.fit_output.leaderboard(detailed=True)
        df_leaderboard.insert(0, 'model_id', df_leaderboard.index)
        df_leaderboard.reset_index(drop=True, inplace=True)       

        dict_model_info = self.fit_output.show_models()

        best_models_info = []
        for i in range(n):
            try:
                model_id = df_leaderboard.loc[i, 'model_id']
            except KeyError:
                raise ValueError(f'No {n} models found. Please try at max {len(df_leaderboard)} models.')
                
            model_info = self._get_info_by_id(model_id, df_leaderboard, dict_model_info, flat=False)
            best_models_info.append(model_info)

        return best_models_info

    #---------------------------------------------------------------------------------------------#
    def _get_info_by_id(self, model_id, leaderboard, model_info, flat: bool = False):
        model_info_by_id = model_info[model_id]
        model_loss = model_info_by_id['cost']
        model_object = model_info_by_id['sklearn_classifier']
        model_name = model_object.__class__.__name__
        model_params = model_object.get_params()
        model_data_preprocessor_name = leaderboard.loc[leaderboard['model_id'] == model_id, 'data_preprocessors'].values[0]
        model_feature_preprocessor_name = leaderboard.loc[leaderboard['model_id'] == model_id, 'feature_preprocessors'].values[0]
   
        if flat:
            info_by_id = {
                'model_id': model_id,
                'model_loss': model_loss,
                'model_object': model_object,
                'model_name': model_name,
                'model_params': model_params,
                'model_data_preprocessor_name': model_data_preprocessor_name,
                'model_feature_preprocessor_name': model_feature_preprocessor_name,
            }
        else:
            info_by_id = {
                'tags': {
                    'model_id': model_id,
                    'model_name': model_name,
                    'model_data_preprocessor_name': model_data_preprocessor_name,
                    'model_feature_preprocessor_name': model_feature_preprocessor_name,
                },
                'params': model_params,
                'metrics': {
                    'model_loss': model_loss,
                },
                'artifact': model_object,
                'model': model_object,
            }

        return info_by_id