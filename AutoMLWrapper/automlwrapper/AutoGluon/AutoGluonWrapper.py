from autogluon.tabular import TabularPredictor
from autogluon.multimodal import MultiModalPredictor
from autogluon.timeseries import TimeSeriesPredictor

import os
import yaml
import mlflow.pyfunc
from datetime import datetime

from ..AutoMLLibrary import AutoMLLibrary
from ..ModelInfo import ModelInfo
from ..PyfuncModel import PyfuncModel
from .AutoGluonConfig import AutoGluonConfig

class AutoGluonWrapper(AutoMLLibrary):
    __slots__ = ['autogluon_problem_type']
    #---------------------------------------------------------------------------------------------#
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = AutoGluonConfig(os.path.join(os.path.dirname(__file__), 'AutoGluonConfig.yaml'))
        self.output_path = os.path.join(os.path.dirname(__file__),
                                         f'../output/autogluon/{datetime.timestamp(datetime.now())}')
    #---------------------------------------------------------------------------------------------#
    def _map_problem_type(self):

        if self.data_type == 'tabular':
            if self.task_type == 'classification':
                if self.problem_type == 'binary':
                    self.autogluon_problem_type = 'binary'
                elif self.problem_type == 'multiclass':
                    self.autogluon_problem_type = 'multiclass'
                else:
                    raise Exception(f'Unknown problem type {self.problem_type} for {self.task_type}')
            
            elif self.task_type == 'regression':
                if self.problem_type == 'regression':
                    self.autogluon_problem_type = 'regression'
                elif self.problem_type == 'quantile':
                    self.autogluon_problem_type = 'quantile'
                else:
                    raise Exception(f'Unknown problem type {self.problem_type} for {self.task_type}')
        
        elif self.data_type == 'image':
            if self.task_type == 'classification':
                if self.problem_type == 'binary':
                    self.autogluon_problem_type = 'binary'
                elif self.problem_type == 'multiclass':
                    self.autogluon_problem_type = 'multiclass'
                elif self.problem_type == 'zero-shot':
                    self.autogluon_problem_type = 'zero_shot_image_classification'
                else:
                    raise Exception(f'Unknown problem type {self.problem_type} for {self.task_type}')
            
            elif self.task_type == 'regression':
                if self.problem_type == 'regression':
                    self.autogluon_problem_type = 'regression'
                else:
                    raise Exception(f'Unknown problem type {self.problem_type} for {self.task_type}')
            
            elif self.task_type == 'object_detection':
                if self.problem_type == 'object_detection':
                    self.autogluon_problem_type = 'object_detection'
                elif self.problem_type == 'zero-shot':
                    self.autogluon_problem_type == 'open_vocabulry_object_detection'
                else:
                    raise Exception(f'Unknown problem type {self.problem_type} for {self.task_type}')
            
            elif self.task_type == 'similarity':
                if self.problem_type == 'similarity':
                    self.autogluon_problem_type = 'image_similarity'
                else:
                    raise Exception(f'Unknown problem type {self.problem_type} for {self.task_type}')
        
        elif self.data_type == 'text':
            if self.task_type == 'classification':
                if self.problem_type == 'binary':
                    self.autogluon_problem_type = 'binary'
                elif self.problem_type == 'multiclass':
                    self.autogluon_problem_type = 'multiclass'
                elif self.problem_type == 'few-shot':
                    self.autogluon_problem_type = 'few_shot_text_classification'
                else:
                    raise Exception(f'Unknown problem type {self.problem_type} for {self.task_type}')
            
            elif self.task_type == 'similarity':
                if self.problem_type == 'similarity':
                    self.autogluon_problem_type = 'text_similarity'
                else:
                    raise Exception(f'Unknown problem type {self.problem_type} for {self.task_type}')
        
        elif self.data_type == 'timeseries':
            if self.task_type == 'forecast':
                if self.problem_type == 'forecast':
                    pass
        
        else:
            raise Exception(f'Unknown data type {self.data_type}')
    
     #---------------------------------------------------------------------------------------------#
    def data_preprocessing(self, data, target_column):
        
        if self.custom_data_preprocessing_func is not None:
            data = self.custom_data_preprocessing_func(data)
            return data
        
        # done automatically by autogluon
        #if self.data_type == 'tabular':
        #    train, test = self.split(data, target_column, type='pandas')
        #    return {'train_data': train, 'tuning_data': test}

        return {'train_data': data}
    
    #---------------------------------------------------------------------------------------------#
    def _train_model(self, data, target_column, user_hyperparameters: dict = {}):
        
        self.config.map_hyperparameters(user_hyperparameters)
        self._map_problem_type()
        
        if self.data_type == 'image' or self.data_type == 'text':
            self.model = MultiModalPredictor(
                label=target_column,
                path=self.output_path,
                problem_type=self.autogluon_problem_type,
                **(self.config.get_params_constructor_by_key('MultiModalPredictor') or {})
            )
            self.log_model_type = self.config._get_mlflow_details('MultiModalPredictor').get('__log_model_type', {})

        elif self.data_type == 'tabular':
            self.model = TabularPredictor(
                label=target_column,
                path=self.output_path,
                problem_type=self.autogluon_problem_type,
                **(self.config.get_params_constructor_by_key('TabularPredictor') or {})
            )
            self.log_model_type = self.config._get_mlflow_details('TabularPredictor').get('__log_model_type', {})

        elif self.data_type == 'timeseries':
            self.model = TimeSeriesPredictor(
                target=target_column,
                path=self.output_path,
                **(self.config.get_params_constructor_by_key('TimeSeriesPredictor') or {})
            )
            self.log_model_type = self.config._get_mlflow_details('TimeSeriesPredictor').get('__log_model_type', {})
        
        self.fit_output = self.model.fit(
            **(self.data_preprocessing(data, target_column)),
            **(self.config.get_params_fit_by_key('TimeSeriesPredictor' if self.problem_type == 'timeseries' 
                                          else 'TabularPredictor' if self.problem_type == 'tabular' 
                                          else 'MultiModalPredictor') or {})
        )

    #---------------------------------------------------------------------------------------------#
    def _create_model_info(self, n_models: int = 1):

        best_models_info = []
        for i in range(n_models):
            if self.data_type in ['tabular', 'timeseries']:
                model_info = ModelInfo(
                    **(self._get_info_from_fit_summary(i) or {})
                )
            elif self.data_type in ['image', 'text']:
                model_info = ModelInfo(
                    **(self._get_info_from_config_yaml() or {})
                )

            best_models_info.append(model_info)

        return best_models_info
    
    #---------------------------------------------------------------------------------------------#
    def  _get_info_from_fit_summary(self, n_th_model):

        dict_summary = self.model.fit_summary(0)
        df_leaderboard = self.model.leaderboard(silent=True)
        try:
            model_name = df_leaderboard.loc[n_th_model, 'model']
        except KeyError:
            raise ValueError(f'No more than {n_th_model-1} models found. Please try at max {len(df_leaderboard)} models.')
                    
        model_info_args = {
            'model_name': dict_summary['model_types'].get(model_name, None),
            'model_library': 'autogluon',
            'model_object': PyfuncModel('autogluon', self.model),
            'model_path': self.output_path + 'models/',
            'model_params_dict': dict_summary['model_hyperparams'].get(model_name, {}),
            'model_metrics_dict': {
                'val_score': df_leaderboard.loc[df_leaderboard['model'] == model_name, 'score_val'].values[0]
            },
            'model_tags_dict': {},
        }

        return model_info_args
        
      
    #---------------------------------------------------------------------------------------------#
    def _get_info_from_config_yaml(self):
        with open(self.output_path + '/config.yaml') as file:
            out_config = yaml.load(file, Loader=yaml.FullLoader)
        
        with open(self.output_path + '/hparams.yaml') as file:
            hparams = yaml.load(file, Loader=yaml.FullLoader)
        
        model_name = out_config['model'].get('names', ['neural net ' + self.data_type + ' ' + self.task_type])[0]
        
        model_info_args = {
            'model_name': model_name,
            'model_library': 'autogluon',
            'model_object': PyfuncModel('autogluon', self.model),
            'model_path': self.output_path + 'models/',
            'model_params_dict': {k: v for k, v in hparams.items() if v is not None}, #out_config.get('optimization', {}),
            'model_metrics_dict': self.model.fit_summary(0),
            'model_tags_dict': {'checkpoint_name': out_config['model'].get(model_name, {}).get('checkpoint_name', '')},
        }

        return model_info_args

    #---------------------------------------------------------------------------------------------#
    def _define_pyfunc_model(self):
        
        class autogluon_model(mlflow.pyfunc.PythonModel):
            def __init__(self, model):
                self.model = model

            def predict(self, context, model_input):
                return self.model.predict(model_input)

        return autogluon_model(self.model)        
        
    
