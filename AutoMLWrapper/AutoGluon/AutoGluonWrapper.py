from autogluon.tabular import TabularPredictor
from autogluon.multimodal import MultiModalPredictor
from autogluon.timeseries import TimeSeriesPredictor

import os
from datetime import datetime
from ..AutoMLLibrary import AutoMLLibrary
from .AutoGluonConfig import AutoGluonConfig

class AutoGluonWrapper(AutoMLLibrary):
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
                if self.problem_type == 'zero-shot':
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

        elif self.data_type == 'tabular':
            self.model = TabularPredictor(
                label=target_column,
                path=self.output_path,
                problem_type=self.autogluon_problem_type,
                **(self.config.get_params_constructor_by_key('TabularPredictor') or {})
            )

        elif self.data_type == 'timeseries':
            self.model = TimeSeriesPredictor(
                target=target_column,
                path=self.output_path,
                **(self.config.get_params_constructor_by_key('TimeSeriesPredictor') or {})
            )
        
        self.model.fit(
            data,
            **(self.config.get_params_fit_by_key('TimeSeriesPredictor' if self.problem_type == 'timeseries' 
                                          else 'TabularPredictor' if self.problem_type == 'tabular' 
                                          else 'MultiModalPredictor') or {})
        )
