try:
    from autogluon.tabular import TabularPredictor
    from autogluon.multimodal import MultiModalPredictor
    from autogluon.timeseries import TimeSeriesPredictor
except ImportError as e:
    print(f"WARNING AutoGluon could not be mported. It might not b available in this environment. Err: \n {e}.")


import os
import yaml
import mlflow.pyfunc
from datetime import datetime
import pandas as pd
from PIL import Image
import numpy as np

from ..AutoMLLibrary import AutoMLLibrary
from ..ModelInfo import ModelInfo
from ..PyfuncModel import PyfuncModel
from .AutoGluonConfig import AutoGluonConfig

class AutoGluonWrapper(AutoMLLibrary):
    __slots__ = ['autogluon_problem_type']
    #---------------------------------------------------------------------------------------------#
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.autogluon_problem_type = None
        self.config = AutoGluonConfig(os.path.join(os.path.dirname(__file__), 'AutoGluonConfig.yaml'))
        if not os.path.exists(os.path.join(os.getcwd(), 'AutoMLOutput')):
            os.makedirs(os.path.join(os.getcwd(), 'AutoMLOutput'))
        self.output_path = os.path.join(os.getcwd(),
                                         f'AutoMLOutput/autogluon{datetime.timestamp(datetime.now())}')
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
                else:
                    raise Exception(f'Unknown problem type {self.problem_type} for {self.task_type}')
            
            elif self.task_type == 'few-shot-classification':
                if self.problem_type == 'binary':
                    self.autogluon_problem_type = 'few_shot_classification'
                elif self.problem_type == 'multiclass':
                    self.autogluon_problem_type = 'few_shot_classification'
                else:
                    raise Exception(f'Unknown problem type {self.problem_type} for {self.task_type}')
            
            elif self.task_type == 'zero-shot-classification':
                if self.problem_type == 'binary':
                    self.autogluon_problem_type = 'zero_shot_image_classification'
                elif self.problem_type == 'multiclass':
                    self.autogluon_problem_type = 'zero_shot_image_classification'
                else:
                    raise Exception(f'Unknown problem type {self.problem_type} for {self.task_type}')
            
            elif self.task_type == 'regression':
                if self.problem_type == 'regression':
                    self.autogluon_problem_type = 'regression'
                else:
                    raise Exception(f'Unknown problem type {self.problem_type} for {self.task_type}')
            
            elif self.task_type == 'object-detection':
                if self.problem_type == 'object-detection':
                    self.autogluon_problem_type = 'object_detection'
                else:
                    raise Exception(f'Unknown problem type {self.problem_type} for {self.task_type}')

            elif self.task_type == 'zero-shot-object-detection':
                if self.problem_type == 'object-detection': 
                    self.autogluon_problem_type = 'open_vocabulary_object_detection'
                else:
                    raise Exception(f'Unknown problem type {self.problem_type} for {self.task_type}')
            
            elif self.task_type == 'similarity':
                if self.problem_type == 'similarity':
                    self.autogluon_problem_type = 'image_similarity'
                else:
                    raise Exception(f'Unknown problem type {self.problem_type} for {self.task_type}')
            
            elif self.task_type == 'segmentation':
                if self.problem_type == 'segmentation':
                    self.autogluon_problem_type = 'semantic_segmentation'
                else:
                    raise Exception(f'Unknown problem type {self.problem_type} for {self.task_type}')

        elif self.data_type == 'text':
            if self.task_type == 'classification':
                if self.problem_type == 'binary':
                    self.autogluon_problem_type = 'binary'
                elif self.problem_type == 'multiclass':
                    self.autogluon_problem_type = 'multiclass'
                elif self.problem_type == 'few-shot':
                    self.autogluon_problem_type = 'few_shot_classification'
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
    def data_preprocessing(self, data, target_column = 'label'):
        
        if self.custom_data_preprocessing_func is not None:
            data = self.custom_data_preprocessing_func(data)
            return data
        
        # if self.autogluon_problem_type in ['semantic_segmentation']:
        #     if data.iloc[0][target_column].endswith(('png', 'jpg', 'jpeg')):
        #         path = os.path.dirname(data.iloc[0][target_column])
        #         save = os.path.join(os.path.dirname(path), 'mask_conv')

        #         # sort data[target_column] to ensure that the masks are in the same order as the images
        #         data[target_column] = data[target_column].sort_values().reset_index(drop=True)
                
        #         edited_masks = self._prepare_masks(data[target_column], save)

        #         data[target_column] = edited_masks

        return data
    
    #---------------------------------------------------------------------------------------------#
    def _train_model(self, data, val_data,  target_column, user_hyperparameters: dict = {}):
        
        if type(data) not in [pd.DataFrame]:
            raise ValueError(f'data must be of type pandas DataFrame, but got {type(data)}')        
        
        self.config.map_hyperparameters(user_hyperparameters)
        self._map_problem_type()
        
        data = self.data_preprocessing(data, target_column)
        val_data = self.data_preprocessing(val_data, target_column)
            
        if self.autogluon_problem_type in ['semantic_segmentation']:
            sample_data = data.iloc[0][target_column]
        else:
            sample_data = data
                
        if self.data_type == 'image' or self.data_type == 'text':
            self.model = MultiModalPredictor(
                sample_data_path=sample_data,
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
        

        if self.autogluon_problem_type in ['open_vocabulary_object_detection', 'zero_shot_image_classification']:
            self.eval_output = self._handle_zero_shot(data, target_column)
            return self.eval_output
            
        
        self.fit_output = self.model.fit(
            train_data=data,
            tuning_data=val_data,
            **(self.config.get_params_fit_by_key('TimeSeriesPredictor' if self.data_type == 'timeseries' 
                                          else 'TabularPredictor' if self.data_type == 'tabular' 
                                          else 'MultiModalPredictor') or {})
        )

    #---------------------------------------------------------------------------------------------#
    def _evaluate_model(self, test_data, target_column, **kwargs):
        if self.autogluon_problem_type in ['open_vocabulary_object_detection', 'zero_shot_image_classification']:
            print('Zero-shot models will not be evaluated. The predictions from the training data have been returned.')
            return self.eval_output
        
        self.eval_output = self.model.evaluate(
            data = self.data_preprocessing(test_data, target_column),    
            **kwargs
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
    def _get_info_from_eval(self):
        pass

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
    def _handle_zero_shot(self, data, target_column):
        print('Zero-shot models will not be trained. The Model will now make predictions on the data and return the results.')
        
        targets = self.config.get_params_predict_by_key('TimeSeriesPredictor' if self.problem_type == 'timeseries' 
                                          else 'TabularPredictor' if self.problem_type == 'tabular' 
                                          else 'MultiModalPredictor')['candidate_data'] 
        
        targets = self._convert_to_zero_shot_prompt(targets, 
                                                    classification=True if self.autogluon_problem_type in ['zero_shot_image_classification'] else False)  

        label_col = 'prompt' if self.autogluon_problem_type == 'zero_shot_image_classification' else 'text'       
           
        if target_column in data.columns:
            data = data.drop(columns=[target_column])
        else:
            pass      

        data[label_col] = [targets] * len(data)

        if self.autogluon_problem_type == 'zero_shot_image_classification':
            ev = self.model.predict(
                data['image'].to_list(),
                {"text": targets},
                as_pandas=True
                )
            
        elif self.autogluon_problem_type == 'open_vocabulary_object_detection':
            ev = self.model.predict(data, as_pandas=True)
        
        return ev

    #---------------------------------------------------------------------------------------------#
    def _convert_to_zero_shot_prompt(self, targets, classification=False):
        
        # return a list of targets
        if classification:
            if isinstance(targets, list):
                return targets 
            elif isinstance(targets, str):
                if ',' in targets:
                    return targets.split(',')
                elif '.' in targets:
                    return targets.split('.')
                else:
                    return [targets]
        
        #return a period-separated string
        else:
            if isinstance(targets, list):
                return ['. '.join(targets)]
            elif isinstance(targets, str):
                if ',' in targets:
                    return [targets.replace(',', '. ')]
                else:
                    return [targets]
    
    # #---------------------------------------------------------------------------------------------#
    # def _prepare_masks(self, mask_series, converted_mask_path):
    
    #     def convert_and_binarize(read_path, save_path):
    #         with Image.open(read_path) as img:
    #             img = img.convert('L')
    #             img_array = np.array(img)
    #             binary_array = np.where(img_array > 0, 1, 0)
    #             binary_img = Image.fromarray(binary_array.astype(np.uint8) * 255, 'L')
    #             binary_img.save(save_path, 'PNG')

    #     if not os.path.exists(converted_mask_path):
    #         os.makedirs(converted_mask_path)
    #     else:
    #         for file in os.listdir(converted_mask_path):
    #             os.remove(os.path.join(converted_mask_path, file))
        
    #     new_masks = []
    #     for fullfile in mask_series:
    #         if fullfile.endswith('.png') or fullfile.endswith('.jpg'):
    #             filename = os.path.basename(fullfile)
    #             save_path = os.path.join(converted_mask_path, filename)
    #             convert_and_binarize(fullfile, save_path)
    #             new_masks.append(save_path)
        
    #     return new_masks