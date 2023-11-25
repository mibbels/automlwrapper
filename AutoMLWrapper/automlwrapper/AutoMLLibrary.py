import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple

class AutoMLLibrary:
    __slots__ = ['model', 'config', 'task_type', 'data_type', 'problem_type', 'is_initialized', 'output_path', 'custom_data_preprocessing_func']
    #---------------------------------------------------------------------------------------------#
    def __init__(self, **kwargs: str) -> None:
        self.model =            None
        self.config =           None
        self.task_type =        None
        self.data_type =        None
        self.problem_type =     None
        self.is_initialized =   False
        self.output_path =      None
        self.custom_data_preprocessing_func = None
        
    
    #---------------------------------------------------------------------------------------------#
    def _set_out_path(self, out_path: str) -> None:
        self.output_path = out_path
    
    #---------------------------------------------------------------------------------------------#
    def _set_custom_data_preprocessing(self, custom_preprocessing_func):
        self.custom_data_preprocessing_func = custom_preprocessing_func
  
    #---------------------------------------------------------------------------------------------#
    def _set_data_type(self, data_type: str) -> None:
        self.data_type = data_type
    
    #---------------------------------------------------------------------------------------------#
    def _infer_data_type(self, data: pd.DataFrame, target_column: str) -> None:
        if self.data_type is not None:
            return        
        ### TODO ###
    
    #---------------------------------------------------------------------------------------------#
    def _set_problem_type(self, problem_type: str) -> None:
        self.problem_type = problem_type
    
    #---------------------------------------------------------------------------------------------#
    def _infer_problem_type(self, data: pd.DataFrame, target_column: str) -> None:
        if self.problem_type is not None:
            return        
        ### TODO ###

    #---------------------------------------------------------------------------------------------#
    def _set_task_type(self, task_type: str) -> None:
        self.task_type = task_type

    #---------------------------------------------------------------------------------------------#
    def _infer_task_type(self, data: pd.DataFrame, target_column: str) -> None: 
        if self.task_type is not None:
            return  # Use the explicitly set task type

        target_values = data[target_column]

        if target_values.dtype in (int, float):
            self.task_type = "regression"
        elif target_values.dtype == object and len(target_values.unique()) <= 2:
            self.task_type = "classification"
        else:
            self.task_type = "classification"
    
    #---------------------------------------------------------------------------------------------#
    def _infer_all(self, data: pd.DataFrame, target_column: str) -> None:
        self._infer_task_type(data, target_column)
        self._infer_data_type(data)
        self._infer_problem_type(data)

    
    #---------------------------------------------------------------------------------------------#
    def split(self, data, target, ratio: float = 0.2, type = 'pandas'):

        if type == 'pandas':
            return self._split_test_train_df(data, ratio)
        elif type == 'numpy':
            return self._split_test_train_np(data, ratio)
        else:
            raise Exception(f'Unknown type {type}')

    #---------------------------------------------------------------------------------------------#
    def separate(self, data, target, type = 'pandas'):
        if type == 'pandas':
            return self._separate_x_y_df(data, target)
        elif type == 'numpy':
            return self._separate_x_y_np(data, target)
        else:
            raise Exception(f'Unknown type {type}')

    #---------------------------------------------------------------------------------------------#     
    def split_and_separate(self, data, target, ratio: float = 0.2, type = 'pandas'):
        if type == 'pandas':
            train_data, test_data = self._split_test_train_df(data, ratio)
            X_train, y_train = self._separate_x_y_df(train_data, target_column=target)
            X_test, y_test = self._separate_x_y_df(test_data, target_column=target)
            return X_train, y_train, X_test, y_test
        
        elif type == 'numpy':
            train_data, test_data = self._split_test_train_np(data, ratio)
            X_train, y_train = self._separate_x_y_np(train_data, target_column=target)
            X_test, y_test = self._separate_x_y_np(test_data, target_column=target)
            return X_train, y_train, X_test, y_test
        else:
            raise Exception(f'Unknown type {type}')

    #---------------------------------------------------------------------------------------------#
    def _split_test_train_df(self, data: pd.DataFrame, ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_data, test_data = train_test_split(data, test_size=ratio)
        return train_data, test_data

    #---------------------------------------------------------------------------------------------#
    def _separate_x_y_df(self, data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        return X, y   

    #---------------------------------------------------------------------------------------------#
    def _split_test_train_np(self, data: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
        return train_data, test_data

    #---------------------------------------------------------------------------------------------#
    def _separate_x_y_np(self, data: np.ndarray, target_column_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        X = np.delete(data, target_column_idx, axis=1)
        y = data[:, target_column_idx]
        return X, y     