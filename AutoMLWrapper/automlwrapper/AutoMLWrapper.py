from typing import Any, Optional, Dict
import mlflow

from .AutoGluon.AutoGluonWrapper import AutoGluonWrapper
from .AutoKeras.AutoKerasWrapper import AutoKerasWrapper
from .AutoSklearn.AutoSklearnWrapper import AutoSklearnWrapper

class AutoMLWrapper:
    __slots__ = ['__library', '__library_name', '__is_initialized', '__out_path']
    __properties__ = ['_library', '_library_name', '_is_initialized', '_out_path']
    #---------------------------------------------------------------------------------------------#
    def __init__(self, library_name: str) -> None:
        self.__library = None
        self.__library_name = library_name
        self.__is_initialized = False
        self.__out_path = None
        
    #---------------------------------------------------------------------------------------------#
    def SetOutputDirectory(self, outputDirectory: str) -> None:
        """Set the output directory for the AutoML process.
        """
        self.__out_path = outputDirectory

    #---------------------------------------------------------------------------------------------#
    def SetCustomDataPreprocessing(self, custom_preprocessing_func):
        """
        Set a custom data preprocessing function that will be applied before model training.
        """
        self.__library._set_custom_data_preprocessing(custom_preprocessing_func)

    #---------------------------------------------------------------------------------------------#
    @property
    def _library(self) -> Any:
        return self.__library
    
    @_library.setter
    def _library(self, _) -> None:
        if self.__library_name == "autosklearn":
            self.__library = AutoSklearnWrapper()

        elif self.__library_name == "autokeras":
            self.__library = AutoKerasWrapper()

        elif self.__library_name == "autogluon":
            self.__library = AutoGluonWrapper()

        else:
            raise ValueError("Invalid library name")
    
    #---------------------------------------------------------------------------------------------#
    @property
    def _library_name(self) -> str:
        return self.__library_name
    
    @_library_name.setter
    def _library_name(self, library_name: str) -> None:
        raise NotImplementedError("This property is read-only")
    
    #---------------------------------------------------------------------------------------------#
    @property
    def _is_initialized(self) -> bool:
        return self.__is_initialized

    @_is_initialized.setter
    def _is_initialized(self, is_initialized: bool) -> None:
        raise NotImplementedError("This property is read-only")

    #---------------------------------------------------------------------------------------------#
    @property
    def _out_path(self) -> str:
        return self.__out_path
    
    @_out_path.setter
    def _out_path(self, out_path: str) -> None:
        ### add validation ??? ###
        self.__out_path = out_path

    #---------------------------------------------------------------------------------------------#
    def Initialize(self, data_sample: Any, target_column: str, task_type: Optional[str] = None, data_type: Optional[str] = None, problem_type: Optional[str] = None) -> None:
        """Initialize the AutoML process. Use for determining the task type, data type, and problem type beforehand.
        """

        if self.__library == None:
            self._library = self.__library_name 

        if task_type is not None:
            self.__library._set_task_type(task_type)
        else:
            if data_sample is not None:
                self.__library._infer_task_type(data_sample, target_column)
        
        if data_type is not None:
            self.__library._set_data_type(data_type)
        else:
            if data_sample is not None:
                self.__library._infer_data_type(data_sample, target_column)

        if problem_type is not None:
            self.__library._set_problem_type(problem_type)
        else:
            if data_sample is not None:
                self.__library._infer_problem_type(data_sample, target_column)

        if self.__out_path is not None:
            self.__library._set_out_path(self.__out_path)

        if data_sample is not None:
             self.__is_initialized = True
        else:
            self.__is_initialized = False

    #---------------------------------------------------------------------------------------------#
    def Train(self, data: Any, target_column: str, task_type: Optional[str] = None, data_type: Optional[str] = None, 
              problem_type: Optional[str] = None, hyperparameters: Optional[Dict[str, Any]] = {}) -> None:
        """ Invoke the underlying AutoML library to train a model.
        """

        if not self.__is_initialized:
            self.Initialize(data, target_column, task_type, data_type, problem_type)

        self.__library._train_model(data, target_column, hyperparameters)

    #---------------------------------------------------------------------------------------------#
    def Evaluate(self, test_data: Any) -> float:
        if self.__library is None:
            raise ValueError("You must call 'train' before 'evaluate'")
        
        pass

    #---------------------------------------------------------------------------------------------#
    def Output(self, nBestModels: int = 1, outputForMLFlow : bool = True):
        if self.__library is None:
            raise ValueError("You must call 'train' before 'output'")

        if outputForMLFlow:
            return self.__library._mlflow_ready_output(nBestModels)
        else:
            pass
    
    #---------------------------------------------------------------------------------------------#
    def MlflowUploadBest(self, user_tags: dict):
        best_model = self.__library._mlflow_ready_output(1)[0]  # assuming the best model is at index 0

        with mlflow.start_run() as run:
            for key, value in user_tags.items():
                mlflow.set_tag(key, str(value))

            for key, value in best_model['tags'].items():
                mlflow.set_tag(key, str(value))
            for key, value in best_model['params'].items():
                mlflow.log_param(key, value)
            for key, value in best_model['metrics'].items():
                mlflow.log_metric(key, value)

            if self.__library.log_model_type == 'sklearn':
                mlflow.sklearn.log_model(best_model['model'], "model")
            elif self.__library.log_model_type == 'pyfunc':
                mlflow.pyfunc.log_model("model", python_model=best_model['model'])
            elif self.__library.log_model_type == 'keras':
                mlflow.keras.log_model(best_model['model'], "model")
            else:
                raise ValueError(f"Unknown model type {self.__library.log_model_type}")

    #---------------------------------------------------------------------------------------------#
    def MlflowUploadTopN(self, n: int, user_tags: dict):
        top_n_models = self.__library._mlflow_ready_output(n)

        for i, model in enumerate(top_n_models):
            with mlflow.start_run() as run:
                for key, value in user_tags.items():
                    mlflow.set_tag(f"{key}_{i+1}", str(value))  # add the model rank to the tag key

                # Log the model's tags, params, and metrics
                for key, value in model['tags'].items():
                    mlflow.set_tag(key, str(value))
                for key, value in model['params'].items():
                    mlflow.log_param(key, value)
                for key, value in model['metrics'].items():
                    mlflow.log_metric(key, value)

                if self.__library.log_model_type == 'sklearn':
                    mlflow.sklearn.log_model(model['model'], "model")
                elif self.__library.log_model_type == 'pyfunc':
                    mlflow.pyfunc.log_model("model", model['model'])
                elif self.__library.log_model_type == 'keras':
                    mlflow.keras.log_model(model['model'], "model")
                else:
                    raise ValueError(f"Unknown model type {self.__library.log_model_type}")