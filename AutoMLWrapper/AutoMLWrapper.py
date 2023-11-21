from typing import Any, Optional, Dict



from .AutoGluon.AutoGluonWrapper import AutoGluonWrapper
from .AutoKeras.AutoKerasWrapper import AutoKerasWrapper
from .AutoSklearn.AutoSklearnWrapper import AutoSklearnWrapper

class AutoMLWrapper:
    #---------------------------------------------------------------------------------------------#
    def __init__(self, library_name: str) -> None:
        self.library = None
        self.library_name = library_name
        self.is_initialized = False

    #---------------------------------------------------------------------------------------------#
    def initialize(self, data: Any, target_column: str, task_type: Optional[str] = None, data_type: Optional[str] = None, problem_type: Optional[str] = None) -> None:
        
        if self.library_name == "auto-sklearn":
            self.library = AutoSklearnWrapper()

        elif self.library_name == "auto-keras":
            self.library = AutoKerasWrapper()

        elif self.library_name == "autogluon":
            self.library = AutoGluonWrapper()

        else:
            raise ValueError("Invalid library name")

        if task_type is not None:
            self.library._set_task_type(task_type)
        else:
            if data:
                self.library._infer_task_type(data, target_column)
        
        if data_type is not None:
            self.library._set_data_type(data_type)
        else:
            if data is not None:
                self.library._infer_data_type(data, target_column)

        if problem_type is not None:
            self.library._set_problem_type(problem_type)
        else:
            if data is not None:
                self.library._infer_problem_type(data, target_column)

        if data is not None:
             self.is_initialized = True
        else:
            self.is_initialized = False

    #---------------------------------------------------------------------------------------------#
    def train(self, data: Any, target_column: str, task_type: Optional[str] = None, data_type: Optional[str] = None, 
              problem_type: Optional[str] = None, hyperparameters: Optional[Dict[str, Any]] = {}) -> None:
        if not self.is_initialized:
            self.initialize(data, target_column, task_type, data_type, problem_type)

        self.library._train_model(data, target_column, hyperparameters)

    #---------------------------------------------------------------------------------------------#
    def evaluate(self, test_data: Any) -> float:
        if self.library is None:
            raise ValueError("You must call 'train' before 'evaluate'")
        
        return self.library.evaluate(test_data)