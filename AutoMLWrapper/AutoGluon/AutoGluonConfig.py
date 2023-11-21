from ..Configuration import Configuration

class AutoGluonConfig(Configuration):
    #---------------------------------------------------------------------------------------------#
    def __init__(self, config_path: str):
        super().__init__(config_path)

    #---------------------------------------------------------------------------------------------#
    def map_hyperparameters(self, user_hyperparameters: dict):
        self.user_hyperparameters = user_hyperparameters

    #---------------------------------------------------------------------------------------------#
    def get_params_constructor_by_key(self, key):
        return self.map_user_params('constructor', model_type=key, user_hyperparameters=self.user_hyperparameters)

    #---------------------------------------------------------------------------------------------#
    def get_params_fit_by_key(self, key):
        return self.map_user_params(func_type='fit', model_type=key, user_hyperparameters=self.user_hyperparameters)