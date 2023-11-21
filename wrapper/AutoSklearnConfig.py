from Configuration import Configuration

class AutoSklearnConfig:
    #---------------------------------------------------------------------------------------------#
    def __init__(self):
        config_file_path = 'AutoSklearnConfig.yaml'
        self.conf = Configuration(config_file_path)

    #---------------------------------------------------------------------------------------------#
    def map_hyperparameters(self, user_hyperparameters: dict):
        self.user_hyperparameters = user_hyperparameters

    #---------------------------------------------------------------------------------------------#
    def get_params_constructor_by_key(self, key):
        return self.conf.map_user_params('constructor', model_type=key, user_hyperparameters=self.user_hyperparameters)

    #---------------------------------------------------------------------------------------------#
    def get_params_fit_by_key(self, key):
        return self.conf.map_user_params(func_type='fit', model_type=key, user_hyperparameters=self.user_hyperparameters)