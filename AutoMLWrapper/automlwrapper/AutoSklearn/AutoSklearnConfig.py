from ..Configuration import Configuration

class AutoSklearnConfig(Configuration):
    __slots__ = []
    #---------------------------------------------------------------------------------------------#
    def __init__(self, config_path: str):
        super().__init__(config_path)

    #---------------------------------------------------------------------------------------------#
    def get_params_constructor_by_key(self, key):
        params =  self.map_user_params('constructor', model_type=key, user_hyperparameters=self.user_hyperparameters)
        """
            memory_limit default 3072 MB produces error.
            n_jobs < 5 produces error.
        """

        if params.get('n_jobs', 0) < 5 and params.get('memory_limit', 0) < 6000:
            params['memory_limit'] = 6000
            params['n_jobs'] = 5

        return params
    #---------------------------------------------------------------------------------------------#
    def get_params_fit_by_key(self, key):
        params = self.map_user_params(func_type='fit', model_type=key, user_hyperparameters=self.user_hyperparameters)        

        return params