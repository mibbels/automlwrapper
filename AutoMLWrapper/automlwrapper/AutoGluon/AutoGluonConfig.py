from ..Configuration import Configuration

class AutoGluonConfig(Configuration):
    __slots__ = []
    #---------------------------------------------------------------------------------------------#
    def __init__(self, config_path: str):
        super().__init__(config_path)

    #---------------------------------------------------------------------------------------------#
    def get_params_predict_by_key(self, key):
        return self.map_user_params('predict', model_type=key, user_hyperparameters=self.user_hyperparameters)
    

    #---------------------------------------------------------------------------------------------#
    def get_params_constructor_by_key(self, key):
        return self.map_user_params('constructor', model_type=key, user_hyperparameters=self.user_hyperparameters)

    #---------------------------------------------------------------------------------------------#
    def get_params_fit_by_key(self, key):
        params = self.map_user_params(func_type='fit', model_type=key, user_hyperparameters=self.user_hyperparameters)
        
        """
            hyperparameter_tune_kwargs for TabularPredictor requires scheduler and searcher.
            hyperparameter_tune_kwargs for MultiModalPredictor, TimeSeriesPredictor would require concrete search space.
        """

        if key in ['MultiModalPredictor', 'TimeSeriesPredictor']:
            try:
                params['hyperparameter_tune_kwargs'].pop('num_trials', None)
            except:
                pass
        elif key in ['TabularPredictor']:
            try:
               params['hyperparameter_tune_kwargs']['scheduler'] = 'local'
               params['hyperparameter_tune_kwargs']['searcher'] = 'auto'
 
            except:
                pass
        return params