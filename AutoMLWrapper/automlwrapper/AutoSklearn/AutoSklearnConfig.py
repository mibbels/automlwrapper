from ..Configuration import Configuration
import autosklearn

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
        
        metric = params.get('evaluation_metric', "")
        if metric == 'accuracy':
            params['evaluation_metric'] = autosklearn.metrics.accuracy
        elif metric == 'balanced_accuracy':
            params['evaluation_metric'] = autosklearn.metrics.balanced_accuracy
        elif metric == 'f1':
            params['evaluation_metric'] = autosklearn.metrics.f1
        elif metric == 'f1_macro':
            params['evaluation_metric'] = autosklearn.metrics.f1_macro
        elif metric == 'f1_micro':
            params['evaluation_metric'] = autosklearn.metrics.f1_micro
        elif metric == 'f1_weighted':
            params['evaluation_metric'] = autosklearn.metrics.f1_weighted
        elif metric == 'roc_auc':
            params['evaluation_metric'] = autosklearn.metrics.roc_auc
        elif metric == 'precision':
            params['evaluation_metric'] = autosklearn.metrics.precision
        elif metric == 'precision_macro':
            params['evaluation_metric'] = autosklearn.metrics.precision_macro
        elif metric == 'precision_weighted':
            params['evaluation_metric'] = autosklearn.metrics.precision_weighted
        elif metric == 'average_precision':
            params['evaluation_metric'] = autosklearn.metrics.average_precision
        elif metric == 'recall':
            params['evaluation_metric'] = autosklearn.metrics.recall
        elif metric == 'recall_macro':
            params['evaluation_metric'] = autosklearn.metrics.recall_macro
        elif metric == 'recall_weighted':
            params['evaluation_metric'] = autosklearn.metrics.recall_weighted
        elif metric == 'log_loss':
            params['evaluation_metric'] = autosklearn.metrics.log_loss
        elif metric == 'r2':
            params['evaluation_metric'] = autosklearn.metrics.r2
        elif metric == 'mean_squared_error':
            params['evaluation_metric'] = autosklearn.metrics.mean_squared_error
        elif metric == 'mean_absolute_error':
            params['evaluation_metric'] = autosklearn.metrics.mean_absolute_error
        elif metric == 'median_absolute_error':
            params['evaluation_metric'] = autosklearn.metrics.median_absolute_error
        else:
            params['evaluation_metric'] = autosklearn.metrics.accuracy

        return params
    #---------------------------------------------------------------------------------------------#
    def get_params_fit_by_key(self, key):
        params = self.map_user_params(func_type='fit', model_type=key, user_hyperparameters=self.user_hyperparameters)        

        return params