
class ModelInfo:
    #---------------------------------------------------------------------------------------------#
    def __init__(self, model_name, model_type, model_object, model_path, model_params_dict,
                  model_metrics_dict, model_tags_dict):
        self.model_name = model_name
        self.model_type = model_type
        self.model_object = model_object
        self.model_path = model_path
        self.model_params_dict = model_params_dict
        self.model_metrics_dict = model_metrics_dict
        self.model_tags_dict = model_tags_dict
    
    #---------------------------------------------------------------------------------------------#
    def to_dict(self):
        return {
            'tags': {
                'model_name': self.model_name,
                'model_type': self.model_type,
                **(self.model_tags_dict or {})
            },
            'params': self.model_params_dict,
            'metrics': self.model_metrics_dict,
            #'artifact': self.model,
            'model': self.model_object,
            'model_path': self.model_path,
        }