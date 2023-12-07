import unittest
from parameterized import parameterized

from automlwrapper.AutoMLWrapper import AutoMLWrapper
from .DataSamples import glass_df, mnist_byte_df, m4_df, mnist_tp

hp = {'epochs':20, 'time_limit': 60*5}#, 'num_trials': 4,}
class TestAutoMLWrapper(unittest.TestCase):

    #---------------------------------------------------------------------------------------------#
    @parameterized.expand([
        #("autosklearn", "tabular", "classification", "multiclass", glass_df, 'Type'),

        #("autokeras", "tabular", "classification", "multiclass", glass_df, 'Type'),
        ("autokeras", "image", "classification", "multiclass", mnist_tp, 'label'),
        #("autokeras", "timeseries", "forecast", "forecast", m4_df, 'target'),
        
        #("autogluon", "tabular", "classification", "multiclass", glass_df, 'Type'),
        #("autogluon", "image", "classification", "multiclass", mnist_byte_df, 'label'),
        #("autogluon", "timeseries", "forecast", "forecast", m4_df, 'target'),

    ])
    def test_auto_ml_wrapper(self, library, data_type, task_type, problem_type, data_sample, target_column):

        with self.subTest(library=library,
                          data_type=data_type,
                          task_type=task_type,
                          problem_type=problem_type,
                          data_sample=data_sample,
                          target_column=target_column
                          ):

            wrapper = AutoMLWrapper(library)

            wrapper.Initialize(data_sample, target_column, task_type, data_type, problem_type)
            wrapper.Train(data_sample, target_column, task_type, data_type, problem_type, hp)

#=================================================================================================#
if __name__ == '__main__':
    unittest.main()