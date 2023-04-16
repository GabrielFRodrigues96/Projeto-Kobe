"""
This is a boilerplate pipeline 'train_models'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import conform_data_2PT,conform_data_3PT,train_logistic_regression,train_data_2PT,best_classification,report_model,normalize, train_data_3PT

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
    [
        # node(
        #     func = conform_data_2PT,
        #     name = 'conform_data_2PT',
        #     inputs = 'raw_data',
        #     outputs = 'conformed_data_2PT',
        # ),
        node(
            func = conform_data_3PT,
            name = 'conform_data_3PT',
            inputs= 'raw_data',
            outputs= 'conformed_data_3PT',
        ),
        node(
            func = train_data_3PT,
            name = 'train_data_3PT',
            inputs = 'conformed_data_3PT',
            outputs = ['x_train','x_test','y_train','y_test'],
        ),       
        # node(
        #     func = train_data_2PT,
        #     name = 'train_data',
        #     inputs = 'conformed_data_2PT',
        #     outputs = ['x_train','x_test','y_train','y_test']
        # ),
        ##node(
        ##    func =best_classification,
        ##    name = 'best_classificator',
        ##    inputs= 'conformed_data',
        ##    outputs = 'best_model',
        ##),
        node(
            func = normalize,
            name = 'normalize',
            inputs = 'x_train',
            outputs = 'x_train_norm',
        ),
        node(
            func = train_logistic_regression,
            name = 'train_logistic_regression',
            inputs = ['x_train_norm','y_train'],
            outputs = 'logistic_regression_model',
        ),
    ])
