"""
This is a boilerplate pipeline 'train_models'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import conform_data, train_logistic_regression,train_data,best_classification,report_model,normalize


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
    [
        node(
            func = conform_data,
            name = 'conform_data',
            inputs = 'raw_data',
            outputs = 'conformed_data',
        ),
        node(
            func = train_data,
            name = 'train_data',
            inputs = 'conformed_data',
            outputs = ['x_train','x_test','y_train','y_test']
        ),
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
