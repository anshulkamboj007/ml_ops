import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

import mlflow
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> RegressorMixin:
    """
    Args:
        x_train: pd.DataFrame
        x_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
    Returns:
        model: RegressorMixin
    """
    try:
        model = None
        tuner = None
        
        if config.model_name == "linear_regression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(x_train, y_train)
            return trained_model
        else:
            raise ValueError("Model name not supported")
    
    except Exception as e:
        logging.error(e)
        raise e
