import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

@step
def train_model(X_train:pd.DataFrame,X_test:pd.DataFrame,
                y_train:pd.DataFrame,y_test:pd.DataFrame)->RegressorMixin:
    """
    trains model on ingested data

    Args:
    X_train:pd.DataFrame,X_test:pd.DataFrame,
    y_train:pd.DataFrame,y_test:pd.DataFrame

    """
    try:
        model=None
        
        if config.model_name == "LinearRegression":
            model=LinearRegressionModel()
            trained_model=model.train(X_train,y_train)
            return trained_model
        else: 
            raise ValueError("model {} not supported".format(config.model_name))
    except Exception as e:
        logging.error("eror in traing model;{}".format(e))
        raise e
