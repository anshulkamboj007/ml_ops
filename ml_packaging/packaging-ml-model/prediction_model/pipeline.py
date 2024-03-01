from sklearn.pipeline import Pipeline

from prediction_model.config import config
import prediction_model.processing.preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np


classification_pipeline=Pipeline(
    [
        ('Mean Imputation',pp.MeanImputer(variables=config.NUM_FEATURES)),
        ('Mode imputation',pp.ModeImputer(variables=config.CAT_FEATURES)),
        ('domain processing',pp.Domainprocessing(variable_to_modify=config.FEATURES_TO_MODIFY,variable_to_add=config.FEATURES_TO_ADD)),
        ('drop features',pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
        ('label encoder',pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
        ('log transform',pp.LogTransforms(variables=config.LOG_FEATURES)),
        ('minmax scale',MinMaxScaler()),
        ('logistic classifier',LogisticRegression(random_state=0))
    ]
)