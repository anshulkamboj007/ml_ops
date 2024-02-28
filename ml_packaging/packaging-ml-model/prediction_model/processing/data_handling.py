import os
import pandas as pd
import joblib
import logging

from prediction_model.config import config

def load_dataset(file_name):
    filepath=os.path.join(config.DATAPATH,file_name)
    _data=pd.read_csv(filepath)
    return _data

def save_pipeline(pipeline_to_save):
    save_path=os.path.join(config.SAVE_MODEL_PATH,config.MODEL_NAME)
    joblib.dump(pipeline_to_save,save_path)
    logging.info(f"model has been saved under name {config.MODEL_NAME}")

def load_pipeline(pipeline_to_load):
    save_path=os.path.join(config.SAVE_MODEL_PATH,config.MODEL_NAME)
    model_loaded=joblib.load(save_path)
    logging.info(f"model has been loaded")
    return model_loaded