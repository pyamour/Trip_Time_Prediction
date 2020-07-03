import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from Trip_Time_Prediction.config.constants import *

def predict_use_xgboost(x_pred, xgb_path):
    if os.path.isfile(xgb_path):
        xgb_model = xgb.Booster()
        xgb_model.load_model(xgb_path)
        dtest = xgb.DMatrix(x_pred)
        y_pred_log = xgb_model.predict(dtest)
        return y_pred_log
    else:
        return None

def predict_model_xgb():
    for city in city_list:
        xgb_path = XGB_PATH + city + '_' + XGB_MODEL_SUFFIX
        df = pd.read_csv(DATA_PATH + '/to_predict/' + city + '_data.csv')
        x_pred = df[TRAIN_COLS]
        mapper_path = MAPPER_PATH + city + '_' + XGB_MAPPER_SUFFIX
        with open(mapper_path, "rb") as f:
            mapper= pickle.load(f)
        x_pred = mapper.transform(x_pred)
        y_pred_log = predict_use_xgboost(x_pred, xgb_path)
        df["log_trip_time"] = y_pred_log
        df["triptime"] = np.expm1(df["log_trip_time"])
        df.to_csv(DATA_PATH + '/predict_result/' + city + '_data.csv',  index=False)

if __name__ == '__main__':
    predict_model_xgb()