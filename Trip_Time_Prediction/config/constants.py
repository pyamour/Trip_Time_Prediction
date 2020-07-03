import os

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TRAIN_COLS = ['start_lat', 'start_lng', 'end_lat', 'end_lng', 'month', 'day', 'weekday', 'time']
city_list = ['Scarbourgh', 'Markham']
DATA_PATH = ROOT_PATH + "/data"
DB_PATH = DATA_PATH + '/crawl/trip.db'
XGB_PATH = ROOT_PATH + "/models/xgb/"
XGB_MODEL_SUFFIX = "xgb.pkl"
XGB_MAPPER_SUFFIX = "mapper.pkl"
MAPPER_PATH = ROOT_PATH + '/mapper/'