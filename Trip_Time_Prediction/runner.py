import os, sys
sys.path.append(os.path.dirname(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")))

from Trip_Time_Prediction.data.crawl_data.crawl_data import Crawl_Data
from Trip_Time_Prediction.data.make_dataset.make_dataset import *
from Trip_Time_Prediction.models.train_model import *
from Trip_Time_Prediction.models.predict_model import *
import traceback

def crawl_data_from_google():
    crawl_data = Crawl_Data()
    try:
        crawl_data.city_crawler()
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    crawl_data_from_google()
    get_raw_data()
    transform_data()
    process_data()
    train_model_xgb()
    predict_model_xgb()
