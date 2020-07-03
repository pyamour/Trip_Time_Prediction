import pandas as pd
import numpy as np
from sklearn_pandas import DataFrameMapper
import sklearn
import pickle
import sqlite3
import datetime
from Trip_Time_Prediction.config.constants import *


def get_raw_data():
    conn = sqlite3.connect(DB_PATH)
    sql = "SELECT city, start_lat, start_lng, end_lat, end_lng, distance, triptime, timestamp from triptime_distance"
    df = pd.read_sql(sql, conn)
    df.to_csv(DATA_PATH + '/raw/data.csv', index=False)
    return df

def transform_data(df=None):
    if not df:
        df = pd.read_csv(DATA_PATH + '/raw/data.csv')
    df['distance'] = list(map(lambda x: convert_distance_to_km(x), df['distance'].values))
    df = df[df["distance"] != 0]
    df['triptime'] = list(map(lambda x: convert_traveltime_to_min(x), df['triptime'].values))
    df = df[df["triptime"] != 0]
    df['month'] = list(map(lambda x: float(x.split('-')[1]), df['timestamp'].values))
    df['day'] = list(map(lambda x: float(x.split('-')[2]), df['timestamp'].values))
    df["date"] = list(
        map(lambda x: x.split("-")[0] + "-" + x.split("-")[1] + "-" + x.split("-")[2], df["timestamp"].values))
    df['weekday'] = list(
        map(lambda x: float(
            datetime.date(int(x.split('-')[0]), int(x.split('-')[1]), int(x.split('-')[2])).weekday()) + 1,
            df['date'].values))
    df['time'] = list(map(lambda x: x.split('-')[-1][0:4], df['timestamp'].values))
    df['time'] = list(map(lambda x: convert_timestamp_to_numeric(x), df['time'].values))
    for city in city_list:
        df[df["city"] == city].drop(['city'], axis=1).to_csv(DATA_PATH + '/interim/' + city + '_data.csv', index=False)

def convert_distance_to_km(distance):
    dis_arr = distance.split(' ')
    km = 0
    for i in range(len(dis_arr)):
        try:
            if dis_arr[i] == 'km':
                if "," not in dis_arr[i - 1]:
                    if float(dis_arr[i - 1]) < 100:
                        km += float(dis_arr[i - 1])
            elif dis_arr[i] == 'm':
                km += float(dis_arr[i - 1]) / 1000
        except Exception:
            print(dis_arr)
            pass
    return km

def convert_traveltime_to_min(travel_time):
    time_arr = travel_time.split(' ')
    min = 0
    for i in range(len(time_arr)):
        if time_arr[i] == 'min':
            min += float(time_arr[i - 1])
        elif time_arr[i] == 'h':
            min += float(time_arr[i - 1]) * 60
    if min > 30:
        return 0  #just predict short distance triptime
    return min

def convert_timestamp_to_numeric(timestamp):
    hour = timestamp[0:2]
    min = timestamp[2:4]
    time = float(hour) + float(min) / 60
    return time

def init_mapper(df, mapper_path):
    mapper = DataFrameMapper([
        (['start_lat'], sklearn.preprocessing.MinMaxScaler()),
        (['start_lng'], sklearn.preprocessing.MinMaxScaler()),
        (['end_lat'], sklearn.preprocessing.MinMaxScaler()),
        (['end_lng'], sklearn.preprocessing.MinMaxScaler()),
        (['month'], sklearn.preprocessing.MinMaxScaler()),
        (['day'], sklearn.preprocessing.MinMaxScaler()),
        (['weekday'], sklearn.preprocessing.MinMaxScaler()),
        (['time'], sklearn.preprocessing.MinMaxScaler()),
    ], df_out=True)

    data_mapper = np.round(mapper.fit_transform(df.copy()).astype(np.double), 3)
    if os.path.isfile(mapper_path):
        os.remove(mapper_path)
    with open(mapper_path, "wb") as f:
        pickle.dump(mapper, f)
    print("Fitting: ", type(mapper))
    return data_mapper


def process_data():
    for city in city_list:
        df = pd.read_csv(DATA_PATH + '/interim/' + city + '_data.csv')
        df['log_trip_time'] = np.log1p(df['triptime'])
        mapper_path = MAPPER_PATH + city + '_' + XGB_MAPPER_SUFFIX
        init_mapper(df.drop(['log_trip_time','triptime'], axis=1), mapper_path).to_csv(DATA_PATH + '/processed/' + city + '_x_data.csv', index=False)
        df[['log_trip_time']].to_csv(DATA_PATH + '/processed/' + city + '_y_data.csv', index=False)

if __name__ == "__main__":
    get_raw_data()
    transform_data()
    process_data()
