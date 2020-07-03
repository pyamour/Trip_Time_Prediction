import sqlite3
from Trip_Time_Prediction.data.crawl_data.distance_crawler import DistanceCrawler
from Trip_Time_Prediction.config.constants import *
import datetime
import warnings
import traceback
from multiprocessing import cpu_count, Pool

warnings.filterwarnings("ignore")

class Crawl_Data():

    def __init__(self):
        self.cores = int(cpu_count() * 0.7)
        self.city_list = city_list

    def city_crawler(self):
        for city in self.city_list:
            crawl_loc_list = []
            loc_list = self.get_location(str(city))
            for start_loc in loc_list:
                for end_loc in loc_list:
                    if start_loc != end_loc:
                        crawl_loc_list.append([start_loc, end_loc, str(city)])
            pool = Pool(processes=self.cores)
            pool.map(line_crawler, crawl_loc_list)
            pool.close()
            pool.join()

    def get_location(self, city):
        conn = sqlite3.connect(DB_PATH)
        sql = 'SELECT lat, lng from location where city = "' + city + '" limit 6'
        print(sql)
        cursor = conn.execute(sql)
        loc_list = list(cursor.fetchall())
        conn.close()
        return loc_list

def insert_result_into_db(time_dist_list, city, start_loc, end_loc):
    conn = sqlite3.connect(DB_PATH)
    triptime = time_dist_list[0]
    distance = time_dist_list[1]
    start_lat = start_loc[0]
    start_lng = start_loc[1]
    end_lat = end_loc[0]
    end_lng = end_loc[1]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S%f")
    print(timestamp)
    val = (str(city), float(start_lat), float(start_lng), float(end_lat), float(end_lng), str(distance), str(triptime), str(timestamp))
    sql = "INSERT INTO triptime_distance (city, start_lat, start_lng, end_lat, end_lng, distance," \
              "triptime, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    conn.execute(sql, val)
    conn.commit()
    conn.close()
    return

def line_crawler(crawl_loc):
    start_loc = crawl_loc[0]
    end_loc = crawl_loc[1]
    city = crawl_loc[2]
    try:
        dist_crawler = DistanceCrawler()
        time_dist_list = dist_crawler.get(start_loc, end_loc)
        dist_crawler.quit()
        insert_result_into_db(time_dist_list, city, start_loc, end_loc)
    except Exception as e:
        traceback.print_exc()
    return

if __name__ == "__main__":

    crawl_data = Crawl_Data()
    try:
        crawl_data.city_crawler()
    except Exception as e:
        traceback.print_exc()


