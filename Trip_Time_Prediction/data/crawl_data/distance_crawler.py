from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from Trip_Time_Prediction.config.constants import *


class DistanceCrawler():
    waiting_time = 3

    def __init__(self):
        self.options = webdriver.ChromeOptions()
        self.options.add_argument('--no-sandbox')
        self.options.add_argument("--headless")
        chromedriver_mac_path = ROOT_PATH + "/Trip_Time_Prediction/data/crawl_data/chromedriver_mac"
        self.driver = webdriver.Chrome(chromedriver_mac_path, options=self.options)
        self.wait = WebDriverWait(self.driver, self.waiting_time)

    def quit(self):
        self.driver.close()
        self.driver.quit()
        print('chromedrivers have been killed')

    def get(self, start, end):
        start = str(start[0]) + "," + str(start[1])
        end = str(end[0]) + "," + str(end[1])
        url = 'https://www.google.com/maps/dir/' + start + '/' + end
        self.driver.get(url)
        xpath = "//div[@class='section-directions-trip clearfix selected']"
        condition = EC.visibility_of_element_located((By.XPATH, xpath))
        try:
            self.wait.until(condition)
            res = self.driver.find_element(By.XPATH, xpath).text.split("\n")[:2]
            return res
        except:
            print('DistanceCrawler: calling_url_error')
            pass
        finally:
            print(url)

#Just for test
if __name__ == "__main__":
    start = (43.86911432714224, -80.0661016312976)
    end = (43.8677599959127, -80.060698885946)
    crawler = DistanceCrawler()
    print(crawler.get(start, end))
    crawler.quit()
