from datetime import datetime, timedelta
from lib.data_serializer import DataSerializer
from pymongo import MongoClient


client = MongoClient('0.0.0.0', 27017)
db = client.trump_stock
collection = db.tweets


def save_db(year_datas):
    for data in year_datas:
        collection.insert_one(data)



prefix = 'condensed_'
suffix = '.json'

for year in range(2017, 2020):
    filename = prefix + str(year) + suffix
    serializer = DataSerializer(filename)
    year_datas = serializer.text_datas()
    save_db(year_datas)
