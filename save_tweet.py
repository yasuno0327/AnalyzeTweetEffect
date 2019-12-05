from datetime import datetime, timedelta
from lib.data_serializer import DataSerializer
from pymongo import MongoClient


client = MongoClient('0.0.0.0', 27017)
db = client.trump_stock
collection = db.tweets


def save_db(year_datas):
    for data in year_datas:
        tweet_date = data['date']

        if tweet_date.hour < 9 and tweet_date.minute < 30: # 9:30よりも前は前日扱いする
            data['date'] -= timedelta(days=1)
        if tweet_date.hour > 15: # 取引所は16:00 closeなので16:00以降は翌日扱いする
            data['date'] += timedelta(days=1)
        data['date'] = data['date'].replace(minute=0, hour=0, second=0, microsecond=0)
        collection.insert_one(data)



prefix = 'condensed_'
suffix = '.json'

for year in range(2009, 2019):
    filename = prefix + str(year) + suffix
    serializer = DataSerializer(filename)
    year_datas = serializer.text_datas()
    save_db(year_datas)
