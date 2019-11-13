import datetime
from lib.data_serializer import DataSerializer
from pymongo import MongoClient


client = MongoClient('0.0.0.0', 27017)
db = client.trump_stock
collection = db.tweets


def separate_every_day(datas):
    separated = []
    day_datas = []
    first = datas[0]
    for data in datas:
        if(first['date'].date() == data['date'].date()):
            day_datas.append(data)
        else:
            date = day_datas[0]['date']
            date = datetime.datetime(date.year, date.month, date.day)
            separated.append(
                {"datas": day_datas, "date": date})
            first = data
            day_datas = []
            day_datas.append(data)
    return separated[::-1]


prefix = 'condensed_'
suffix = '.json'

for year in range(2009, 2019):
    filename = prefix + str(year) + suffix
    serializer = DataSerializer(filename)
    year_datas = serializer.text_datas()
    separated = separate_every_day(year_datas)
    for data in separated:
        collection.insert_one(data)
