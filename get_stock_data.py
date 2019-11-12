import datetime
from lib.data_serializer import DataSerializer
from pymongo import MongoClient
import pandas_datareader.data as web
import numpy as np


def daterange(_start, _end):
    for n in range((_end - _start).days):
        yield _start + datetime.timedelta(n)


client = MongoClient('0.0.0.0', 27017)
db = client.trump_stock
collection = db.dow_jones

year = 2018

start = datetime.datetime(year, 1, 1)
end = datetime.datetime(year, 12, 31)
djia = web.DataReader('DJIA', 'fred', start=start, end=start)['DJIA']

for date in daterange(start, end):
    djia = web.DataReader('DJIA', 'fred', start=date, end=date)['DJIA']
    if(len(djia) == 0 or np.isnan(djia[0])):
        collection.insert_one({'date': date, 'price': 0})
    else:
        collection.insert_one({'date': date, 'price': djia[0]})
