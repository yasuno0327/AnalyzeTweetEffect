from datetime import datetime, timedelta
from lib.data_serializer import DataSerializer
from pymongo import MongoClient
import numpy as np
import pandas_datareader.data as web
import pandas as pd

def date_range(_start, _end):
        for i in range((_end-_start).days):
                yield _start + timedelta(i)

client = MongoClient('0.0.0.0', 27017)
db = client.trump_stock
collection = db.djia

start = datetime(2017, 1, 20)
end = datetime.now()
djia = web.DataReader('dia', 'yahoo', start=start, end=end)
for pd_date in djia.index:
        date = pd_date.to_pydatetime()
        stock_info = djia.loc[pd_date].to_dict()
        stock_info = {k.lower(): v
                for k, v in
                stock_info.items()
        }
        stock_info['date'] = date
        collection.insert_one(stock_info)