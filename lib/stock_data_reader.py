import pandas_datareader.data as web
import numpy as np
import datetime
from pymongo import MongoClient


class StockDataReader():
    def __init__(self, start, end):
        self.collection = MongoClient('0.0.0.0', 27017).trump_stock.dow_jones
        self.start = start
        self.end = end

    def read_price(self):
        res = self.collection.find({'date': self.start})
        return res[0]['price']

    def period_price_sub(self):
        res = self.collection.find(
            {'date': {'$lte': self.end, '$gte': self.start}})
        result = [res[0], res[-1]]
        return result

    def period_price_delta(self):
        res = self.collection.find(
            {'data': {'$gte': self.start, '$lte': self.end}})
        delta = (max(res) - min(res)) / res[0]
        return delta
