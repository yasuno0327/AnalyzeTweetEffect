import pandas_datareader.data as web
import numpy as np
import datetime


class StockDataReader():
    # period = number of data (period >= 2)
    def __init__(self, period):
        self.period = datetime.timedelta(days=period)

    # Get Dow-Jones Price at specify period
    # date => tweet date
    def read_data(self, date):
        period = self.period
        start = date + datetime.timedelta(days=1)
        end = start + period
        djia = web.DataReader('DJIA', 'fred', start=start, end=end)
        datas = djia.dropna()['DJIA']

        while None in datas:
            start = start + datetime.timedelta(days=1)
            end = start + period
            djia = web.DataReader('DJIA', 'fred', start=start, end=end)
            datas = djia.dropna()['DJIA']

        print(datas)
        delta = (max(datas) - min(datas)) / datas[0]
        return delta
