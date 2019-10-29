import pandas_datareader.data as web
import datetime


class StockDataReader():
    def __init__(self, period):
        self.period = datetime.timedelta(weeks=period)

    # Get Dow-Jones Price at specify period
    # date => tweet date
    def read_data(self, date):
        period = self.period
        start = date + datetime.timedelta(days=1)
        end = date + period
        djia = web.DataReader('DJIA', 'fred', start=start, end=end)
        datas = djia.dropna()['DJIA']
        delta = (max(datas) - min(datas)) / datas[0]
        return delta
