
import datetime
import numbers
from lib.data_serializer import DataSerializer
from lib.text_evaluator import TextEvaluator
from lib.stock_data_reader import StockDataReader
from lib.correlation_evaluator import CorrelationEvaluator
from lib.data_visualizer import DataVisualizer
from pymongo import MongoClient
import matplotlib.pyplot as plt
import numpy as np
from emotion_recognition.emotion_predictor import EmotionPredictor


def zscore(x, axis=None):
    xmean = np.array(x).mean(axis=axis, keepdims=True)
    xstd = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore


# Initialize db
client = MongoClient('0.0.0.0', 27017)
db = client.trump_stock
collection = db.tweet_col
dow_col = db.dow_jones
datas = collection.find_one({'year': 2013})['tweet_data']
timeseries = []
stock_data = []

# days tweet count vs stock price
count_data = []
datareader = StockDataReader(0)

# days tweet sentiment vs stock price
tweet_senti = []

for data in datas:
    stock_price = dow_col.find_one(
        {'date': data['date'] + datetime.timedelta(days=11)})
    if(stock_price == None):
        continue
    if(stock_price['price'] == 0):
        continue
    text_ev = TextEvaluator(data)
    stock_data.append(stock_price['price'])
    timeseries.append(data['date'])
    count_data.append(len(data['datas']))
    tweet_senti.append(text_ev.evaluate())

count_stock_cor = CorrelationEvaluator(stock_data, count_data)
compound_cor = CorrelationEvaluator(stock_data, tweet_senti)
print("count", count_stock_cor.evaluate())
print("compound", compound_cor.evaluate())
plt.subplot(2, 1, 1)
plt.plot(timeseries, zscore(count_data), color='b', label="NumberOfTweet")
plt.plot(timeseries, zscore(stock_data), color='r', label="Dow-Jones")
plt.title("cor=" + str(count_stock_cor.evaluate()))
plt.xlabel("datetime")
plt.ylabel("Zscore")
plt.legend(loc='best')

plt.subplot(2, 1, 2)
plt.plot(timeseries, zscore(tweet_senti), color='g', label="compound")
plt.plot(timeseries, zscore(stock_data), color='r', label="Dow-Jones")
plt.title("cor=" + str(compound_cor.evaluate()))
plt.xlabel("datetime")
plt.ylabel("Zscore")
plt.legend(loc='best')
plt.show()
