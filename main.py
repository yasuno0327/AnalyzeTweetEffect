
import datetime
import numbers
from lib.data_serializer import DataSerializer
from lib.text_evaluator import TextEvaluator
from lib.stock_data_reader import StockDataReader
from lib.correlation_evaluator import CorrelationEvaluator
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
datas = collection.find_one({'year': 2018})['tweet_data']
timeseries = []
stock_data = []
lag = 11

# days tweet count vs stock price
count_data = []
# days tweet sentiment vs stock price
compound_data = []
pos_data = []
neg_data = []

for data in datas:
    stock_price = dow_col.find_one(
        {'date': data['date'] + datetime.timedelta(days=lag)})
    if(stock_price == None):
        continue
    if(stock_price['price'] == 0):
        continue
    text_ev = TextEvaluator(data)
    stock_data.append(stock_price['price'])
    timeseries.append(data['date'])
    count_data.append(len(data['datas']))
    score = text_ev.evaluate()
    compound_data.append(score['compound'])
    pos_data.append(score['pos'])
    neg_data.append(score['neg'])

count_stock_cor = CorrelationEvaluator(stock_data, count_data)
compound_cor = CorrelationEvaluator(stock_data, compound_data)
pos_cor = CorrelationEvaluator(stock_data, pos_data)
neg_cor = CorrelationEvaluator(stock_data, neg_data)

plt.subplot(221)
plt.plot(timeseries, zscore(count_data), color='b', label="NumberOfTweet")
plt.plot(timeseries, zscore(stock_data), color='r', label="Dow-Jones")
plt.title(f"lag {lag} Dow x tweet count")
plt.xlabel("datetime")
plt.ylabel("Zscore")
plt.legend(loc='best')

plt.subplot(222)
plt.plot(timeseries, zscore(compound_data), color='b', label="compound")
plt.plot(timeseries, zscore(stock_data), color='r', label="Dow-Jones")
plt.title(f"lag {lag} Dow x compound")
plt.xlabel("datetime")
plt.ylabel("Zscore")
plt.legend(loc='best')

plt.subplot(223)
plt.plot(timeseries, zscore(pos_data), color='b', label="positive")
plt.plot(timeseries, zscore(stock_data), color='r', label="Dow-Jones")
plt.title(f"lag {lag} Dow x positive")
plt.xlabel("datetime")
plt.ylabel("Zscore")
plt.legend(loc='best')

plt.subplot(224)
plt.plot(timeseries, zscore(neg_data), color='b', label="negative")
plt.plot(timeseries, zscore(stock_data), color='r', label="Dow-Jones")
plt.title(f"lag {lag} Dow x negative")
plt.xlabel("datetime")
plt.ylabel("Zscore")
plt.legend(loc='best')


plt.tight_layout()
plt.show()

print("count", count_stock_cor.evaluate())
print("compound", compound_cor.evaluate())
print("positive", pos_cor.evaluate())
print("negative", neg_cor.evaluate())
