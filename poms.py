
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
from statistics import mean


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
lag = 2

# Initialize poms model
model = EmotionPredictor(classification='poms', setting='mc')
tweets = []

# days tweet count vs stock price
count_data = []
# days tweet sentiment vs stock price
anger = []
depression = []
fatigue = []
vigour = []
tension = []
confusion = []


for data in datas:
    stock_price = dow_col.find_one(
        {'date': data['date'] + datetime.timedelta(days=lag)})
    if(stock_price == None):
        continue
    if(stock_price['price'] == 0):
        continue
    stock_data.append(stock_price['price'])
    timeseries.append(data['date'])
    tweets_of_days = []
    # day tweets
    for tweet in data['datas']:
        text = tweet['text']
        tweets_of_days.append(text)
    anger.append(max(model.predict_probabilities(tweets_of_days)['Anger']))
    depression.append(max(model.predict_probabilities(tweets_of_days)['Depression']))
    fatigue.append(max(model.predict_probabilities(tweets_of_days)['Fatigue']))
    vigour.append(max(model.predict_probabilities(tweets_of_days)['Vigour']))
    tension.append(max(model.predict_probabilities(tweets_of_days)['Tension']))
    confusion.append(max(model.predict_probabilities(tweets_of_days)['Confusion']))

zscore(anger)
zscore(depression)
zscore(fatigue)
zscore(vigour)
zscore(tension)
zscore(confusion)
zscore(stock_data)

anger_cor = CorrelationEvaluator(stock_data, anger)
dep_cor = CorrelationEvaluator(stock_data, depression)
fat_cor = CorrelationEvaluator(stock_data, fatigue)
vig_cor = CorrelationEvaluator(stock_data, vigour)
ten_cor = CorrelationEvaluator(stock_data, tension)
con_cor = CorrelationEvaluator(stock_data, confusion)

# plt.subplot(221)
# plt.plot(timeseries, zscore(count_data), color='b', label="NumberOfTweet")
# plt.plot(timeseries, zscore(stock_data), color='r', label="Dow-Jones")
# plt.title(f"lag {lag} Dow x tweet count")
# plt.xlabel("datetime")
# plt.ylabel("Zscore")
# plt.legend(loc='best')

# plt.tight_layout()
# plt.show()

print("anger", anger_cor.evaluate())
print("depression", dep_cor.evaluate())
print("fatigue", fat_cor.evaluate())
print("vigour", vig_cor.evaluate())
print("tension", ten_cor.evaluate())
print("confusion", con_cor.evaluate())