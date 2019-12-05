
from datetime import datetime, timedelta
import numbers
from lib.data_serializer import DataSerializer
from lib.text_evaluator import TextEvaluator
from lib.stock_data_reader import StockDataReader
from lib.correlation_evaluator import CorrelationEvaluator
from pymongo import MongoClient, DESCENDING
import matplotlib.pyplot as plt
import numpy as np
from emotion_recognition.emotion_predictor import EmotionPredictor
from statistics import mean


def zscore(x, axis=None):
    xmean = np.array(x).mean(axis=axis, keepdims=True)
    xstd = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore

def rate_of_change(base, changed):
    return (changed-base)/base

def fluctuating(base, changed):
    return base-changed


# Initialize db
client = MongoClient('0.0.0.0', 27017)
db = client.trump_stock
tweet_col = db.tweets
dow_col = db.djia
take_office_at = datetime(2017, 1, 20)
tweets = tweet_col.find({'date': {'$gte': take_office_at}}).sort('date', 1)
timeseries = []
rate_datas = []
fluctuation_datas = []

tweets_of_days = []
day_first = tweets[0]

# Initialize poms model
model = EmotionPredictor(classification='poms', setting='mc')

# days tweet sentiment vs stock price
anger = []
depression = []
fatigue = []
vigour = []
tension = []
confusion = []

for tweet in tweets:
    tweeted_at = tweet['date']
    opened_at = tweeted_at.replace(hour=9,minute=30,second=0,microsecond=0)
    closed_at = tweeted_at.replace(hour=16,minute=0,second=0,microsecond=0)
    tweeted_day = tweeted_at.replace(hour=0, minute=0,second=0,microsecond=0)
    base_stock = {}

    if tweeted_at < opened_at: # 開場前 前日の情報
        tweets_of_days.append(tweet['text'])
    else: # 閉場後 営業時間内 当日の情報
        if day_first['date'].date() != tweet['date'].date():

            # 株価情報取得 Noneの場合は飛ばす
            prev_tweet_day = day_first['date'].replace(hour=0,minute=0,second=0,microsecond=0)
            base_stock = dow_col.find_one({'date': prev_tweet_day})
            if base_stock is None:
                tweets_of_days = []
                day_first = tweet
                tweets_of_days.append(tweet['text'])
                continue

            # 感情指数を計算し、保持
            anger.append(max(model.predict_probabilities(tweets_of_days)['Anger']))
            depression.append(max(model.predict_probabilities(tweets_of_days)['Depression']))
            fatigue.append(max(model.predict_probabilities(tweets_of_days)['Fatigue']))
            vigour.append(max(model.predict_probabilities(tweets_of_days)['Vigour']))
            tension.append(max(model.predict_probabilities(tweets_of_days)['Tension']))
            confusion.append(max(model.predict_probabilities(tweets_of_days)['Confusion']))

            # 株価情報を計算し、保持
            prev_tweet_day = day_first['date'].replace(hour=0,minute=0,second=0,microsecond=0)
            base_stock = dow_col.find_one({'date': prev_tweet_day})
            rate_datas.append(rate_of_change(base_stock['open'], base_stock['close'])) # closeまでに株価に反映されていると考える
            fluctuation_datas.append(fluctuating(base_stock['open'], base_stock['close']))

            tweets_of_days = [] # 次の日なので初期化
            day_first = tweet
            tweets_of_days.append(tweet['text']) # ここから再スタート
        else:
            tweets_of_days.append(tweet['text'])

# 相関係数の計算
# anger
ang_rate = CorrelationEvaluator(rate_datas, anger)
ang_fluc = CorrelationEvaluator(fluctuation_datas, anger)

# depression
dep_rate = CorrelationEvaluator(rate_datas, depression)
dep_fluc = CorrelationEvaluator(fluctuation_datas, depression)

# fatigue
fat_rate = CorrelationEvaluator(rate_datas, fatigue)
fat_fluc = CorrelationEvaluator(fluctuation_datas, fatigue)

# vigour
vig_rate = CorrelationEvaluator(rate_datas, vigour)
vig_fluc = CorrelationEvaluator(fluctuation_datas, vigour)

# tension
ten_rate = CorrelationEvaluator(rate_datas, tension)
ten_fluc = CorrelationEvaluator(fluctuation_datas, tension)

# confusion
con_rate = CorrelationEvaluator(rate_datas, confusion)
con_fluc = CorrelationEvaluator(fluctuation_datas, confusion)


# plt.subplot(221)
# plt.plot(timeseries, zscore(count_data), color='b', label="NumberOfTweet")
# plt.plot(timeseries, zscore(stock_data), color='r', label="Dow-Jones")
# plt.title(f"lag {lag} Dow x tweet count")
# plt.xlabel("datetime")
# plt.ylabel("Zscore")
# plt.legend(loc='best')

# plt.tight_layout()
# plt.show()

# 結果プリント
print("anger rate", ang_rate.evaluate())
print("anger fluctuation", ang_fluc.evaluate())

print("depression rate", dep_rate.evaluate())
print("depression fluctuation", dep_fluc.evaluate())

print("fatigue rate", fat_rate.evaluate())
print("fatigue fluctuation", fat_fluc.evaluate())

print("vigour rate", vig_rate.evaluate())
print("vigour fluctuation", vig_fluc.evaluate())

print("tension rate", ten_rate.evaluate())
print("tension fluctuation", ten_fluc.evaluate())

print("confusion rate", con_rate.evaluate())
print("confusion fluctuation", con_fluc.evaluate())