
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
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
import pandas as pd


def zscore(x, axis=None):
    xmean = np.array(x).mean(axis=axis, keepdims=True)
    xstd = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore

def rate_of_change(base, changed):
    return (changed-base)/base

def fluctuating(base, changed):
    return base-changed

def granger_causality(df, time_key, causality_key):
    return grangercausalitytests(df[[time_key, causality_key]], maxlag=20)

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

df = pd.DataFrame({
    'rate': rate_datas,
    'fluctuation': fluctuation_datas,
    'anger': anger,
    'depression': depression,
    'fatigue': fatigue,
    'tension': tension,
    'confusion': confusion
})

f_datas = []
p_datas = []
mood_index = ['anger', 'depression', 'fatigue', 'tension', 'confusion']

# グレンジャー因果検定
for mood in mood_index:
    response = granger_causality(df, 'rate', mood)
    ftest = lag['ssr_ftest']
    f = ftest[0]
    p = ftest[1]
    f_datas.append(f)
    p_datas.append(p)
    f_df = pd.DataFrame({
    'f': f_datas,
    'p': p_datas
    })

    f_df.to_csv(f'result/{mood}.csv')