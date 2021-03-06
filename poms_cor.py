from datetime import datetime, timedelta
import numbers
from pymongo import MongoClient
import matplotlib.pyplot as plt
import numpy as np
from emotion_recognition.emotion_predictor import EmotionPredictor
from statistics import mean
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.tsa.api as sm
from statsmodels.tsa.api import VAR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

client = MongoClient('0.0.0.0', 27017)
db = client.trump_stock
tweet_col = db.days_tweets
dow_col = db.djia
maxlag = 20


def zscore(x, axis=None):
    xmean = np.array(x).mean(axis=axis, keepdims=True)
    xstd = np.std(x, axis=axis, keepdims=True)
    zscore = (x - xmean) / xstd
    return zscore


def rate_of_change(base, changed):
    return (changed - base) / base


def fluctuating(base, changed):
    return changed - base


def granger_causality(df, time_key, causality_key):
    return grangercausalitytests(df[[time_key, causality_key]], maxlag=maxlag)


def get_stock_data(data_type, date):
    base_date = None
    comparison_date = None
    reflected_date = None
    if data_type == 'before_open':
        base_date = date - timedelta(1)
        base_type = 'close'
        comparison_date = date
        comparison_type = 'open'
        reflected_date = comparison_date.replace(hour=9,
                                                 minute=30,
                                                 second=0,
                                                 microsecond=0)
    elif data_type == 'opening':
        base_date = date
        base_type = 'open'
        comparison_date = date
        comparison_type = 'close'
        reflected_date = comparison_date.replace(hour=16,
                                                 minute=0,
                                                 second=0,
                                                 microsecond=0)
    elif data_type == 'after_close':
        base_date = date
        base_type = 'close'
        comparison_date = date + timedelta(1)
        comparison_type = 'open'
        reflected_date = comparison_date.replace(hour=9,
                                                 minute=30,
                                                 second=0,
                                                 microsecond=0)
    else:
        return None
    base_stock = dow_col.find_one({'date': base_date})
    while base_stock is None:
        base_date -= timedelta(1)
        base_type = 'close'
        base_stock = dow_col.find_one({'date': base_date})

    comparison_stock = dow_col.find_one({'date': comparison_date})
    while comparison_stock is None:
        comparison_date += timedelta(1)
        base_type = 'open'
        reflected_date = comparison_date.replace(hour=16,
                                                 minute=30,
                                                 second=0,
                                                 microsecond=0)
        comparison_stock = dow_col.find_one({'date': comparison_date})

    return {
        'base': base_stock[base_type],
        'comparison': comparison_stock[comparison_type],
        'date': reflected_date
    }


take_office_at = datetime(2017, 1, 20)
days_tweets = tweet_col.find({
    'date': {
        '$gte': take_office_at
    }
}).sort('date', 1)
rate_datas = []
fluctuation_datas = []
prices = []
tweet_texts = []
dates = []
positives = []
negatives = []
compounds = []

# Initialize poms model
# model = EmotionPredictor(classification='poms', setting='mc')
vader_analyzer = SentimentIntensityAnalyzer()

for tweets in days_tweets:
    texts = ''
    for tweet in tweets['datas']:
        texts += tweet['text'] + " "
    tweet_texts.append(texts)
    vader = vader_analyzer.polarity_scores(texts)
    price = get_stock_data(tweets['data_type'], tweets['date'])
    if price is None:
        print('continue')
        continue
    positives.append(vader['pos'])
    negatives.append(vader['neg'])
    compounds.append(vader['compound'])
    rate = rate_of_change(price['base'], price['comparison'])
    fluctuation = fluctuating(price['base'], price['comparison'])
    rate_datas.append(rate)
    prices.append(price['comparison'])
    dates.append(price['date'])
    fluctuation_datas.append(fluctuation)

# result = model.predict_probabilities(tweet_texts)
# df = pd.DataFrame({
#     'rate': rate_datas,
#     # 'fluctuation': fluctuation_datas,
#     'prices': prices,
#     'anger': zscore(result['Anger'].values),
#     'depression': result['Depression'].values,  # ゆううつ
#     'fatigue': zscore(result['Fatigue'].values),
#     'tension': result['Tension'].values,  # 緊張
#     'confusion': zscore(result['Confusion'].values),
#     'vigour': zscore(result['Vigour'].values),
#     # 'dates': dates  # ツイート反映後の株価情報の日時
# })
df_pos = pd.DataFrame({
    'rate': rate_datas,
    'positive': positives,
})
df_neg = pd.DataFrame({
    'rate': rate_datas,
    'negative': negatives
})
# df_ten = pd.DataFrame({
#     'rate': rate_datas,
#     'tension': result['Tension'].values,
# })
print(df_pos.corr(method='pearson'))
print(df_neg.corr(method='pearson'))

# df = df.set_index('dates')
# df.to_csv('result/df/poms_frame.csv')

# Make a VAR model
# dep_model = VAR(df_dep[:-1])
# ten_model = VAR(df_ten[:-1])
# dep_results = dep_model.fit(maxlags=20, ic='aic')
# ten_results = ten_model.fit(maxlags=20, ic='aic')
# print(dep_results)

# dep_irf = dep_results.irf(20)
# dep_irf.plot(orth=False)
# ten_irf = ten_results.irf(20)
# ten_irf.plot(orth=False)
# plt.show()