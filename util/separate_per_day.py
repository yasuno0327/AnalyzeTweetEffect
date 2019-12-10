from datetime import datetime, timedelta
from pymongo import MongoClient
from bson.objectid import ObjectId


client = MongoClient('0.0.0.0', 27017)
db = client.trump_stock
tweet_col = db.tweets
djia_col = db.djia
days_col = db.days_tweets

take_office_at = datetime(2017, 1, 20)
tweets = tweet_col.find({'date': {'$gte': take_office_at}}).sort('date', 1)

for tweet in tweets:
  tweeted_at = tweet['date']
  opened_at = tweeted_at.replace(hour=9,minute=30,second=0,microsecond=0)
  closed_at = tweeted_at.replace(hour=16,minute=0,second=0,microsecond=0)
  tweeted_day = tweeted_at.replace(hour=0, minute=0,second=0,microsecond=0)
  if tweeted_at < opened_at:
    base_date = tweeted_day - timedelta(1)
    next_date = tweeted_day
    data_type = 'before_open' #前日の終値
  elif opened_at <= tweeted_at < closed_at:
    base_date = tweeted_day
    next_date = base_date
    data_type = 'opening' #当日の
  elif tweeted_at >= closed_at:
    base_date = tweeted_day
    next_date = base_date + timedelta(1)
    data_type = 'after_close'
  else:
    continue

  days_tweet = days_col.find_one({'$and': [{'date': base_date}, {'data_type': data_type}]})
  if days_tweet is None:
    days_tweet = {'date': base_date, 'data_type': data_type, 'datas': [tweet]}
    days_col.save(days_tweet)
  else:
    days_tweet['datas'].append(tweet)
    days_col.update({'_id': ObjectId(days_tweet['_id'])}, {'$set': days_tweet})