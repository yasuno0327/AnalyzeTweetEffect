from datetime import datetime, timedelta
from pymongo import MongoClient
from bson.objectid import ObjectId


client = MongoClient('0.0.0.0', 27017)
db = client.trump_stock
tweet_col = db.tweets
djia_col = db.djia

take_office_at = datetime(2017, 1, 20)
tweets = tweet_col.find({'date': {'$gte': take_office_at}}).sort('date', 1)
for tweet in tweets:
  tweeted_at = tweet['date']
  opened_at = tweeted_at.replace(hour=9,minute=30,second=0,microsecond=0)
  closed_at = tweeted_at.replace(hour=16,minute=0,second=0,microsecond=0)
  tweeted_day = tweeted_at.replace(hour=0, minute=0,second=0,microsecond=0)
  base_date = None
  next_date = None
  data_type = ''

  if tweeted_at < opened_at:
    base_date = tweeted_day - timedelta(1)
    next_date = tweeted_day
    base_key = 'close'
    next_key = 'open'
    data_type = 'before_open'
  elif opened_at <= tweeted_at < closed_at:
    base_date = tweeted_day
    next_date = base_date
    base_key = 'open'
    next_key = 'close'
    data_type = 'opening'
  elif tweeted_at >= closed_at:
    base_date = tweeted_day
    next_date = base_date + timedelta(1)
    base_key = 'close'
    next_key = 'open'
    data_type = 'after_close'
  else:
    continue

  base_stock = djia_col.find_one({'date': base_date})
  while base_stock is None:
      base_date -= timedelta(1)
      base_stock = djia_col.find_one({'date': base_date})
      base_key = 'close'

  next_stock = djia_col.find_one({'date': next_date})
  while next_stock is None:
    next_date += timedelta(1)
    next_stock = djia_col.find_one({'date': next_date})
    next_key = 'open'

  base_price = base_stock[base_key]
  next_price = base_stock[next_key]
  tweet_col.update_one({'_id': ObjectId(tweet['_id'])}, {'$set': {'djia': {'base': base_price, 'next': next_price, 'date': base_stock['date'], 'data_type': data_type}}})