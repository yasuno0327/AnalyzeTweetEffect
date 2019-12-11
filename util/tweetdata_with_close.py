from datetime import datetime, timedelta
from pymongo import MongoClient
from bson.objectid import ObjectId


client = MongoClient('0.0.0.0', 27017)
db = client.trump_stock
tweet_col = db.tweets
djia_col = db.djia
close_col = db.close_base

take_office_at = datetime(2017, 1, 20)
tweets = tweet_col.find({'date': {'$gte': take_office_at}}).sort('date', 1)
for tweet in tweets:
  tweeted_at = tweet['date']
  closed_at = tweeted_at.replace(hour=16,minute=0,second=0,microsecond=0)
  tweeted_day = tweeted_at.replace(hour=0, minute=0,second=0,microsecond=0)

  if tweeted_at >= closed_at:
    tweeted_day += timedelta(1)

  stock = djia_col.find_one({'date': tweeted_day})
  while stock is None:
      tweeted_day += timedelta(1)
      stock = djia_col.find_one({'date': tweeted_day})

  tweet_with_close = close_col.find_one({'date': tweeted_day})
  if tweet_with_close is None:
    tweet_with_close = {'date': tweeted_day, 'close_price': stock['close'], 'tweets': [tweet]}
    close_col.save(tweet_with_close)
  else:
    tweet_with_close['tweets'].append(tweet)
    close_col.update({'_id': ObjectId(tweet_with_close['_id'])}, {'$set': tweet_with_close})