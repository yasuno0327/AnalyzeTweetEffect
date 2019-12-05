
import datetime
from pymongo import MongoClient
from flask import Flask, render_template

app = Flask(__name__)

# Initialize db
client = MongoClient('0.0.0.0', 27017)
db = client.trump_stock
collection = db.tweet_col
datas = collection.find_one({'year': 2018})['tweet_data']
tweets = []

for data in datas:
    for tweet in data['datas']:
        tweets.append(tweet['text'])


@app.route('/')
def print():
    return render_template('print.html', tweets=tweets)


if __name__ == "__main__":
    app.run(debug=True)
