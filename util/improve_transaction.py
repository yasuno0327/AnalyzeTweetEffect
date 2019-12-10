from pymongo import MongoClient

client = MongoClient('0.0.0.0', 27017)
db = client.trump_stock
collection = db.djia
collection.create_index('date')