
from emotion_recognition.emotion_predictor import EmotionPredictor


model = EmotionPredictor(classification='poms', setting='mc')

tweets = [
  "...during the talks the U.S. will start, on September 1st, putting a small additional Tariff of 10% on the remaining 300 Billion Dollars of goods and products coming from China into our Country. This does not include the 250 Billion Dollars already Tariffed at 25%... ",
  "Getting VERY close to a BIG DEAL with China. They want it, and so do we!",
  "Will be meeting at 9:00 with top automobile executives concerning jobs..."
]

result = model.predict_probabilities(tweets)

print(result)