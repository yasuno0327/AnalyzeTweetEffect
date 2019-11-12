from nltk.sentiment.vader import SentimentIntensityAnalyzer


class TextEvaluator():

    def __init__(self, day_data):
        self.day_data = day_data
        self.vader_analyzer = SentimentIntensityAnalyzer()

    def evaluate(self):
        len_datas = len(self.day_data['datas'])
        for i in range(len_datas):
            compound = 0
            positive = 0
            negative = 0
            score = self.vader_analyzer.polarity_scores(
                self.day_data['datas'][i]['text'])
            compound += score['compound']
            positive += score['pos']
            negative += score['neg']

        compound = compound/len_datas
        positive = positive/len_datas
        negative = negative/len_datas
        return compound
