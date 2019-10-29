from nltk.sentiment.vader import SentimentIntensityAnalyzer
import percache


class TextEvaluator():
    cache = percache.Cache('trump_stock_data')

    def __init__(self, datas):
        self.datas = datas
        self.vader_analyzer = SentimentIntensityAnalyzer()

    @cache
    def evaluate(self):
        for i in range(len(self.datas)):
            score = self.vader_analyzer.polarity_scores(self.datas[i]['text'])
            self.datas[i]['compound'] = score['compound']
            self.datas[i]['pos'] = score['pos']
            self.datas[i]['neg'] = score['neg']
        return self.datas
