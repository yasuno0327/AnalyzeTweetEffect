from nltk.sentiment.vader import SentimentIntensityAnalyzer


class TextEvaluator():

    def __init__(self, datas):
        self.datas = datas
        self.vader_analyzer = SentimentIntensityAnalyzer()

    def evaluate(self):
        for i in range(len(self.datas)):
            score = self.vader_analyzer.polarity_scores(self.datas[i]['text'])
            self.datas[i]['compound'] = abs(score['compound'])
            self.datas[i]['pos'] = score['pos']
            self.datas[i]['neg'] = score['neg']
            if(score['compound'] >= 0.05):
                self.datas[i]['score'] = 1
            elif(score['compound'] <= 0.05):
                self.datas[i]['score'] = -1
            else:
                self.datas[i]['score'] = 0
        return self.datas
