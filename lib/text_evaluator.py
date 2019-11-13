from nltk.sentiment.vader import SentimentIntensityAnalyzer


class TextEvaluator():

    def __init__(self, day_data):
        self.day_data = day_data
        self.vader_analyzer = SentimentIntensityAnalyzer()

    # Calculate score and return daily max score
    def evaluate(self):
        len_datas = len(self.day_data['datas'])
        for i in range(len_datas):
            compound = 0
            positive = 0
            negative = 0
            score = self.vader_analyzer.polarity_scores(
                self.day_data['datas'][i]['text'])
            if(score['compound'] > compound):
                compound += score['compound']
            if(score['pos'] > positive):
                positive += score['pos']
            if(score['neg'] > negative):
                negative += score['neg']
        score = {'compound': compound,
                 'pos': positive, 'neg': negative}
        return score
