
import datetime
from lib.data_serializer import DataSerializer
from lib.text_evaluator import TextEvaluator
from lib.stock_data_reader import StockDataReader
from lib.correlation_evaluator import CorrelationEvaluator
from lib.data_visualizer import DataVisualizer

filename = 'condensed_2018.json'
serializer = DataSerializer(filename)
datas = serializer.text_datas()
# median = datas[int(len(datas)/3) - 1]
# datas = serializer.limit_data_with_date(median['date'])
datas = datas[0:500]
print("splited data size", len(datas))
analyzer = TextEvaluator(datas)
datas = analyzer.evaluate()
datareader = StockDataReader(14)

# Calculate average
# Todo: Cache datas
for i in range(len(datas)):
    average = datareader.read_data(datas[i]['date'])
    datas[i]['average'] = average

compounds = list(map(lambda data: data['compound'], datas))
positives = list(map(lambda data: data['pos'], datas))
negatives = list(map(lambda data: data['neg'], datas))
scores = list(map(lambda data: data['score'], datas))
averages = list(map(lambda data: data['average'], datas))

# Caluculate correlation
com_cor = CorrelationEvaluator(compounds, averages)
pos_cor = CorrelationEvaluator(positives, averages)
neg_cor = CorrelationEvaluator(negatives, averages)
score_cor = CorrelationEvaluator(scores, averages)

# Show data scatter plot
# DataVisualizer(compounds, 'compund').scatter()
# DataVisualizer(positives, 'positives').scatter()
# DataVisualizer(negatives, 'negatives').scatter()
DataVisualizer(scores, 'scores').scatter()
DataVisualizer(averages, 'delta').scatter()

# Print result to stdout
print('com', com_cor.evaluate())
print('positive', pos_cor.evaluate())
print('negative', neg_cor.evaluate())
print('score', score_cor.evaluate())

# 1ヶ月など短い期間でやってみる
# 散布図などにまとめてデータの性質をつかんでみる
# 株価の変動率の日数を変動させてみる
# 株価関連の辞書がないか => 株に関するワードに対するポジティブ、ネガティブ値を集めた辞書 調べる
