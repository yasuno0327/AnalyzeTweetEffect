
from lib.data_serializer import DataSerializer
from lib.text_evaluator import TextEvaluator
from lib.stock_data_reader import StockDataReader
from lib.correlation_evaluator import CorrelationEvaluator
import datetime
import matplotlib.pyplot as plt

filename = 'condensed_2018.json'
serializer = DataSerializer(filename)
datas = serializer.text_datas()
median = datas[int(len(datas)/2)]
datas = serializer.limit_data_with_date(median['date'])
print("splited data size", len(datas))
analyzer = TextEvaluator(datas)
datas = analyzer.evaluate()
datareader = StockDataReader(1)
# Calculate average
# Todo: Cache datas
for i in range(len(datas)):
    average = datareader.read_data(datas[i]['date'])
    datas[i]['average'] = average

compounds = map(lambda data: data['compound'], datas)
positives = map(lambda data: data['pos'], datas)
print(list(positives))
negatives = map(lambda data: data['neg'], datas)
print(list(negatives))
averages = map(lambda data: data['average'], datas)
com_cor = CorrelationEvaluator(compounds, averages)
neg_cor = CorrelationEvaluator(negatives, averages)
pos_cor = CorrelationEvaluator(positives, averages)
print('com', com_cor.evaluate())
print('\n')
print('neg', neg_cor.evaluate())
print('\n')
print('pos', pos_cor.evaluate())

# 1ヶ月など短い期間でやってみる
# 散布図などにまとめてデータの性質をつかんでみる
# 株価の変動率の日数を変動させてみる
# 株価関連の辞書がないか => 株に関するワードに対するポジティブ、ネガティブ値を集めた辞書 調べる
