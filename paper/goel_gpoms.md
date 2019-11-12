# Stock Prediction Using Twitter Sentiment Analysis (訳)

## ABSTRACT
In this paper, we apply sentiment analysis and machine learning principles to find the correlation between "public sentiment" and "market sentiment".
この論文では、感情分析と機械学習を適用して大衆の感情と株式市場の間の相関を見つける。

We use twitter data to predict public mood and use the predicted mood and previous days DJIA values to predict the stock market movements.
大衆の感情をツイッターのデータから予測し、予測した感情データと前の期間のDJIAの価格を利用して株式市場の動向を予測する。

In order to test our results, we propose a new cross validation method for financial data and obtain 75.56% accuracy using Self Organizing Fuzzy Neural Networks(SOFNN) on the Twitter feeds and DJIA values from the period June 2009 to December 2009.
結果をテストするために、財務データのための新しい相互検証方式を提案し、2009/1~2009/12 までのTwitterのフィードとDJIAの値で構築したSOFNNから75.56%のaccuracyを得た。

We also implement a naive protfolio management strategy based on our predicted values.
また、予測値に基づいたプロトフォリオ管理戦略を実装する。

Our work is based on Bollen et al's famous paper which predicted the same with 87% accuracy.
この手法はボーレンのものをベースとしており、同じ手法で87%のaccuracyを出しています。

## INTRODUCTION
