import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# loc[行ラベル,列ラベル]
# iloc[行の番号,列の番号]
# ix[行番号orラベル, 列番号orラベル]


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        xset = []
        for j in range(
                dataset.shape[1]
        ):  # 列数 dataset.shape[:] => (1599, 2) ※ depression, rate 2カラムの場合
            a = dataset.iloc[i:i + look_back, j]
            xset.append(a)
        dataY.append(dataset.iloc[i + look_back,
                                  0])  # rateの次のデータをYに格納 => 現在のデータで予測するため
        dataX.append(xset)  # [[rate data], [depression data]]となるように格納される
    return np.array(dataX), np.array(dataY)


# Get poms data
look_back = 13
df = pd.read_csv('result/df/poms_frame.csv').set_index('dates')
dataset = df.loc[:, ['rate', 'depression']]
X, Y = create_dataset(dataset, look_back)
(x_train, x_test, y_train, y_test) = train_test_split(X,
                                                      Y,
                                                      test_size=0.1,
                                                      random_state=0)

# Build stock prediction model
model = Sequential()
model.add(LSTM(300, input_shape=(x_train.shape[1], look_back)))
model.add(Dense(1))
model.add(Activation('relu'))
model.compile(loss="mean_squared_error", optimizer='adam', metrics=['mape'])

# Learning model
model.fit(x_train, y_train, batch_size=1, epochs=200)
model.save('model/rate_dep.h5')

# Prediction
predicted = model.predict(x_test)
result = pd.DataFrame(predicted)
result.columns = ['predict']
result['actual'] = y_test
result.plot()
plt.show()

score = model.evaluate(x_test, y_test)
print(score)