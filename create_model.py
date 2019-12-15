import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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
            a = dataset[i:i + look_back, j]
            xset.append(a)
        dataY.append(dataset[i + look_back,
                             0])  # rateの次のデータをYに格納 => 現在のデータで予測するため
        dataX.append(xset)  # [[rate data], [depression data]]となるように格納される
    return np.array(dataX), np.array(dataY)


# Get poms data
look_back = 14
scaler = MinMaxScaler(feature_range=(0, 1))
df = pd.read_csv('result/df/poms_frame.csv').set_index('dates')
dataset = df.loc[:, ['prices', 'depression']]
dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset) * 0.9)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
x_train, y_train = create_dataset(train, look_back)
x_test, y_test = create_dataset(test, look_back)

# Build stock prediction model
model = Sequential()
model.add(LSTM(128, input_shape=(x_train.shape[1], look_back)))
model.add(Dense(500))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer='adam')

# Learning model
early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=0)
model.fit(x_train,
          y_train,
          batch_size=1,
          epochs=1000,
          callbacks=[early_stopping],
          validation_split=0.1)
model.save('model/deppression.h5')

# Prediction
pad_col = np.zeros(dataset.shape[1] - 1)


def pad_array(val):
    return np.array([np.insert(pad_col, 0, x) for x in val])


predicted = scaler.inverse_transform(pad_array(model.predict(x_test)))[:, 0]
actual = scaler.inverse_transform(pad_array(y_test))[:, 0]
result = pd.DataFrame({'predict': predicted, 'actual': actual})
result.plot()
plt.show()