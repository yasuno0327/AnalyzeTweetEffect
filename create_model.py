import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K


def mda(y_true, y_pred):
    s = np.equal(np.sign(y_true[1:] - y_true[:-1]),
                 np.sign(y_pred[1:] - y_true[:-1]))
    return np.mean(s.astype(np.int))

def mean_directional(y_true, y_pred):
    s = np.equal(np.sign(y_true[1:] - y_true[:-1]),
                 np.sign(y_pred[1:] - y_true[:-1]))
    return s

def mda_value_length(s):
    num_true = list(filter(lambda n:n==True, s))
    num_false = list(filter(lambda n:n==False, s))
    return [len(num_true), len(num_false)]



def mda_loss(y, y_hat):
    s = K.equal(
            K.sign(y[1:] - y[:1]),
            K.sign(y_hat[1:] - y_hat[:-1])
        )
    return K.mean(K.cast(s, K.floatx()))


def binary(y_true, y_pred):
    s = np.equal(np.sign(y_true), np.sign(y_pred))
    return np.mean(s.astype(np.int))


def pad_array(val):
    return np.array([np.insert(pad_col, 0, x) for x in val])


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


model_keys = [['prices'], ['prices', 'confusion'], ['prices', 'depression'], ['prices', 'tension']]
pre_prices = None
pr_score = None
pr_num = None
pr_true = None
pr_false = None

pre_tension = None
ten_score = None
ten_num = None
ten_true = None
ten_false = None

pre_depression = None
dep_tension = None
dep_num = None
dep_true = None
dep_false = None

actual = None
confusion_score = None
confusion_num = None
confusion_true = None
confusion_false = None

for key in model_keys:

    # Get poms data
    look_back = 4
    if key[-1] == 'tension':
        look_back = 7
    elif key[-1] == 'confusion':
        look_back = 6
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = pd.read_csv('result/df/poms_frame.csv').set_index('dates')
    dataset = df.loc[:, key]
    dataset = scaler.fit_transform(dataset)
    # train_size = len(dataset) - 1023
    # test_size = 1023
    # train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    train, test = dataset[0:1599, :], dataset[1599:1745]
    x_train, y_train = create_dataset(train, look_back)
    x_test, y_test = create_dataset(test, look_back)

    # Build stock prediction model
    model = load_model(f'model/{key[-1]}.h5')
    # model = Sequential()
    # model.add(LSTM(32, input_shape=(x_train.shape[1], look_back)))
    # model.add(Dense(1))
    # model.add(Activation('tanh'))
    # model.compile(
    #     loss="mape",  #"mean_squared_error",
    #     # loss=mda_loss,
    #     optimizer='adam',
    #     metrics=['mape']) #4% 3.68% 3.84% 3.4%

    # # Learning model
    # # early_stopping = EarlyStopping(monitor='mape', mode='auto')
    # history = model.fit(
    #     x_train,
    #     y_train,
    #     batch_size=64,
    #     epochs=3000,
    #     verbose=0
    # )
    # print(f"{key[-1]}: {history.history['mape'][-1]}")
    # # callbacks=[early_stopping])
    # model.save(f'model/{key[-1]}.h5')

    # Prediction
    pad_col = np.zeros(dataset.shape[1] - 1)
    predicted = scaler.inverse_transform(pad_array(model.predict(x_test)))[:,
                                                                           0]
    # Test
    actual = scaler.inverse_transform(pad_array(y_test))[:, 0]
    mda_score = mda(actual, predicted) * 100
    binary_score = binary(actual, predicted) * 100
    s = mean_directional(actual, predicted)
    num_result = mda_value_length(s)
    num_true = num_result[0]
    num_false = num_result[1]

    # result = pd.DataFrame({'predict': predicted, 'actual': actual})
    # result.plot()
    # plt.show()

    if key[-1] == 'prices':
        pre_prices = predicted
        pr_score = mda_score
        pr_num = num_true + num_false
        pr_true = num_true
        pr_false = num_false
        pr_actual = actual
    elif key[-1] == 'tension':
        pre_tension = predicted
        ten_score = mda_score
        ten_num = num_true + num_false
        ten_true = num_true
        ten_false = num_false
        ten_actual = actual
    elif key[-1] == 'depression':
        pre_depression = predicted
        dep_score = mda_score
        dep_num = num_true + num_false
        dep_true = num_true
        dep_false = num_false
        dep_actual = actual
    elif key[-1] == 'confusion':
        pre_confusion = predicted
        confusion_score = mda_score
        confusion_num = num_true + num_false
        confusion_true = num_true
        confusion_false = num_false
        confusion_actual = actual


print(f"price score: {pr_score} price len: {pr_num} price true: {pr_true} price false {pr_false}")
print(f"confusion score: {confusion_score} confusion len: {confusion_num} confusion true: {confusion_true} confusion false: {confusion_false}")
print(f"depression score: {dep_score} depression len: {dep_num} depression true: {dep_true} depression false: {dep_false}")
print(f"tension score: {ten_score} tension len: {ten_num} tension true: {ten_true} tension false {ten_false}")

plt.rcParams["font.family"] = "IPAexGothic"
pr_res_df = pd.DataFrame({
    '株価を利用した予測値': pre_prices,
    '実測値': pr_actual
})
axes = pr_res_df.plot()
plt.ylabel('株価')
plt.savefig('result/images/res_price.png')
plt.close()

ten_res_df = pd.DataFrame({
    '株価 x tensionを利用した予測値': pre_tension,
    '実測値': ten_actual
})
axes = ten_res_df.plot()
plt.ylabel('株価')
plt.savefig('result/images/res_tension.png')
plt.close()

dep_res_df = pd.DataFrame({
    '株価 x depressionを利用した予測値': pre_depression,
    '実測値': dep_actual
})
axes = dep_res_df.plot()
plt.ylabel('株価')
plt.savefig('result/images/res_depression.png')
plt.close()

con_res_df = pd.DataFrame({
    '株価 x confusionを利用した予測値': pre_confusion,
    '実測値': confusion_actual
})

axes = con_res_df.plot()
plt.ylabel('株価')
plt.savefig('result/images/res_confusion.png')
plt.close()
# res_df = pd.DataFrame({
#     '株価単体での予測': pre_prices,
#     'tensionを使った予測': pre_tension,
#     'depressionを使った予測': pre_depression,
#     '実際の株価': actual
# })

# axes = res_df.plot(subplots=True, layout=(2, 2), sharex=True, sharey=True)
# plt.ylabel('株価')
# plt.savefig('result/images/lstm_res.png')
# plt.close()


def pad_array(val):
    return np.array([np.insert(pad_col, 0, x) for x in val])