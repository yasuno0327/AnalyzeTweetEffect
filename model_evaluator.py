import pandas as pd
import numpy as np
from keras.metrics import SparseCategoricalAccuracy
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

model = load_model('model/rate_dep.h5')

# Prediction
predicted = model.predict(x_test)
result = pd.DataFrame(predicted)
result.columns = ['predict']
result['actual'] = y_test
result.plot()
plt.show()