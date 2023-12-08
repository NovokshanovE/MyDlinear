import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

def load_data(series, n_steps):
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i:(i + n_steps)])
        y.append(series[(i + n_steps)])
    return np.array(X), np.array(y)

def baseline_model(n_steps, n_neurons):
    model = Sequential()
    model.add(Dense(n_neurons, input_shape=(n_steps, 1), activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(optimizer=Adam(0.01), loss=MeanSquaredError())
    return model

def moving_average(x, window):
    return np.convolve(x, np.ones(window), 'valid') / window

def roll_predict(roll, train, test, window):
    preds = []
    for i in range(roll, len(test)):
        cut = int(i - roll * window)
        if cut > 0:
            rmse = np.sqrt(mean_squared_error(test[cut-1:cut], train[cut-1:cut]))
            preds.append(preds[-1] * (1 - rmse) + test[cut-1:cut].mean() * rmse)
        else:
            preds.append(test[cut-1:cut].mean())
    return preds

# Определение окна для просмотра
window_view = 3

# Определение окна для обучения
window_train = 21

# Определение количества промежутков на временном ряду
n_steps = 5

# Определение количества нейронов
n_neurons = 50

# Определение размера сети
batch_size = 1
epochs = 100

# Загрузка данных
series = np.array(data)

# Масштабирование данных
scaler = MinMaxScaler()
series = scaler.fit_transform(series.reshape(-1, 1))

# Подготовка данных для обучения и проверки
X_train, y_train = load_data(series, n_steps)
X_test, y_test = load_data(series[window_train:], n_steps)

# Обучение модели
model = baseline_model(n_steps, n_neurons)
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, shuffle=False)

# Генерация предсказаний
y_pred = model.predict(X_test)

# Корректировка предсказаний с использованием скользящего среднего
y_pred_corr = moving_average(y_pred, window_view)

# Оценка ошибки предсказания
rmse = np.sqrt(mean_squared_error(y_test, y_pred_corr))
print(f'RMSE: {rmse}')

# Повторное обучение модели на более большом окне просмотра
window_roll = 21
X_train, y_train = load_data(series, window_roll)
model.fit(X_train, y_train, epochs=epochs, batch_size=