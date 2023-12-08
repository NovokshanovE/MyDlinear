import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

class DLinearModel(nn.Module):
    def __init__(self, input_size, seasonality_size, trend_size):
        super(DLinearModel, self).__init__()
        self.seasonality_layer = nn.Linear(input_size, seasonality_size)
        self.trend_layer = nn.Linear(input_size, trend_size)
        self.bias = nn.Parameter(torch.Tensor(1))
        
    def forward(self, x_seasonality, x_trend):
        seasonality_output = self.seasonality_layer(x_seasonality)
        trend_output = self.trend_layer(x_trend)
        return seasonality_output + trend_output + self.bias

class TimeSeriesPredictor:
    def __init__(self, input_size, seasonality_size, trend_size, window_size):
        self.model = DLinearModel(input_size, seasonality_size, trend_size)
        self.window_size = window_size

    def train(self, data, learning_rate, num_epochs):
        criterion = nn.L1Loss()#MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            for i in range(len(data) - self.window_size):
                optimizer.zero_grad()

                # Получаем выборку для обучения
                x_seasonality = data[i:i+self.window_size, 0].reshape(-1, 1)
                x_trend = data[i:i+self.window_size, 1:]
                y_true = data[i+self.window_size, 0].reshape(-1, 1)

                # Преобразуем данные в тензоры PyTorch
                x_seasonality = torch.tensor(x_seasonality, dtype=torch.float32)
                x_trend = torch.tensor(x_trend, dtype=torch.float32)
                y_true = torch.tensor(y_true, dtype=torch.float32)

                # Прогоняем данные через модель
                y_pred = self.model(x_seasonality, x_trend)

                # Вычисляем loss и делаем шаг обратного распространения
                loss = criterion(y_pred, y_true)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    def predict(self, data, num_steps):
        predictions = []

        for i in range(len(data) - self.window_size):
            x_seasonality = data[i:i+self.window_size, 0].reshape(-1, 1)
            x_trend = data[i:i+self.window_size, 1:]

            # Преобразуем данные в тензоры PyTorch
            x_seasonality = torch.tensor(x_seasonality, dtype=torch.float32)
            x_trend = torch.tensor(x_trend, dtype=torch.float32)

            # Прогнозируем следующие значения
            with torch.no_grad():
                prediction = self.model(x_seasonality, x_trend).item()

            predictions.append(prediction)

        # Продолжаем предсказывать следующие значения на основе предыдущих предсказаний
        for _ in range(num_steps - 1):
            x_seasonality = predictions[-self.window_size:]
            x_trend = predictions[-self.window_size:]

            # Преобразуем данные в тензоры PyTorch
            x_seasonality = torch.tensor(x_seasonality, dtype=torch.float32).view(1, -1)
            x_trend = torch.tensor(x_trend, dtype=torch.float32).view(1, -1)

            # Прогнозируем следующее значение
            with torch.no_grad():
                prediction = self.model(x_seasonality, x_trend).item()

            predictions.append(prediction)

        return predictions

    def load_data_from_csv(self, file_path):
        # Загружаем данные из CSV-файла с использованием библиотеки pandas
        df = pd.read_csv(file_path)
        
        # Предполагаем, что первый столбец - это временные метки
        # Остальные столбцы - признаки
        data = df.to_numpy()

        return data

# Пример использования класса TimeSeriesPredictor с загрузкой данных из CSV
input_size = 2  # Пример количества признаков во входных данных
seasonality_size = trend_size = 1  # Пример размеров слоев сезонности и тренда
window_size = 5  # Пример размера окна

# Создаем экземпляр класса TimeSeriesPredictor
predictor = TimeSeriesPredictor(input_size, seasonality_size, trend_size, window_size)

# Загружаем данные из CSV
data_path = "ETTh1.csv"  # Замените на путь к вашему файлу данных
data = predictor.load_data_from_csv(data_path)

# Обучаем модель
predictor.train(data, learning_rate=0.01, num_epochs=100)

# Предсказываем следующие 3 значения
predictions_3_steps = predictor.predict(data, num_steps=3)
print("Predictions for 3 steps ahead:", predictions_3_steps)
