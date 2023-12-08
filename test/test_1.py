import torch
import torch.nn as nn
import torch.optim as optim

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
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            for i in range(len(data) - self.window_size):
                optimizer.zero_grad()

                # Получаем выборку для обучения
                x_seasonality = data[i:i+self.window_size, 1:]  # Все признаки с временного шага
                x_trend = data[i:i+self.window_size, 1:]
                y_true = data[i+self.window_size, 0]  # Значение, которое мы пытаемся предсказать

                # Преобразуем данные в тензоры PyTorch
                x_seasonality = x_seasonality.float().view(1, -1, self.window_size)
                x_trend = x_trend.float().view(1, -1, self.window_size)
                y_true = y_true.float().view(1, -1, 1)  # Изменено на 1, чтобы соответствовать размерности y_pred

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
            x_seasonality = data[i:i+self.window_size, 1:].view(1, -1, self.window_size)
            x_trend = data[i:i+self.window_size, 1:].view(1, -1, self.window_size)

            # Прогнозируем следующие значения
            with torch.no_grad():
                prediction = self.model(x_seasonality, x_trend).squeeze().tolist()

            predictions.append(prediction)

        # Продолжаем предсказывать следующие значения на основе предыдущих предсказаний
        for _ in range(num_steps - 1):
            x_seasonality = predictions[-self.window_size:]
            x_trend = predictions[-self.window_size:]

            # Преобразуем данные в тензоры PyTorch
            x_seasonality = torch.tensor(x_seasonality).float().view(1, -1, self.window_size)
            x_trend = torch.tensor(x_trend).float().view(1, -1, self.window_size)

            # Прогнозируем следующее значение
            with torch.no_grad():
                prediction = self.model(x_seasonality, x_trend).squeeze().tolist()

            predictions.append(prediction)

        return predictions

# Пример использования класса TimeSeriesPredictor с временным рядом с k признаками
input_size = 5  # Пример количества признаков во входных данных
seasonality_size = trend_size = 1  # Пример размеров слоев сезонности и тренда
window_size = 5  # Пример размера окна

# Создаем экземпляр класса TimeSeriesPredictor
predictor = TimeSeriesPredictor(input_size, seasonality_size, trend_size, window_size)

# Генерируем временной ряд с k признаками для примера
data = torch.rand((1000, input_size + 1))  # 10 временных шагов, k признаков + 1 для времени

# Обучаем модель
predictor.train(data, learning_rate=0.01, num_epochs=100)

# Предсказываем следующие 3 значения
predictions_3_steps = predictor.predict(data, num_steps=1)
print("Predictions for 3 steps ahead:", predictions_3_steps)
