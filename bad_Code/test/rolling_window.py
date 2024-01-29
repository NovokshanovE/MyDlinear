import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pprint
class TimeSeriesModel(nn.Module):
    def __init__(self, input_size, trend_size, season_size):
        super(TimeSeriesModel, self).__init__()

        # Линейные слои для тренда
        self.trend_layer1 = nn.Linear(input_size, trend_size)
        self.trend_layer2 = nn.Linear(trend_size, 1)

        # Линейные слои для сезонности
        self.season_layer1 = nn.Linear(input_size, season_size)
        self.season_layer2 = nn.Linear(season_size, 1)

        # Смещение для окончательного прогноза
        self.bias = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, X):
        # Разделение данных на трендовую и сезонную составляющие
        trend_component = self.trend_layer2(torch.relu(self.trend_layer1(X)))
        season_component = self.season_layer2(torch.relu(self.season_layer1(X)))

        # Суммирование тренда и сезонности
        final_output = trend_component + season_component + self.bias

        return final_output

# Пример использования для предсказания на n шагов вперед
def predict_n_steps(model, X, n_steps):
    predictions = []

    for _ in range(n_steps):
        # Получаем текущий прогноз
        current_prediction = model(X)

        # Добавляем прогноз в список
        predictions.append(current_prediction.item())

        # Обновляем входные данные для следующего шага
        X = torch.cat([X[:,1:], current_prediction], dim=1)

    return predictions

# Пример входных данных
input_size = 1000  # размерность входных данных (многомерный вектор)
trend_size = 500   # размерность тренда
season_size = 300  # размерность сезонности

model = TimeSeriesModel(input_size, trend_size, season_size)

# Генерируем случайные входные данные
X = torch.randn(5, input_size)


n_steps = 100
predictions = predict_n_steps(model, X, n_steps)

# Выводим результат
print(f"Прогноз на {n_steps} шагов вперед:", predictions)
