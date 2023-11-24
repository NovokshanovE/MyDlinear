import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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

# Пример использования
input_size = 10  # размерность входных данных
trend_size = 5   # размерность тренда
season_size = 3  # размерность сезонности

model = TimeSeriesModel(input_size, trend_size, season_size)

# Пример входных данных
X = torch.randn(1, input_size)

# Получаем прогноз
output = model(X)
print("Ряд:", X)
# Выводим результат
print("Прогноз:", output.item())
