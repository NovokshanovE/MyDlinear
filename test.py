import torch
import torch.nn as nn
import torch.optim as optim

class TimeSeriesModel(nn.Module):
    def __init__(self, input_size, time_size):
        super(TimeSeriesModel, self).__init__()
        self.Ws = nn.Linear(input_size, 1)
        self.Wt = nn.Linear(time_size, 1)
        self.b = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, Xs, Xt):
        # Применяем линейные слои к входным данным
        WsXs = self.Ws(Xs)
        WtXt = self.Wt(Xt)
        
        # Суммируем все компоненты
        y = WsXs + WtXt + self.b
        
        return y

# Пример использования
input_size = 10  # размерность входных данных Xs
time_size = 5    # размерность входных данных Xt
model = TimeSeriesModel(input_size, time_size)

# Пример входных данных
Xs = torch.randn(1, input_size)
Xt = torch.randn(1, time_size)

# Получаем прогноз
output = model(Xs, Xt)

# Выводим результат
print("Прогноз:", output.item())
