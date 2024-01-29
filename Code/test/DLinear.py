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

# Функция для обучения модели с многомерным вектором X
def train_dlinear_model(model, data, window_size, learning_rate, num_epochs):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i in range(len(data) - window_size):
            optimizer.zero_grad()
            
            # Получаем выборку для обучения
            x_seasonality = data[i:i+window_size, :1]  # Пример сезонности
            x_trend = data[i:i+window_size, 1:]       # Пример тренда
            y_true = data[i+window_size, :1]          # Значение, которое мы пытаемся предсказать
            
            # Преобразуем данные в тензоры PyTorch
            x_seasonality = torch.tensor(x_seasonality, dtype=torch.float32).view(1, -1)
            x_trend = torch.tensor(x_trend, dtype=torch.float32).view(1, -1)
            y_true = torch.tensor(y_true, dtype=torch.float32).view(1, -1)
            
            # Прогоняем данные через модель
            y_pred = model(x_seasonality, x_trend)
            
            # Вычисляем loss и делаем шаг обратного распространения
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Пример использования модели
input_size = window_size = 5  # Пример размера окна
seasonality_size = trend_size = 1  # Пример размеров слоев сезонности и тренда

# Генерируем многомерный временной ряд для примера
data = torch.rand((1000, 2))  

model = DLinearModel(input_size, seasonality_size, trend_size)

# Обучаем модель
train_dlinear_model(model, data, window_size, learning_rate=0.01, num_epochs=1000)
