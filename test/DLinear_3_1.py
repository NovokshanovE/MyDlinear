import torch
import numpy as np
from torch import nn
from torch.autograd import Variable

class DLinear(nn.Module):
    def __init__(self, window_size, input_size, hidden_size, output_size):
        super(DLinear, self).__init__()
        self.window_size = window_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden = self.initHidden()

        self.fc1 = nn.Linear(self.window_size * self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

    def forward(self, input):
        input = input.view(1, -1)
        self.hidden = self.initHidden()
        self.hidden = self.fc1(input)
        self.hidden = self.relu(self.hidden)
        output = self.fc2(self.hidden)
        return output, self.hidden

def train(model, data, learning_rate, num_epochs):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (data, target) in enumerate(data):
            data = Variable(torch.tensor(data)).float()
            target = Variable(torch.tensor(target)).float()

            output, hidden = model(data)
            target = target.view(1, -1, model.window_size).expand_as(output)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                      .format(epoch+1, num_epochs, i+1, len(data), loss.item()))

# Уровень среднего значения
mean = 0

# Разброс стандартного отклонения
std_dev = 1

# Число точек для временного ряда
num_points = 100

# Шаг по времени
time_step = 1

# Формирование временного ряда
time_series = []
for i in range(num_points):
    time_series.append([np.random.normal(mean, std_dev)])
data = time_series
model = DLinear(5, 1, 10, 1)
train(model, data, learning_rate=0.01, num_epochs=100)