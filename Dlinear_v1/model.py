import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from decomposition import DecompositionLayer
torch.set_num_threads(10)
class DLinearModel(nn.Module):
    def __init__(self, input_size):
        super(DLinearModel, self).__init__()
        self.linear_seasonal = nn.Linear(input_size, 1)
        self.linear_trend = nn.Linear(input_size, 1)
        self.decomposition = DecompositionLayer(1)
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, context):
        seasonal, trend = self.decomposition(context)
        seasonal_output = self.linear_seasonal(seasonal)
        trend_output = self.linear_trend(trend)
        
        return seasonal_output + trend_output


def train_model(model, dataloader, criterion, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        for X in dataloader:
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, X)
            loss.backward()
            optimizer.step()

#  def main():

#     data = pd.read_csv('ETTh1.csv')  
#     X_s = torch.tensor(data['HUFL'].values[:data_size], dtype=torch.float32).view(-1, 1)
#     X_t = torch.tensor(data['HUFL'].values[:data_size], dtype=torch.float32).view(-1, 1)
#     y = torch.tensor(data['HUFL'].values[:data_size], dtype=torch.float32).view(-1, 1)

    
#     input_size = 1  
#     seasonality_size = 1
#     trend_size = 1
#     learning_rate = 0.0001

#     model = DLinearModel(input_size, seasonality_size, trend_size)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    
#     window_size = 200  
#     dataset = TensorDataset(X_s, X_t, y)
#     dataloader = DataLoader(dataset, batch_size=window_size, shuffle=True)

    
#     train_model(model, dataloader, criterion, optimizer)

    
#     new_X_s = torch.tensor([1.0], dtype=torch.float32).view(-1, 1)
#     new_X_t = torch.tensor([2.0], dtype=torch.float32).view(-1, 1)
#     prediction = model(new_X_s, new_X_t)

#     print("Prediction:", prediction.item())
#     plt.plot(data['HUFL'].values[:data_size])
    
#     plt.scatter(data_size+1, data['HUFL'].values[data_size-1]+prediction.item(), s=50, c='g', linewidths=1, marker='s', edgecolors='r')
#     plt.show()
    
def predict_future_values(model, last_X, m):
    #predicted_values = initial_values.clone().detach().view(-1).tolist()
    predicted_values = []
    for _ in range(m):
        #проверить что модель принимает нужный набор значений
        #last_X = torch.tensor([predicted_values[-1]], dtype=torch.float32).view(-1, 1)
        # last_X_t = torch.tensor([predicted_values[-1]], dtype=torch.float32).view(-1, 1)
        
        prediction = model(last_X)

        
        predicted_values.append(prediction.item())
        last_X = torch.tensor([predicted_values[-1]], dtype=torch.float32).reshape(1, 1, -1)
        

    return predicted_values

# def main2():
    
#     data = pd.read_csv('ETTh1.csv')  
#     X_s = torch.tensor(data['HUFL'].values[:data_size], dtype=torch.float32).view(-1, 1)
#     X_t = torch.tensor(data['HUFL'].values[:data_size], dtype=torch.float32).view(-1, 1)
#     y = torch.tensor(data['HUFL'].values[:data_size], dtype=torch.float32).view(-1, 1)

    
#     input_size = 1  
#     seasonality_size = 1
#     trend_size = 1
#     learning_rate = 0.0001

#     model = DLinearModel(input_size, seasonality_size, trend_size)
#     criterion = nn.L1Loss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    
#     window_size = 15  # Размер окна для rolling window forecasting
#     #переписать
#     dataset = TensorDataset(X_s, X_t, y)
#     dataloader = DataLoader(dataset, batch_size=window_size, shuffle=True)

    
#     train_model(model, dataloader, criterion, optimizer)


    
#     initial_values = torch.cat([X_s[-1], X_t[-1], y[-1]]).view(-1, 1)
#     m = 1000 #на сколько шагов предсказать
#     future_predictions = predict_future_values(model, X_s, X_t, initial_values, m)

#     print("Future Predictions:", future_predictions)
#     plt.plot(data['HUFL'].values[:data_size])
#     #plt.plot(, )
#     pred = data['HUFL'].values[data_size-1]
#     for i in range(m):
        
#         plt.scatter(data_size+1+i, future_predictions[i], s=50, c='g', linewidths=0.5)
        
#     plt.show()
# if __name__ == "__main__":
#     main2()
