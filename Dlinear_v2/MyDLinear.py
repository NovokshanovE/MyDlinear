# %% [markdown]
# ## Подключение библиотек и определение класса Dataset

# %%

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
# from decomposition import DecompositionLayer
torch.set_num_threads(1)


class DLinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(DLinearModel, self).__init__()
        self.linear_seasonal = nn.Linear(input_size, output_size)
        self.linear_trend = nn.Linear(input_size, output_size)
        self.decomposition = DecompositionLayer(input_size)
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, context):
        seasonal, trend = self.decomposition(context)
        #print(seasonal, trend)
        seasonal_output = self.linear_seasonal(seasonal.reshape(1, 1, -1))
        trend_output = self.linear_trend(trend.reshape(1, 1, -1))
        
        return seasonal_output + trend_output
    
class MyDataset(TensorDataset):
    def __init__(self, data, window, output):
        self.data = data
        self.window = window
        self.output = output

    def __getitem__(self, index):
        x = self.data[index:index+self.window]
        y = self.data[index+self.window:index+self.window+self.output]
        return x, y

    def __len__(self):
        return len(self.data) - self.window - self.output
    
class DecompositionLayer(nn.Module):
    """
    Returns the trend and the seasonal parts of the time series.
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0) # moving average 

    def forward(self, x):
        """Input shape: Batch x Time x EMBED_DIM"""
        # padding on the both ends of time series
        num_of_pads = (self.kernel_size) // 2
        front = x[:, 0:1, :].repeat(1, num_of_pads, 1)
        end = x[:, -1:, :].repeat(1, num_of_pads, 1)
        x_padded = torch.cat([front, x, end], dim=1)

        # calculate the trend and seasonal part of the series
        x_trend = self.avg(x_padded.permute(0, 2, 1))[:,:,:-1].permute(0, 2, 1)
        #print(x_trend.shape)
        x_seasonal = x - x_trend
        return x_seasonal, x_trend
    
class DLinear:
    def __init__(self, data_set = 1000, input_size = 100, output_size = 100, learning_rate = 0.0000001, step = 1, data_size = 3000, column_name = "HUFL"):
        self.input_size = input_size
        self.pred = self.input_size
        self.learning_rate = learning_rate
        self.output_size = output_size
        self.data_size  = data_size
        self.step = step
        # self.m = 10 #на сколько шагов предсказать
        self.data_set = data_set
        self.column_name = column_name
        self.model_name = f"update_model_v8_L1_SGD_{self.column_name}_input{self.input_size}_output{self.output_size}"
        self.model = None
        # self.data = None
        # self.X = None
        # self.x = None
    def train_model(self, model, dataloader, criterion, optimizer, num_epochs=100):
        for epoch in range(num_epochs):
            print("Epoch = ", epoch)
            for X, Y in dataloader:
                
                optimizer.zero_grad()
                #print(X, Y)
                output = model.forward(X).view(1, -1, 1)
                #print(torch.tensor([output.tolist()]), Y)
                loss = criterion(output, Y)
                loss.backward()
                optimizer.step()
    def data_reader(self):
        self.data = pd.read_csv('ETTh1.csv')  
        self.X = torch.tensor(self.data[self.column_name].values[:self.data_size:self.step], dtype=torch.float32).view(-1, 1)
        # self.x = pd.read_csv('ETTh1.csv').HUFL
        return self.data
    def set_data(self, func):
        self.X = torch.tensor([func(i) for i in range(self.data_size)], dtype=torch.float32).view(-1, 1)
        
    def set_model(self):
        self.model = DLinearModel(self.input_size, self.output_size)
    def train(self):
        
        criterion = nn.L1Loss()
        
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

        window_size = self.input_size  
        dataset = MyDataset(self.X, window_size, self.output_size)

        dataloader = DataLoader(dataset)#, shuffle=True)

        len(dataloader)

        self.train_model(self.model, dataloader, criterion, optimizer)



        torch.save(self.model.state_dict(), self.model_name)
    def load_modal(self):
        self.model.load_state_dict(torch.load(f"{self.model_name}"))
        self.model.eval()


        self.model.parameters

    def prediction(self):
        pred = self.input_size
        X_f = torch.tensor(self.data[self.column_name].values[self.data_set-pred:self.data_set-pred+pred*self.step:self.step], dtype=torch.float32).view(-1, 1)
        print(X_f)
        X_t = X_f.tolist()
        predicted_values = []
        X = torch.tensor([X_t])
        prediction = self.model(X)
        print(prediction)
        self.predicted_values = prediction.tolist()[-1][-1]
        return self.predicted_values
    def prediction_custom_data(self, func):
        pred = self.input_size
        X_f = torch.tensor([func(i) for i in range(self.data_set-pred, self.data_set-pred+pred*self.step,self.step)], dtype=torch.float32).view(-1, 1)
        print(X_f)
        X_t = X_f.tolist()
        predicted_values = []
        X = torch.tensor([X_t])
        prediction = self.model(X)
        print(prediction)
        self.predicted_values = prediction.tolist()[-1][-1]
        return self.predicted_values
    def MAE(self, ):
        i = [self.data_set+1+i*self.step for i in range(self.output_size)]

        actual = np.array([self.data[self.column_name].values[k] for k in i])
        prediction = self.prediction()

        l1_loss = abs(actual - prediction)
        mae_cost = l1_loss.mean()
        print("MAE:", mae_cost)
        
    def MAPE(self):
        i = [self.data_set+1+i*self.step for i in range(self.output_size)]

        actual = np.array([self.data[self.column_name].values[k] for k in i])
        prediction = self.prediction()

        l1_loss = abs(actual+1 - prediction-1)/abs(actual+1)

        """
        Output:
        [0 1 2 2]
        """

        mape_cost = np.sum(l1_loss)/self.output_size
        print("MAPE:", mape_cost)
        
        
        














      
    







# %%


# %%




