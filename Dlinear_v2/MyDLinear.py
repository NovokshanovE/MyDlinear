

from alive_progress import alive_bar

import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
# from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
import numpy as np
# import torch.optim.lr_scheduler as lr_scheduler
from statsmodels.tsa.seasonal import STL
# import intel_extension_for_pytorch as ipex
# from decomposition import DecompositionLayer
#torch.set_num_threads(10)
# from statsmodels.tsa.seasonal import seasonal_decompose



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
    
    def decompos(self, x):
        seasonal, trend = self.decomposition(x)
        return seasonal, trend
    
class OneLayerModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(OneLayerModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
        
        # self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, context):
        # seasonal, trend = self.decomposition(context)
        #print(seasonal, trend)
        # seasonal_output = self.linear_seasonal(seasonal.reshape(1, 1, -1))
        output = self.linear(context.reshape(1, 1, -1))
        
        return output
    
    def decompos(self, x):
        seasonal, trend = self.decomposition(x)
        return seasonal, trend
    
class DLinearModelSTL(nn.Module):
    def __init__(self, input_size, output_size):
        super(DLinearModelSTL, self).__init__()
        self.linear_seasonal = nn.Linear(input_size, output_size)
        self.linear_trend = nn.Linear(input_size, output_size)
        self.linear_resid = nn.Linear(input_size, output_size)
        # self.decomposition = DecompositionLayer(input_size)
        
        
        
    def forward(self, context):
        seasonal, trend, resid = self.decomposition(context)
        #print(seasonal, trend)
        seasonal_output = self.linear_seasonal(seasonal.reshape(1, 1, -1))
        trend_output = self.linear_trend(trend.reshape(1, 1, -1))
        resid_output = self.linear_resid(resid.reshape(1, 1, -1))
        
        return seasonal_output + trend_output + resid_output
    
    def decomposition(self, x):
        stl = STL(pd.Series(x.view(-1).tolist()), period=10)
        res = stl.fit()
        seasonal, trend, resid = torch.tensor(res.seasonal.tolist(), dtype=torch.float32).view(1, -1, 1),\
        torch.tensor(res.trend.tolist(), dtype=torch.float32).view(1, -1, 1),\
        torch.tensor(res.resid.tolist(), dtype=torch.float32).view(1, -1, 1)
        return seasonal, trend, resid

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
        # print(x)
        # result_add = seasonal_decompose(x_padded, model='additive')
        
        # calculate the trend and seasonal part of the series
        x_trend = self.avg(x_padded.permute(0, 2, 1))[:,:,:-1].permute(0, 2, 1)
        # print("Delta trend:", x_trend - result_add.trend)
        #print(x_trend.shape)
        x_seasonal = x - x_trend
        return x_seasonal, x_trend
    
class DLinear:
    def __init__(self, data_set = 1000, input_size = 100, output_size = 100, learning_rate = 0.00001, step = 1, data_size = 3000, column_name = "HUFL", dataset_name = 'dataset'):
        torch.set_num_threads(20)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        print()

        #Additional Info when using cuda
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        self.input_size = input_size
        self.pred = self.input_size
        self.learning_rate = learning_rate
        self.output_size = output_size
        self.data_size  = data_size
        self.step = step
        # self.m = 10 #на сколько шагов предсказать
        self.data_set = data_set
        self.column_name = column_name
        self.model_name = f"dlinear(test2_MA)_{dataset_name}_{self.column_name}_input{self.input_size}_output{self.output_size}"
        self.model = None
        # self.data = None
        # self.X = None
        # self.x = None
    def train_model(self, model, dataloader, criterion, optimizer, num_epochs=100):
        with alive_bar(num_epochs) as bar:
            for epoch in range(num_epochs):
                # bar = statusbar.StatusBar("Status")
                
                # bar.add_progress(epoch, "#")
                # bar.add_progress(num_epochs-epoch*10, ".")
                # print(" ", bar.format_status())
                #print(f"Epoch = {epoch}", end = '\r')
                # declare your expected total
                            # <<-- your original loop
                # print(epoch, end='\r')
                           # process each item
                bar()
                                  # call `bar()` at the end
                
                for X, Y in dataloader:
                    
                    optimizer.zero_grad()
                    #print(X, Y)
                    output = model.forward(X).view(1, -1, 1)
                    #print(torch.tensor([output.tolist()]), Y)
                    loss = criterion(output, Y)
                    loss.backward()
                    optimizer.step()
    def data_reader(self, file_name, column_name):
        self.column_name = column_name
        self.data = pd.read_csv(file_name)  
        self.X = torch.tensor(self.data[self.column_name].values[:self.data_size:self.step], dtype=torch.float32).view(-1, 1)
        # self.x = pd.read_csv('ETTh1.csv').HUFL
        return self.data
        
    def set_data_function(self, func): 
        """ This methos set data by function in func

        Args:
            func(x): link to function which return y(x)
        """
        self.X = torch.tensor([func(i) for i in range(self.data_size)], dtype=torch.float32).view(-1, 1)
    def set_data(self, df):
        self.data = df
        self.X = torch.tensor(self.data.values[:self.data_size:self.step], dtype=torch.float32).view(-1, 1)
    def set_model(self, type = "ma"):
        """This method set model.
        type:
            "stl": DLinear winth STL decomposition,
            "ma":  DLinear winth MA decomposition,
            "one_layer": One Layer model.

        Args:
            stl (bool, optional): set model type. Defaults to false.
        """
        def ma_model():
            self.model = DLinearModel(self.input_size, self.output_size)
            self.model_name += "MA"
        def stl_model():
            self.model = DLinearModelSTL(self.input_size, self.output_size)
            self.model_name += "STL"
        def oneLayer_model():
            self.model = OneLayerModel(self.input_size, self.output_size)
            self.model_name += "oneLayer"

        setting = {
            "stl": stl_model,
            "ma": ma_model,
            "one_layer": oneLayer_model
        }
        setting[type]()
            
    def train(self, num_epochs = 100, gpu=False):
        # if gpu:
        #     optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        #     self.model = self.model.to("xpu")
        #     self.model, optimizer = ipex.optimize(self.model, optimizer=optimizer, dtype=torch.float32)
        # else:
        print(datetime.datetime.now())
        
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.L1Loss()
        window_size = self.input_size  
        dataset = MyDataset(self.X, window_size, self.output_size)
        print(f"Len Dataset = {len(dataset)}")
        dataloader = DataLoader(dataset)#, shuffle=True)
        len(dataloader)
        self.train_model(self.model, dataloader, criterion, optimizer, num_epochs=num_epochs)
        torch.save(self.model.state_dict(), self.model_name)
        print(f"Model save as {self.model_name}")
        print(datetime.datetime.now())

    

    def load_modal(self, name):
        self.model_name = name
        self.model.load_state_dict(torch.load(f"{self.model_name}"))
        self.model.eval()
        self.model.parameters
        
        
    def train__with_metrics(self, num_epochs = 1000, data_set = 3000):
        window_size = self.input_size  
        dataset = MyDataset(self.X, window_size, self.output_size)
        dataloader = DataLoader(dataset)#, shuffle=True)
        criterion = nn.L1Loss()
    
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', cooldown = 10)
        # lambda_1 = lambda epoch: 0.65 ** epoch
        #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-10, last_epoch=-1)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(dataloader)*10, gamma=0.9)
        
        
        print("Windows: ", len(dataloader))
        for epoch in range(num_epochs):
            print("Epoch = ", epoch)
            print(f"LR = {optimizer.param_groups[0]['lr']}")
            for X, Y in dataloader:
                if(epoch <= 0):
                    
                    optimizer.zero_grad()
                    #print(X, Y)
                    output = self.model.forward(X).view(1, -1, 1)
                    #print(torch.tensor([output.tolist()]), Y)
                    loss = criterion(output, Y)
                    
                    loss.backward()
                    
                    optimizer.step()
                    
                else:
                    optimizer.zero_grad()
                    #print(X, Y)
                    output = self.model.forward(X).view(1, -1, 1)
                    #print(torch.tensor([output.tolist()]), Y)
                    loss = criterion(output, Y)
                    
                    loss.backward()
                    
                    optimizer.step()
                    scheduler.step(loss)
                
            # result = self.prediction(data_set=data_set)
            self.MAPE(data_set=data_set)
        
    
    def decomposition(self, data_set = 3000):
       
        X = torch.tensor([self.data[self.column_name].values[data_set:data_set+self.input_size]], dtype=torch.float32).view(1, -1, 1)
        # self.x = pd.read_csv('ETTh1.csv').HUFL
        
        print(X)
        
            
            #print(X, Y)
        trend, seasonal = self.model.decompos(X)
        print(trend, seasonal)
        summa = trend+seasonal
        return trend.reshape(1, 1, -1).tolist()[0][0], seasonal.reshape(1, 1, -1).tolist()[0][0], summa.reshape(1, 1, -1).tolist()[0][0]
            #print(torch.tensor([output.tolist()]), Y)
            
            
            
                
            # result = self.prediction(data_set=data_set)
            
        
        
        
        
        
    def prediction(self, data_set):
        pred = self.input_size
        X_f = torch.tensor(self.data[self.column_name].values[data_set-pred:data_set-pred+pred*self.step:self.step], dtype=torch.float32).view(-1, 1)
        #print(X_f)
        X_t = X_f.tolist()
        # predicted_values = []
        X = torch.tensor([X_t])
        prediction = self.model(X)
        #print(prediction)
        self.predicted_values = prediction.tolist()[-1][-1]
        return self.predicted_values
    def prediction_custom_data(self, func):
        pred = self.input_size
        X_f = torch.tensor([func(i) for i in range(self.data_set-pred, self.data_set-pred+pred*self.step,self.step)], dtype=torch.float32).view(-1, 1)
        #print(X_f)
        X_t = X_f.tolist()
        # predicted_values = []
        X = torch.tensor([X_t])
        prediction = self.model(X)
        #print(prediction)
        self.predicted_values = prediction.tolist()[-1][-1]
        return self.predicted_values
    def MAE(self, data_set):
        i = [self.data_set+1+i*self.step for i in range(self.output_size)]

        actual = np.array([self.data[self.column_name].values[k] for k in i])
        prediction = self.prediction(data_set=data_set)

        l1_loss = abs(actual - prediction)
        mae_cost = l1_loss.mean()
        print("MAE:", mae_cost)
        
    def MAPE(self, data_set):
        i = [data_set+1+i*self.step for i in range(self.output_size)]

        actual = np.array([self.data[self.column_name].values[k] for k in i])
        prediction = self.prediction(data_set=data_set)

        l1_loss = abs(actual+1 - prediction-1)/abs(actual+1)

        """
        Output:
        [0 1 2 2]
        """

        mape_cost = np.sum(l1_loss)/self.output_size
        print("MAPE:", mape_cost)