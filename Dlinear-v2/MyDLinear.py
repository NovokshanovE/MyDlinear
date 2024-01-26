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
# from decomposition import DecompositionLayer
torch.set_num_threads(9)

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

# %% [markdown]
# ## Определение параметров модели

# %%
input_size = 30
pred = input_size
learning_rate = 0.0001
output_size = 100
data_size  = 5000
step = 1
m = 10 #на сколько шагов предсказать
data_set = 10000
column_name = "HUFL"
model_name = f"update_model_v8_L1_SGD_{column_name}_input{input_size}_output{output_size}"


# %% [markdown]
# ## Класс деокмпозиции временного ряда для получение сезонной и трендовой составляющей

# %%
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

# %% [markdown]
# ## Класс DLinear и функция для обучения модели

# %%
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


def train_model(model, dataloader, criterion, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        print("Epoch = ", epoch)
        for X, Y in dataloader:
            
            optimizer.zero_grad()
            output = model.forward(X)
            #print(torch.tensor([output.tolist()]), Y)
            loss = criterion(output, Y)
            loss.backward()
            optimizer.step()


# %% [markdown]
# 

# %%

data = pd.read_csv('ETTh1.csv')  
X = torch.tensor(data[column_name].values[:data_size:step], dtype=torch.float32).view(-1, 1)
x = pd.read_csv("ETTh1.csv").HUFL

# %%
model = DLinearModel(input_size, output_size)
criterion = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
#optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#TODO: переписать dataloader для правильно обучения

window_size = input_size  # Размер окна для rolling window forecasting
#переписать

dataset = MyDataset(X, window_size, output_size)

#print(x.rolling(window_size))

dataloader = DataLoader(dataset)#, shuffle=True)

# %%
len(dataloader)

# %%
train_model(model, dataloader, criterion, optimizer)


# %%
torch.save(model.state_dict(), model_name)

# %%
model.load_state_dict(torch.load(model_name))
model.eval()

# %%
model.parameters

# %%
def predict_future_values(model, X_f, window_size, m):
    #predicted_values = initial_values.clone().detach().view(-1).tolist()
    predicted_values = []
    
    for i in range(m):
        
        
        #проверить что модель принимает нужный набор значений
        #last_X = torch.tensor([predicted_values[-1]], dtype=torch.float32).view(-1, 1)
        # last_X_t = torch.tensor([predicted_values[-1]], dtype=torch.float32).view(-1, 1)
        
        prediction = model(X_f)

        
        predicted_values.append(prediction.tolist()[-1][-1][-1])
        if(window_size > i):
            
            X_f = torch.tensor(data['HUFL'].values[data_size-window_size+i+1::step] + predicted_values, dtype=torch.float32).reshape(1, 1, -1)
        else:
            X_f = torch.tensor(predicted_values, dtype=torch.float32).reshape(1, 1, -1)
            
        

    return predicted_values

# %%
initial_values = torch.cat([X[-input_size:]]).reshape(1, 1, -1)

pred = input_size
X_f = torch.tensor(data[column_name].values[data_set-pred:data_set-pred+pred*step:step], dtype=torch.float32).view(-1, 1)
print(X_f)
dataset_f = MyDataset(X_f, pred, output_size)
print(dataset_f.data)
#print(x.rolling(window_size))

dataloader_f = DataLoader(dataset_f)#, shuffle=True)

X_f


# %%
X_t = X_f.tolist()
X_t

# %%

predicted_values = []




    
X = torch.tensor([X_t])
prediction = model(X)
predicted_values = prediction.tolist()[-1][-1]

      
    


future_predictions = predicted_values

# %%
predicted_values

# %%
future_predictions

# %%
len(future_predictions)

# %%

print("Future Predictions:", future_predictions)
time = [data_set-output_size*step+i*step for i in range(2*output_size)]
print(output_size)
plt.rcParams["figure.figsize"] = (12,9)
plt.rcParams.update({'font.size': 14})
plt.plot(time, data[column_name].values[data_set-output_size*step:data_set+(output_size)*step:step])
#plt.plot(, )
pred = data[column_name].values[data_set-1]
time = [data_set+1+i*step for i in range(output_size)]

    
plt.plot(time, future_predictions[::], 'r--')
#plt.title(model_name)
plt.xlabel('Временные точки', fontsize=14)
plt.ylabel('HUFL', fontsize=14)
plt.show()
#plt.savefig(model_name+"2", dpi=1000)




# %%
i = [data_set+1+i*step for i in range(output_size)]

import numpy as np

actual = np.array([data[column_name].values[k] for k in i])
prediction = np.array(future_predictions)

l1_loss = abs(actual - prediction)

"""
Output:
[0 1 2 2]
"""

mae_cost = l1_loss.mean()
print(mae_cost)

# %%
i = [data_set+1+i*step for i in range(output_size)]

import numpy as np

actual = np.array([data[column_name].values[k] for k in i])
prediction = np.array(future_predictions)

l1_loss = abs(actual+1 - prediction-1)/abs(actual+1)

"""
Output:
[0 1 2 2]
"""

mape_cost = np.sum(l1_loss)/output_size
print(mape_cost)

# %%
MAE_5 = [9.75, 9.42, 8.06]
MAE_15 = [7.37, 6.20, 5.99]
MAE_30 = [6.99, 4.28, 5.87]

MAPE_5 = [1.00, 1.53,  0.92]
MAPE_15 = [2.04, 1.01, 1.21]
MAPE_30 = [1.83, 0.98, 1.17]

steps = [2,12,23]

plt.rcParams["figure.figsize"] = (21,9)
plt.plot(steps, MAE_5, label='размер окна = 5')
plt.plot(steps, MAE_15, label='размер окна = 15')
plt.plot(steps, MAE_30, label='размер окна = 30')
# plt.title("MAE")
plt.xlabel('Шаг', fontsize=14)
plt.ylabel('MAE', fontsize=14)
plt.legend()
plt.show()


plt.plot(steps, MAPE_5, label='размер окна = 5')
plt.plot(steps, MAPE_15, label='размер окна = 15')
plt.plot(steps, MAPE_30, label='размер окна = 30')
# plt.title("MAPE")
plt.xlabel('Шаг', fontsize=14)
plt.ylabel('MAPE', fontsize=14)
plt.legend()
plt.show()

# %% [markdown]
# ## Тест 1
# ![image.png](attachment:image.png)
# ![image-2.png](attachment:image-2.png)

# %% [markdown]
# ## Тест2
# ![](attachment:image.png)
# ![image-2.png](attachment:image-2.png)
# 

# %% [markdown]
# ## Тест 3
# ![image-1.png](attachment:image.png)
# ![image-2.png](attachment:image-2.png)

# %% [markdown]
# ## Тест 4
# 


