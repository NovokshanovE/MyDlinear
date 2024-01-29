from model import *


class MyDataset(TensorDataset):
    def __init__(self, data, window):
        self.data = data
        self.window = window

    def __getitem__(self, index):
        x = self.data[index:index+self.window]
        return x

    def __len__(self):
        return len(self.data) - self.window
    
data_size  = 1000
step = 1
data = pd.read_csv('ETTh1.csv')  
X = torch.tensor(data['HUFL'].values[:data_size:step], dtype=torch.float32).view(-1, 1)
x = pd.read_csv("ETTh1.csv").HUFL

input_size = 1  
seasonality_size = 1
trend_size = 1
learning_rate = 0.0001

model = DLinearModel(input_size)
criterion = nn.L1Loss()
optimizer = optim.Rprop(model.parameters(), lr=learning_rate)#Adam(model.parameters(), lr=learning_rate)


window_size = 5  # Размер окна для rolling window forecasting
#переписать

dataset = MyDataset(X, window_size)

#print(x.rolling(window_size))

dataloader = DataLoader(dataset)#, shuffle=True)
print("---------")
# print(dataset.data)
# print("---------")
# for i in dataloader:
#     print(i)


train_model(model, dataloader, criterion, optimizer)



initial_values = torch.cat([X[-1]]).reshape(1, 1, -1)
m = 100 #на сколько шагов предсказать
future_predictions = predict_future_values(model, initial_values, m)

print("Future Predictions:", future_predictions)
plt.plot(data['HUFL'].values[:data_size])
#plt.plot(, )
pred = data['HUFL'].values[data_size-1]
for i in range(m):
    
    plt.scatter(data_size+1+i, future_predictions[i], c='g')
    
plt.show()