
# import torch
import random

from sympy import N, false
from Dlinear_v2.MyDLinear import DLinear
import datetime
import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
# from torchvision import models
# from torchsummary import summary

def func(x):
    return 0.01*np.sin(x/10)#1.3*x+10#np.sin(x)/100
data_set = 14100
input_size = 100
output_size = 100
learning_rate = 0.001
step = 1
data_size = 14000
column_name = 'value'
dataset_name = 'dataset_1'
type = "ma"
dLinear = DLinear(data_set, input_size, output_size, step = 1, data_size = data_size, column_name=column_name, dataset_name = dataset_name)
data = dLinear.data_reader(file_name=dataset_name +'.csv', column_name=column_name)
# data = None
# print(summary(dLinear))



def train():
    #data = dLinear.set_data(func=func)
    dLinear.set_model(type=type)
    dLinear.load_modal("dlinear(2024-02-23_16-54-53-039394)_dataset_1_value_input100_output100MA")
    # dLinear.train__with_metrics(data_set=data_set, num_epochs=1000)
    # dLinear.train(num_epochs =  1000, gpu=True)


def delta_horisontal_line_mae(predict, real)-> int:
    res = 0
    for i in predict:
        res += np.abs(i-real)
    print(f"Delta MAE: {res/len(predict)}")
    return res/len(predict)

def delta_horisontal_line_mape(predict, real)-> int:
    res = 0
    for i in predict:
        res += np.abs((i-real)/real)
    print(f"Delta MAPE: {res/len(predict)}")
    return res/len(predict)

def test1():
    future_predictions = dLinear.prediction(data_set)
    print("Future Predictions:", future_predictions)
    time = [data_set-output_size*step+i*step for i in range(2*output_size)]
    print(output_size)
    plt.rcParams["figure.figsize"] = (12,9)
    plt.rcParams.update({'font.size': 14})
    plt.plot(time, data[column_name].values[data_set-output_size*step:data_set+(output_size)*step:step])
    #plt.plot(, ) 
    # pred = data[column_name].values[data_set-1]
    time = [data_set+1+i*step for i in range(output_size)]

        
    plt.plot(time, future_predictions[::], 'r--')
    #plt.title(model_name)
    plt.xlabel('Временные точки', fontsize=14)
    plt.ylabel(column_name, fontsize=14)
    dLinear.MAE(data_set=data_set)
    dLinear.MAPE(data_set=data_set)
    delta_horisontal_line_mae(future_predictions, data[column_name].values[data_set])
    delta_horisontal_line_mape(future_predictions, data[column_name].values[data_set])
    plt.ylim([data[column_name].values[data_set]-10,data[column_name].values[data_set]+10])
    plt.savefig(f'results/rw_results/test_{datetime.datetime.now().date()}_{datetime.datetime.now().hour}_{datetime.datetime.now().minute}_{dataset_name}_{column_name}_{data_set}_{type}')
    # plt.show()
    #plt.savefig(model_name+"2", dpi=1000)
    
    
def test2():
    future_predictions = dLinear.prediction_custom_data(func)
    print("Future Predictions:", future_predictions)
    time = [data_set-output_size*step+i*step for i in range(2*output_size)]
    print(output_size)
    plt.rcParams["figure.figsize"] = (12,9)
    plt.rcParams.update({'font.size': 14})
    plt.plot(time, [func(data_set-output_size*step+i*step) for i in range(2*output_size)])
    #plt.plot(, )


    time = [data_set+i*step for i in range(output_size)]
    plt.plot(time, future_predictions[::], 'r--')
    #plt.title(model_name)
    plt.xlabel('Временные точки', fontsize=14)
    plt.ylabel('HUFL', fontsize=14)
    plt.show()
    
def test3():
    future_predictions = dLinear.prediction(data_set)
    print("Future Predictions:", future_predictions)
    time = [data_set-output_size*step+i*step for i in range(2*output_size)]
    print(output_size)
    plt.rcParams["figure.figsize"] = (12,9)
    plt.rcParams.update({'font.size': 14})
    plt.plot(time, data[column_name].values[data_set-output_size*step:data_set+(output_size)*step:step])
    #plt.plot(, )
    # pred = data[column_name].values[data_set-1]
    time = [data_set+1+i*step for i in range(output_size)]

        
    plt.plot(time, future_predictions[::], 'r--')
    #plt.title(model_name)
    plt.xlabel('Временные точки', fontsize=14)
    plt.ylabel('HUFL', fontsize=14)
    plt.show()
    #plt.savefig(model_name+"2", dpi=1000)
    
    dLinear.MAE()
    dLinear.MAPE()
    
def test_decomposition():
    data_set =3000
    trend, season, summa = dLinear.decomposition(data_set)
    
    plt.rcParams["figure.figsize"] = (12,9)
    plt.rcParams.update({'font.size': 14})
    time = [i for i in range(output_size)]
    plt.plot(time, data[column_name].values[data_set:data_set+(output_size)])
    print()
    plt.plot(time, trend, 'g-.')
        
    plt.plot(time, season, 'g--')
    plt.plot(time, summa, 'r--')
    test_data = data[column_name].values[data_set:data_set+(output_size)].tolist()
    
    test_data = pd.Series(
    test_data)#, index = pd.interval_range(start=data_set, end= data_set+len(test_data), periods=len(test_data))
    
    print(test_data)
    print(test_data.describe())
    stl = STL(test_data, seasonal=13, period=10)
    
    res = stl.fit()
    # res.plot()
    seas = res.seasonal
    print(seas.head)
    plt.plot(time, seas, 'b--')
    plt.plot(time, res.trend, 'b-.')
    plt.plot(time, res.trend + seas + res.resid, 'p-.')
    
    plt.show()
# test_decomposition()


def random_walk(
    df_size = 1000, start_value=0, threshold=0.5, 
    step_size=1, min_value=-np.inf, max_value=np.inf
):
    df = pd.DataFrame(index = [i for i in range(df_size)])
    previous_value = start_value
    for index, row in df.iterrows():
        if previous_value < min_value:
            previous_value = min_value
        if previous_value > max_value:
            previous_value = max_value
        probability = random.random()
        if probability >= threshold:
            df.loc[index, 'value'] = previous_value + step_size
        else:
            df.loc[index, 'value'] = previous_value - step_size
        previous_value = df.loc[index, 'value']
        
    return df


def create_rw_ts(name: str, graph: bool, step_size: int):

    res = random_walk(df_size=20000, step_size=step_size, threshold=0.5, start_value=10)
    res.to_csv(path_or_buf=name+".csv")
    
    
    if graph:
        plt.rcParams["figure.figsize"] = (12,9)
        plt.rcParams.update({'font.size': 14})
        plt.plot(res, 'g')
        plt.savefig("dataset_view")


def plot(file_name, column_name):
    X = pd.read_csv(file_name).value
    # X = torch.tensor(self.data[self.column_name].values[:self.data_size:self.step], dtype=torch.float32).view(-1, 1)
    plt.rcParams["figure.figsize"] = (12,9)
    plt.rcParams.update({'font.size': 14})
    plt.plot(X[:11000], 'g')
    plt.grid(visible=True)
    plt.xlabel('Временные точки', fontsize=14)
    plt.ylabel(column_name, fontsize=14)
    plt.savefig("dataset_view")





def train_model(model: str, test_preferences: dict, rw_range: list, dataset_generation: bool, train_preferences: dict):
    """About train_model

    Args:
        model (str): type of model. Example: "ma", "stl", "base";
        set_data (int): set data to test;
        size_data (int): set data to learning;
        rw_range (list): range to create new dataset for tests.
    """
    if dataset_generation:

        for i in range(rw_range[0], rw_range[1]):
            """ 
                Generate/create new test dataset
            """
            create_rw_ts(name=f"test_ds_({i/10})", graph=false, step_size=i/10)
    if train_preferences:
        data_set = train_preferences["set_data"]
        input_size = train_preferences["input_size"]
        output_size = train_preferences["output_size"]
        learning_rate = train_preferences["learning_rate"]
        step = train_preferences["step"]
        data_size = train_preferences["data_size"]
        column_name = train_preferences["column_name"]
        dataset_name = 'dataset_1'
        type = train_preferences["model_type"]
        dLinear = DLinear(data_set, input_size, output_size, step = 1, data_size = data_size, column_name=column_name, dataset_name = dataset_name)
        data = dLinear.data_reader(file_name=dataset_name +'.csv', column_name=column_name)
        #data = dLinear.set_data(func=func)
        dLinear.set_model(type=type)
        dLinear.load_modal("dlinear(2024-02-23_16-54-53-039394)_dataset_1_value_input100_output100MA")
        # dLinear.train__with_metrics(data_set=data_set, num_epochs=1000)
        # dLinear.train(num_epochs =  1000, gpu=True)
    



if __name__ == "__main__":
    pass
    # dLinear.set_model(stl=False)
    #plot("dataset_1.csv", "value")
    # train()
    # test1()