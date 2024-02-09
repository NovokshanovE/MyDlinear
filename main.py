from Dlinear_v2.MyDLinear import DLinear

import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
def func(x):
    return 0.01*np.sin(x/10)#1.3*x+10#np.sin(x)/100
data_set = 5000
input_size = 100
output_size = 100
learning_rate = 0.001
step = 1
data_size = 3000
column_name = 'HUFL'
dLinear = DLinear(data_set, input_size, output_size, step = 1, data_size = 3000, column_name=column_name)
data = dLinear.data_reader()
#data = dLinear.set_data(func=func)
dLinear.set_model()
dLinear.load_modal("model_10000epoch")
# dLinear.train__with_metrics(data_set=data_set, num_epochs=1000)
#dLinear.train()




def test1():
    future_predictions = dLinear.prediction(data_set)
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
    
    dLinear.MAE()
    dLinear.MAPE()
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
    pred = data[column_name].values[data_set-1]
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
    plt.plot(time, trend, 'b--')
        
    plt.plot(time, season, 'g--')
    plt.plot(time, summa, 'r--')
    test_data = data[column_name].values[data_set:data_set+(output_size)].tolist()
    
    test_data = pd.Series(
    test_data, index = pd.interval_range(start=data_set, end= data_set+len(test_data), periods=len(test_data))
    )
    print(test_data)
    print(test_data.describe())
    stl = STL(test_data, seasonal=13, period=100)
    
    res = stl.fit()
    # res.plot()
    seas = res.seasonal
    print(seas.head)
    plt.plot(time, seas, 'g-.')
    
    plt.show()
test_decomposition()
