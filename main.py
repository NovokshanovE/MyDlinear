from Dlinear_v2.MyDLinear import DLinear

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
def func(x):
    return np.sin(x)#1.3*x+10#np.sin(x)/100
data_set = 4000
input_size = 100
output_size = 100
learning_rate = 0.001
step = 1
data_size = 3000
column_name = 'HUFL'
dLinear = DLinear(data_set, input_size, output_size, step = 1, data_size = 3000, column_name=column_name)
data = dLinear.data_reader()
dLinear.set_model()
# dLinear.load_modal()
dLinear.train()




def test1():
    future_predictions = dLinear.prediction()
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
    
    
test1()
