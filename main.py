
# import torch
import random
from turtle import title

from sympy import N, false
# from traitlets import dlink
from Dlinear_v2.MyDLinear import DLinear
import datetime
import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
# from torchvision import models
# from torchsummary import summary
def base_test():
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

    res = random_walk(df_size=15000, step_size=step_size, threshold=0.5, start_value=10)
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
    plt.savefig(f"dataset_view_{file_name.split('.')[0]+file_name.split('.')[1]}")
    plt.close()


def delta_horisontal_line_mae(predict, real, output)-> int:
        res = 0
        for i in predict:
            res += np.abs(i-real)
        if output:
            print(f"Delta MAE: {res/len(predict)}")
        return res/len(predict)

def delta_horisontal_line_mape(predict, real, output)-> int:
    res = 0
    for i in predict:
        res += np.abs((i-real)/real)
    if output: 
        print(f"Delta MAPE: {res/len(predict)}")
    return res/len(predict)

def delta_horisontal_line_mse(predict, real, output)-> int:
    res = 0
    for i in predict:
        res += np.power((i-real)/real, 2)
    if output:
        print(f"Delta MAPE: {res/len(predict)}")
    return res/len(predict)
def model_settings(preferences, cur_d_size, cur_step):
    dataset_name = f"test_ds_({cur_step/10})"
    model_name = f"dlinear_(name_ds{dataset_name})_size{cur_d_size}"
    # set_data = preferences["set_data"]
    input_size = preferences["input_size"]
    output_size = preferences["output_size"]
    # learning_rate = preferences["learning_rate"]
    # step = preferences["step"]
    # data_size = train_preferences["data_size"]
    column_name = preferences["column_name"]
    type = preferences["model_type"]
    def ma_model(model_name):
                    
        model_name += "MA"
        return model_name
    
    def stl_model(model_name):
        
        model_name += "STL"
        return model_name
    
    def oneLayer_model(model_name):
        
        model_name += "oneLayer"
        return model_name

    setting = {
        "stl": stl_model,
        "ma": ma_model,
        "one_layer": oneLayer_model
    }
    model_name = setting[type](model_name)
    dLinear = DLinear(input_size, output_size, step = 1, data_size = cur_d_size, column_name=column_name, dataset_name = dataset_name)
    data = dLinear.data_reader(file_name=dataset_name +'.csv', column_name=column_name)
    dLinear.set_model(type=type)
    dLinear.load_modal("saving_models/group_of_models_by_step_size/"+model_name)
    return dLinear, data

def train_model(test_preferences: dict  = None, rw_range: list = [1,10], dataset_generation: bool = False, train_preferences: dict = None):
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
            for n in range(10):
                create_rw_ts(name=f"test_ds_({i/10})_{n}", graph=False, step_size=i/10)
    if train_preferences:
        data_set = train_preferences["set_data"]
        input_size = train_preferences["input_size"]
        output_size = train_preferences["output_size"]
        learning_rate = train_preferences["learning_rate"]
        step = train_preferences["step"]
        # data_size = train_preferences["data_size"]
        column_name = train_preferences["column_name"]
        type = train_preferences["model_type"]
        for d_size in train_preferences["data_size"]:
            for i in range(rw_range[0], rw_range[1]):
                for n in range(10):
                    dataset_name = f"test_ds_({i/10})_{n}"
                    dLinear = DLinear(data_set, input_size, output_size, step = step, data_size = d_size, column_name=column_name, dataset_name = dataset_name, learning_rate=learning_rate)
                    # dataset_name = 'dataset_1'
                    data = dLinear.data_reader(file_name=dataset_name +'.csv', column_name=column_name)
                    #data = dLinear.set_data(func=func)
                    dLinear.set_model(type=type)
                    # dLinear.load_modal("dlinear(2024-02-23_16-54-53-039394)_dataset_1_value_input100_output100MA")
                    # dLinear.train__with_metrics(data_set=data_set, num_epochs=1000)
                    dLinear.train(num_epochs =  1000, gpu=True)
    

    

def test_dependencies(test_preferences: dict  = None, rw_range: list = [1,10]):
    set_data = test_preferences["set_data"]
    input_size = test_preferences["input_size"]
    output_size = test_preferences["output_size"]
    learning_rate = test_preferences["learning_rate"]
    step = test_preferences["step"]
    # data_size = train_preferences["data_size"]
    column_name = test_preferences["column_name"]
    type = test_preferences["model_type"]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 20))
    fig1, (ax1_25, ax2_75) = plt.subplots(2, 1, figsize=(10, 20))
    for i in range(rw_range[0], rw_range[1]):
        mae = []
        mse = []
        mape = []
        q_25 = []
        q_75 = []
        
        for d_size in test_preferences["data_size"]:
            
            mae_mean = []
            mape_mean = []
            mse_mean = []
            
            for data_set in range(set_data[0],set_data[1],set_data[2]):

                # dataset_name = f"test_ds_({i/10})"
                # model_name = f"dlinear_(name_ds{dataset_name})_size{d_size}"
                # def ma_model(model_name):
                    
                #     model_name += "MA"
                #     return model_name
                
                # def stl_model(model_name):
                    
                #     model_name += "STL"
                #     return model_name
                
                # def oneLayer_model(model_name):
                    
                #     model_name += "oneLayer"
                #     return model_name

                # setting = {
                #     "stl": stl_model,
                #     "ma": ma_model,
                #     "one_layer": oneLayer_model
                # }
                # model_name = setting[type](model_name)
                # dLinear = DLinear(data_set, input_size, output_size, step = 1, data_size = d_size, column_name=column_name, dataset_name = dataset_name)
                # data = dLinear.data_reader(file_name=dataset_name +'.csv', column_name=column_name)
                dLinear, data = model_settings(test_preferences, d_size, i)
                
                # print(f"Quantile MAE 25 step={i/10}: {np.quantile(data[column_name].values, 0.25)}")
                # print(f"Quantile MAE 75 step={i/10}: {np.quantile(data[column_name].values, 0.75)}")
                
                future_predictions = dLinear.prediction(data_set)
                # print(f"--------------\nstep={i/10} d_size={d_size} set_data= {data_set}")
                mae_mean.append(delta_horisontal_line_mae(future_predictions, data[column_name].values[data_set], false))
                mape_mean.append(delta_horisontal_line_mape(future_predictions, data[column_name].values[data_set], false))
                mse_mean.append(delta_horisontal_line_mse(future_predictions, data[column_name].values[data_set], false))
            q_25.append(np.quantile(data[column_name].values[:d_size], 0.25))
            print(f"Quantile 25 step={i/10} size={d_size}: {q_25[-1]}")
            q_75.append(np.quantile(data[column_name].values[:d_size], 0.75))
            print(f"Quantile 75 step={i/10} size={d_size}: {q_75[-1]}")
            mae.append(sum(mae_mean)/len(mae_mean))
            mape.append(sum(mape_mean)/len(mape_mean))
            mse.append(sum(mse_mean)/len(mse_mean))

        
        print(f"Quantile 25 step={i/10}: {np.quantile(data[column_name].values, 0.25)}")
        print(f"Quantile 75 step={i/10}: {np.quantile(data[column_name].values, 0.75)}")
        # print(f"Quantile MAE 25 step={i/10}: {np.quantile(mae, 0.25)}")
        # print(f"Quantile MAE 75 step={i/10}: {np.quantile(mae, 0.75)}")
        # print(f"Quantile MAPE 25 step={i/10}: {np.quantile(mape, 0.25)}")
        # print(f"Quantile MAPE 75 step={i/10}: {np.quantile(mape, 0.75)}")
        # print(f"Quantile MSE 25 step={i/10}: {np.quantile(mse, 0.25)}")
        # print(f"Quantile MSE 75 step={i/10}: {np.quantile(mse, 0.75)}")

        ax1_25.plot(test_preferences["data_size"],q_25, label=f'{i/10}')
        ax2_75.plot(test_preferences["data_size"],q_75, label=f'{i/10}')
        ax1_25.legend()
        ax2_75.legend()
        ax1_25.set_xlabel("data size")
        ax2_75.set_xlabel("data size")
        ax1_25.set_ylabel("quantile25")
        ax2_75.set_ylabel("quantile75")
        fig1.savefig("quantile")

        
        # ax1.plot(test_preferences["data_size"],mae, label=f'{i/10}')
        # ax2.plot(test_preferences["data_size"],mse, label=f'{i/10}')
        # ax3.plot(test_preferences["data_size"],mape, label=f'{i/10}')
        # ax1.legend()
        # ax2.legend()
        # ax3.legend()
        # ax1.set_xlabel("data size")
        # ax2.set_xlabel("data size")
        # ax3.set_xlabel("data size")
        # ax1.set_ylabel("deviation")
        # ax2.set_ylabel("deviation")
        # ax3.set_ylabel("deviation")
        # ax1.set_title("mae")
        # ax2.set_title("mse")
        # ax3.set_title("mape")
        # # fig.colorbar()
        # fig.savefig("tests_4")
        # fig.close()
    
    print(q_25, q_75)
    


# def quantile_for_dataset()

if __name__ == "__main__":
    # pass
    # dLinear.set_model(stl=False)
    #plot("dataset_1.csv", "value")
    train_p = {
        "set_data": 15000,
        "input_size": 100,
        "output_size": 100,
        "learning_rate": 0.00001,
        "step": 1,
        "data_size": [7000, 8000, 9000, 10000, 11000, 12000],
        "column_name": "value",
        "model_type": "ma",
    }
    test_p = {
        "set_data": (13000, 19700, 10),
        "input_size": 100,
        "output_size": 100,
        "learning_rate": 0.00001,
        "step": 1,
        "data_size": [7000, 8000, 9000, 10000, 11000, 12000],
        "column_name": "value",
        "model_type": "ma",
    }
    train_model(train_preferences=train_p, rw_range=[1,5], dataset_generation=True)
    # test1()

    # test_dependencies(test_preferences=test_p, rw_range=[1,5])

    # for i in range(1, 5):
    #     plot(f"test_ds_({i/10}).csv", "value")

    # a = np.array([10, 7, 4])

    # print(np.quantile(a, 0.75))