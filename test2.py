from multiprocessing import Pool
import time
import numpy as np
import matplotlib.pyplot as plt
def square_list(input_list):
    square_sum = 0
    for i in input_list:
        square_sum += i ** 2
    return square_sum
def square(list_element):
    return list_element ** 2

if __name__ == '__main__':
    
    list_lengths = range(10, 100000, 10000)
    plt.figure()
    time_sg = []
    time_mp = []
    for list_length in list_lengths:
        np.random.seed(2)
        input_list = np.random.randn(list_length)
        time1 = time.time()
        square_sum = square_list(input_list)
        time2 = time.time()
        p = Pool(20)
        square_sum = sum(p.map(square, input_list))
        time3 = time.time()
        time_sg.append(time2 - time1)
        time_mp.append(time3 - time2)
    plt.plot(list_lengths, time_sg, color = 'black', label = 'Single Process')
    plt.plot(list_lengths, time_mp, color = 'red', label = 'Multi Process')
    plt.legend()
    plt.show()