import time

import numpy as np
import torch
torch.set_num_threads(1)
INDEX = 10000
NELE = 1000
a = torch.rand(INDEX, NELE)
index = np.random.randint(INDEX-1, size=INDEX*8)
b = torch.from_numpy(index)

start = time.time()
for _ in range(10):
    res = a.index_select(0, b)
print("the number of cpu threads: {}, time: {}".format(torch.get_num_threads(), time.time()-start))