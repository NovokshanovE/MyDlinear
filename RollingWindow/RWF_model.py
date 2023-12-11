import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd
import matplotlib.pyplot as plt
from DLinear.test_class_1 import *
data = pd.read_csv("ETTh1.csv")
ts = data.HUFL
ts.head(10)

ts.rolling(2).mean().head(10)

print(len(ts))

ts_tensor = torch.Tensor(ts).reshape(1, 1, -1)

kernel = [0.5, 0.5]
kernel_tensor = torch.Tensor(kernel).reshape(1, 1, -1)
F.conv1d(ts_tensor, kernel_tensor)

X = data.HUFL
X_tensor = torch.Tensor(X).reshape(1,1,-1)

y = data.HUFL.rolling(5).mean()
y = y[4:, ].to_numpy()
y_tensor = torch.Tensor(y).reshape(1,1,-1)
y_tensor

model = DLinearModel(1, 2, 2)