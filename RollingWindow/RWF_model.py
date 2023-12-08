import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("shampoo.csv")
ts = data.Sales
ts.head(10)