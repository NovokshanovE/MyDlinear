



import random
import numpy as np

a = random.choices([-1,0,1], weights=[40,20,40], k=1000000)
print(sum(a)/len(a))