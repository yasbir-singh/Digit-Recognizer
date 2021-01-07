import math
import numpy as np

def sigmd(value):
    if -value > np.log(np.finfo(type(value)).max):
        return 0.0
    a = np.exp(-value)
    return 1.0 / (1.0 + a)

sig = np.vectorize(sigmd)