import numpy as np
import keras

def MSEs(true, predict):
    return np.average((true- predict)**2)

def MAEs(true, predict):
    return np.average(abs(true - predict))

def Absolute_Error_percentage(true, predict):
    return np.average(abs(true - predict) / true) * 100