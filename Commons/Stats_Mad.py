import sys
import os
sys.path.append(os.path.abspath(r'../Commons'))
import Configuration as co
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def analytical_OLS_betas(x, y):
    ## Getting the number of predictor variables and number of data points
    num_of_predictors = x.shape[0]
    num_of_data_pts = x.shape[1]
    
    ## Adding a column of 1's so as to get the intercept
    x = np.append(x, [np.repeat(1, num_of_data_pts)], axis = 0)
    
    ## Note that every element in x as of now represnents the entire predictor feature series
    ## Transposing x for every element to represent feature value at that point in time
    x = x.transpose()
    
    ## Building x transpose times x matrix
    xTx = np.matmul(x.transpose(), x)
    ## Check if xTx is invertible
    try:
        inv = linalg.inv(xTx)
    except MatrixError:
        print("Matrix is singular and can't be inverted")
        
    ## Using the analytical formula to get betas
    betas = np.matmul(np.matmul(inv, x.transpose()), y)
    
    return betas


### Building a dummy series
dummy_x = np.asarray([np.asarray(range(0,10)), np.asarray([1, 1, 2, 3, 5, 8, 13, 21, 34, 55])]) 
dummy_y = 3*dummy_x[0] - 2*dummy_x[1] + 55

dummy_betas = analytical_OLS_betas(dummy_x, dummy_y)
dummy_betas