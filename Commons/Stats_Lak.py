import sys
import os
sys.path.append(os.path.abspath(r'../Commons'))
import Configuration as co
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def LassoSoftThresholding(alpha, data_x, data_y):
    
    data_x = data_x.to_numpy()
    data_y = data_y.to_numpy()
    
    data_y = data_y - np.mean(data_y)
    data_x = data_x / (np.linalg.norm(data_x,axis = 0))
    
    betaOLS = np.dot(np.dot(np.linalg.inv(np.dot(data_x.T , data_x)),data_x.T) , data_y)

    numBetas = len(betaOLS)
    
    for i in range(100*numBetas):
       
       index = i%numBetas 
       
       Xj = data_x[:, index].reshape(-1,1)
       y_pred  = data_x@betaOLS 
       
       rho = (data_y - y_pred + betaOLS[index]*Xj)@Xj
             
       betaj = SoftThresholding(alpha, rho[0][0])
       betaOLS[index] = betaj
    
    return betaOLS   


def SoftThresholding(alpha, rho):
    
    if rho > alpha :
       return rho-alpha
    elif rho < -alpha:
       return rho+alpha
    else:
       return 0
       
filepath = co.CommonFiles  +  r'/prostate.csv'
data = pd.read_csv(filepath,index_col = 'Id')
datatrain = data[data['train'] == 'T']
datatest = data[data['train'] == 'F']

alpha = [x/100 for x in range(10,300)]
rms = []
test_x = datatest.drop(['lpsa','train'], axis=1).to_numpy()
test_x = test_x / (np.linalg.norm(test_x,axis = 0))
test_y = datatest['lpsa'].to_numpy()
test_y = test_y - np.mean(test_y)
for alp in alpha:
    beta = LassoSoftThresholding( alp , datatrain.drop(['lpsa','train'], axis = 1) , datatrain['lpsa'] )       
    rms.append((test_y - test_x@beta).T@(test_y - test_x@beta))
    
plt.plot(alpha, rms, 'ro')
plt.show()


