import sys
import os
sys.path.append(os.path.abspath(r'../Commons'))
import Configuration as co
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

import numpy as np



class interpolation: 
    
    def __init__(self,vec,i):
        self.vec=np.array(vec)
        self.i=i
        
dummy_betas    def Linear(self):
        if (self.i>len(self.vec) or self.i<0 ):
            return (print("Out of bounds!"))
        elif(len(self.vec)==1):
            return(print("Provide atleast 2 elements"))
        else:
            slopes=self.vec[1:len(self.vec)] - self.vec[0:(len(self.vec)-1)]
            return ( self.vec[int(self.i)] + (self.i - int(self.i) ) * slopes[int(self.i)])
