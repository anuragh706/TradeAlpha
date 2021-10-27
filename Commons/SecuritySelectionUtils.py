import sys
import os
sys.path.append(os.path.abspath(r'../Commons'))

import Configuration as co
import BaseUtils as bu
import MarketUtils as mu
import pandas as pd
import numpy as np

# vanilla Risk Parity Security selection of a bond and Stock Index
class VanillaRPS:
    
    SecurityObjs = []
    
    def __init__(self, kwargs): #   kwargs = {securityno : {filename, Type, TermDates, SecurityName}}
        for key in kwargs:
            self.SecurityObjs.append(mu.SecurityData(**kwargs[key]))
        
    def FilterOnMaturity(self, MaturityString):
        
        for sec in self.SecurityObjs:
            if isinstance(sec, bu.Bond):
               sec.FilterOnMaturity( MaturityString ) 
               
               
# vanilla Pair trading Security selection of a bond and Stock Index
class DummyVanillaPT:
    
    SecurityObjs = []
    
    def __init__(self, kwargs): #   kwargs = {securityno : {filename, Type, TermDates, SecurityName}}
        
        mu = kwargs['mu']
        sigma = kwargs['sigma']
        kappa = kwargs['kappa']
        theta = kwargs['theta']
        eta = kwargs['eta']
        rho = kwargs['rho']
        
        mean = [0,0]
        cov = [[1, rho],[rho, 1]]
        z,w = np.random.multivariate_normal(mean, cov, 20).T
        
        PriceB = [100]
        PriceA = [100]
        X = [0]
        dt = 1/365
        for i in range(9):
            
            PriceB.append(PriceB[i] +  PriceB[i]*(mu*dt + sigma*(z[i+1] - z[i])*(dt**0.5)))
            X.append(X[i] + (kappa*(theta - X[i])*dt + eta*(w[i+1] - w[i])*(dt**0.5)))
            
            PriceA.append(np.exp(X[i+1] + np.log(PriceB[i+1])))
            
        
        DateArr = [i for i in range(10)]
            
        self.SecurityObjs.append(bu.Index(PriceA, DateArr, SecurityName = 'A'))
        self.SecurityObjs.append(bu.Index(PriceB, DateArr, SecurityName = 'B'))
        
        
        
class VanillaPT:
    
    SecurityObjs = []
    
    def __init__(self, kwargs): #   kwargs = {securityno : {filename, Type, TermDates, SecurityName}}
        for key in kwargs:
            self.SecurityObjs.append(mu.SecurityData(**kwargs[key]))
        
    def FilterOnMaturity(self, MaturityString):
        
        for sec in self.SecurityObjs:
            if isinstance(sec, bu.Bond):
               sec.FilterOnMaturity( MaturityString ) 
               
               
               
