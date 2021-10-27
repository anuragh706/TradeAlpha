import sys
import os
sys.path.append(os.path.abspath(r'../Commons'))
import pandas as pd
import BaseUtils as bu
import numpy as np

class VanillaRPS:
    
    def __init__(self, SecurityObjs, VolDays, RebalancingDays=21):
        
        self.SecurityObjs = SecurityObjs
        self.VolDays = VolDays
        self.RebalancingDays = RebalancingDays
        
    def PositionsSize(self, InitialCapital = 100):
        
        VolData = {str(sec) : bu.VolCal(sec , self.VolDays) for sec in self.SecurityObjs}
        VolDataCombined = pd.DataFrame(data = self.SecurityObjs[0].Date, columns = ['Date'])
        
        for sec in self.SecurityObjs:

            VolDataSec = pd.DataFrame(data = np.array(VolData[str(sec)]).T.tolist(), columns = ['Date'] + ['Vol_' + str(sec)])
            PriceDataSec = pd.DataFrame(data = np.array([sec.Date, sec.PriceLevel]).T.tolist(), columns = ['Date'] + ['Price_' + str(sec)])
            VolDataSec = VolDataSec.merge(PriceDataSec, how='inner', on='Date')
            VolDataCombined = VolDataCombined.merge(VolDataSec, how='inner', on='Date')
        
        VolDataCombined.set_index('Date')
        VolDataCombined = VolDataCombined.dropna().reset_index(drop=True)
        VolDataCombined['InverseSumOfVol'] = 0

        for sec in self.SecurityObjs:
            
            VolDataCombined['InverseSumOfVol'] = VolDataCombined['InverseSumOfVol'] + 1/VolDataCombined['Vol_'+str(sec)]
            
        for sec in self.SecurityObjs:
            
            VolDataCombined['Weight_'+str(sec)] = (1/VolDataCombined['Vol_'+str(sec)])/VolDataCombined['InverseSumOfVol']
            VolDataCombined['Position_'+str(sec)] = (VolDataCombined['Weight_'+str(sec)]*InitialCapital)/VolDataCombined['Price_'+str(sec)]
            
        VolDataCombined['Capital'] = 0
        VolDataCombined.loc[0, 'Capital'] = InitialCapital

        
        for index in range(len(VolDataCombined)):
            
            if (index % self.RebalancingDays) == 0:
                
                if index != 0:
                    
                    VolDataCombined.loc[index, 'Capital'] = 0
                    for sec in self.SecurityObjs:
                        VolDataCombined.loc[index, 'Capital'] = VolDataCombined.loc[index, 'Capital'] + VolDataCombined.loc[index, 'Price_'+str(sec)]*VolDataCombined.loc[index-self.RebalancingDays, 'Position_'+str(sec)]
                    
                for sec in self.SecurityObjs:
                    VolDataCombined.loc[index, 'Position_'+str(sec)] = VolDataCombined.loc[index, 'Weight_'+str(sec)]*VolDataCombined.loc[index, 'Capital']/VolDataCombined.loc[index, 'Price_'+str(sec)]
                    
            else:
                
                for sec in self.SecurityObjs:
                    VolDataCombined.loc[index, 'Position_'+str(sec)] = VolDataCombined.loc[index-1, 'Position_'+str(sec)]
                    VolDataCombined.loc[index, 'Weight_'+str(sec)] = VolDataCombined.loc[index-1, 'Weight_'+str(sec)]
                
                VolDataCombined.loc[index, 'Capital'] = 0
                for sec in self.SecurityObjs:
                    VolDataCombined.loc[index, 'Capital'] = VolDataCombined.loc[index, 'Capital'] + VolDataCombined.loc[index, 'Position_'+str(sec)]*VolDataCombined.loc[index, 'Price_'+str(sec)]
                  
        return VolDataCombined   


class VanillaPT:
    
    
    def __init__(self, SecurityObjs, lamda, rate, TimePeriod, deltaT = 1/365, continuous = 1):
        
        self.SecurityObjs = SecurityObjs
        self.lamda = lamda
        self.r = rate
        self.N = TimePeriod
        self.dt = deltaT
        self.PriceData = pd.DataFrame(data = self.SecurityObjs[0].Date, columns = ['Date'])
        self.continuous = continuous
        
        for sec in self.SecurityObjs:
            
            PriceData = pd.DataFrame(data = np.array([sec.Date, sec.PriceLevel]).T.tolist(), columns = ['Date'] + ['Price_' + str(sec)])
            self.PriceData = self.PriceData.merge(PriceData, how='inner', on='Date')
         
        # At index 1 is A at 2 is B  
        self.PriceData['X'] = np.log(self.PriceData.iloc[ : , 1]) - np.log(self.PriceData.iloc[ : , 2])
        self.PriceData['S'] = np.log(self.PriceData.iloc[ : , 2])
        self.T = len(self.PriceData)
        
    def calibrate(self, StartIndex):
        
        PriceData = self.PriceData.iloc[StartIndex : StartIndex + self.N].copy(deep=True).reset_index()

        p_hat = ( self.N * np.sum(PriceData.loc[1:,'X'].values*PriceData.loc[:self.N-2,'X'].values)
                + (PriceData.loc[0, 'X'] - PriceData.loc[self.N-1, 'X'])*np.sum(PriceData.loc[:self.N-2, 'X']) 
                - np.sum(PriceData.loc[ : self.N-2, 'X'])**2)/(self.N * np.sum(PriceData.loc[: self.N - 2, 'X']**2) - np.sum(PriceData.loc[ : self.N-2, 'X'])**2)
        
        q_hat = (np.sum(PriceData.loc[1: , 'X'])
                - p_hat*np.sum(PriceData.loc[: self.N - 2, 'X']))/self.N
                        
        m_hat = ( PriceData.loc[self.N - 1 , 'S']
                - PriceData.loc[0, 'S'] )/self.N
        
        s_hat_square =  (np.sum((PriceData.loc[1: , 'S'].values - PriceData.loc[: self.N-2, 'S'].values)**2) 
                        - 2*m_hat*(PriceData.loc[self.N - 1, 'S']
                        - PriceData.loc[0, 'S'])
                        + self.N*(m_hat**2))/self.N
                        
        v_hat_square = (PriceData.loc[self.N - 1, 'X']**2
                        - PriceData.loc[0, 'X']**2
                        + (1 + p_hat)*np.sum(PriceData.loc[: self.N - 2, 'X']**2) - p_hat*np.sum(PriceData.loc[1:,'X'].values*PriceData.loc[:self.N-2,'X'].values)
                        - self.N*q_hat)/self.N
        
        
        
        c_hat = (np.sum(PriceData.loc[ 1:, 'X'].values*( PriceData.loc[1 : , 'S'].values - PriceData.loc[ : self.N-2, 'S'].values))
                + p_hat*np.sum(PriceData.loc[ : self.N-2, 'S'].values - PriceData.loc[1 : , 'S'].values)
                - m_hat*(PriceData.loc[self.N - 1, 'X'] - PriceData.loc[0, 'X'])
                - m_hat*(1-p_hat)*np.sum(PriceData.loc[ : self.N - 2, 'X']))/(self.N*((v_hat_square*s_hat_square)**0.5))  
        

        sigma = (s_hat_square/self.dt)**0.5
        mu = m_hat/self.dt + 0.5*(sigma**2)
        kappa = -np.log(p_hat)/self.dt
        theta = q_hat/(1 - p_hat)
        eta = (2*v_hat_square*kappa/(1 - p_hat**2))**0.5
        rho = kappa*c_hat*((s_hat_square*v_hat_square)**0.5)/(eta*sigma*(1 - p_hat))
        
        Q = 0.5/((1 - rho**2)*(eta**2)*self.lamda)
        A = mu - self.r + rho*sigma*eta + (eta**2)/2
        D2 = -A + mu - self.r
        D3 = (kappa*A*sigma - kappa*(sigma + rho*eta)*(mu - self.r))/(self.lamda*(eta**2)*sigma*(1 - rho**2))
        D4 = (0.5*( (A*sigma - (sigma + rho*eta)*(mu - self.r))**2 + ((mu - self.r)**2)*(1 - rho**2)*(eta**2)))/(self.lamda*(eta**2)*(sigma**2)*(1 - rho**2))
        
        return sigma, mu, kappa, theta, eta, rho, Q, A, D2, D3, D4
    
        
    def getWeightsForIndex(self, index, capital = 100):
        
        
        if self.continuous == 1:
            
            sigma, mu, kappa, theta, eta, rho, Q, A, D2, D3, D4 = self.calibrate(index - self.N + 1)
        
        else:
            sigma, mu, kappa, theta, eta, rho, Q, A, D2, D3, D4 = self.calibrate(0)
            
        
        L = Q*(kappa**2)*(index - self.T)*self.dt
        M = D2*Q*(kappa**2)*(((index - self.T)*self.dt)**2) + D3*(index - self.T)*self.dt
        N = (D2**2)*Q*(kappa**2)*(((index - self.T)*self.dt)**3)/3 + (D2*D3 - (eta**2)*Q*(kappa**2))*(((index - self.T)*self.dt)**2)/2 - D4*(index - self.T)*self.dt
        
        t11 = 1/(self.lamda*(1 - (rho**2))*(eta**2))
        t12 = -(rho*eta + sigma)/(self.lamda*(1 - (rho**2))*(eta**2)*sigma)
        t21 = t12
        t22 = ((eta**2) + (sigma**2) + 2*rho*eta*sigma)/(self.lamda*(1 - (rho**2))*(eta**2)*(sigma**2))
        
        fx = - (2*L*(theta - self.PriceData.loc[index , 'X']) + M)
        f1 = kappa*(theta - self.PriceData.loc[index , 'X']) + mu + 0.5*(eta**2) + rho*sigma*eta - self.r +  2*self.lamda*((eta**2) + rho*eta*sigma)*fx
        f2 = mu - self.r + 2*self.lamda*rho*sigma*eta*fx
        
        tMat = np.array([[t11,t12],[t21,t22]])
        fMat = np.array([[f1],[f2]])
        
        pi = -0.5*np.matmul(tMat, fMat)
        
        weights = pi/(capital*np.exp(self.r*(self.T - index)*self.dt))
        weights = weights.T.tolist()[0]
        weights = [weights[1], weights[0]]
        weights.append(1-weights[0]-weights[1])

        return weights
        
    def getWeightForIndex(self, index, capital = 100):
        
        
        if self.continuous == 1:
            
            sigma, mu, kappa, theta, eta, rho, Q, A, D2, D3, D4 = self.calibrate(index - self.N + 1)
        
        else:
            sigma, mu, kappa, theta, eta, rho, Q, A, D2, D3, D4 = self.calibrate(0)
            
        
        L = Q*(kappa**2)*(index - self.T)*self.dt
        M = D2*Q*(kappa**2)*(((index - self.T)*self.dt)**2) + D3*(index - self.T)*self.dt
                
        fx = - (2*L*(theta - self.PriceData.loc[index , 'X']) + M)
        pi = - ((kappa*(theta - self.PriceData.loc[index , 'X']) + 0.5*(eta**2) + rho*sigma*eta)/(2*self.lamda*eta*eta)) + fx
        
        weights = pi/(capital*np.exp(self.r*(self.T - index)*self.dt))
        weights = [weights, -weights, 1]
 
        return weights        
#    
#    def PositionSize(self, InitialCapital = 100):
#        
#        PriceData = self.PriceData.copy(deep = True)
#        PriceData['Capital'] = InitialCapital
#        PriceData['RiskFree'] = 10
#        PriceData['Weight1'] = 0
#        PriceData['Weight2'] = 0
#        PriceData['Weight3'] = 0
#        PriceData['Position1'] = 0
#        PriceData['Position2'] = 0
#        PriceData['Position3'] = 0
#                
#        r = self.r
#        N = self.N
#        
#        for index, row in PriceData.iterrows():
#            
#            
#            if index < N-1:
#                
#               continue
#
#            PriceData.loc[index, 'RiskFree'] = PriceData.loc[index-1, 'RiskFree']*(np.exp(r*self.dt))
#            PriceData.loc[index, 'Capital'] = PriceData.loc[index-1, 'Position3']*PriceData.loc[index, 'RiskFree'] + PriceData.loc[index-1, 'Position1']*PriceData.iloc[index, 1] + PriceData.loc[index-1, 'Position2']*PriceData.iloc[index, 2]
#            
#            
#            if PriceData.loc[index, 'Capital'] == 0:
#                
#               capital = InitialCapital
#               
#            else:   
#                
#                capital = PriceData.loc[index , 'Capital']
#            
#            weights = self.getWeightForIndex(index, capital)
#            capitalAloc = [w*capital for w in weights]
#            
#            PriceData.loc[index, 'Weight1'] = weights[0]
#            PriceData.loc[index, 'Weight2'] = weights[1]
#            PriceData.loc[index, 'Weight3'] = weights[2]
#            
#            PriceData.loc[index, 'Position1'] = capitalAloc[0]/PriceData.iloc[index, 1]
#            PriceData.loc[index, 'Position2'] = capitalAloc[1]/PriceData.iloc[index, 2]
#            PriceData.loc[index, 'Position3'] = capitalAloc[2]/PriceData.loc[index, 'RiskFree']
#            
#            
#        return PriceData
            
                        
            
    def PositionSize(self, InitialCapital = 100):
        
        PriceData = self.PriceData.copy(deep = True)
        PriceData['Capital'] = InitialCapital
        PriceData['RiskFree'] = 10
        PriceData['Weight1'] = 0
        PriceData['Weight2'] = 0
        PriceData['Weight3'] = 0
        PriceData['Position1'] = 0
        PriceData['Position2'] = 0
        PriceData['Position3'] = 0       
        r = self.r
        N = self.N
        
        for index, row in PriceData.iterrows():
            
            
            if index < N-1:
                
               continue

            PriceData.loc[index, 'RiskFree'] = PriceData.loc[index-1, 'RiskFree']*(np.exp(r*self.dt))
            PriceData.loc[index, 'Capital'] = PriceData.loc[index-1, 'Position3']*PriceData.loc[index, 'RiskFree'] + PriceData.loc[index-1, 'Position1']*PriceData.iloc[index, 1] + PriceData.loc[index-1, 'Position2']*PriceData.iloc[index, 2]
            
            
            weights = self.getWeightForIndex(index, PriceData.loc[index, 'RiskFree'])
            capitalAloc = [w*InitialCapital for w in weights]
            
            PriceData.loc[index, 'Weight1'] = weights[0]
            PriceData.loc[index, 'Weight2'] = weights[1]
            PriceData.loc[index, 'Weight3'] = weights[2]
            
            PriceData.loc[index, 'Position1'] = capitalAloc[0]/PriceData.iloc[index, 1] + PriceData.loc[index-1, 'Position1']
            PriceData.loc[index, 'Position2'] = capitalAloc[1]/PriceData.iloc[index, 2] + PriceData.loc[index-1, 'Position2']
            PriceData.loc[index, 'Position3'] = capitalAloc[2]/PriceData.loc[index, 'RiskFree']
            
            
        return PriceData        
        
        
        
        
        
        

