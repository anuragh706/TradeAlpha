import sys
import os
sys.path.append(os.path.abspath(r'../Commons'))
import Configuration as co
import pandas as pd
import numpy as np

class Security:
    
    
    def __init__(self, Exchange, Holidays, SecurityName):
        
        self.SecurityName = SecurityName
        self.Exchange = Exchange
        self.Holidays = Holidays
        
    def __str__(self): return self.SecurityName
    

class Bond(Security):
    
    def __init__(self, PriceLevel = [], Date = [], Maturity='10Y', Coupon=[], Type='Sovereign', Exchange='NYSE', Holidays='NY', SecurityName = 'Treasuries'):
        
        self.PriceLevel = PriceLevel
        self.Date = Date
        self.Maturity = Maturity
        self.Coupon = Coupon
        self.Type = Type
        super(Bond, self).__init__(Exchange, Holidays, SecurityName)   
     
    #Filtering data on Maturity   
    def FilterOnMaturity(self, MaturityString):
        
        if MaturityString not in self.Maturity:
           print('Maturity not found for ', self.SecurityName, ' . Please input correct string')
           return
         
        PriceLevel = []
        Date = []
        Maturity = []
        #Add coupon later on
        
        for index in range(len(self.Maturity)):
            
            if self.Maturity[index] == MaturityString:
                
               PriceLevel.append(self.PriceLevel[index])
               Date.append(self.Date[index])
               Maturity.append(self.Maturity[index])
               
        self.Maturity = Maturity
        self.Date = Date
        self.PriceLevel = PriceLevel
        
        
        
        
class Index(Security):
    
    def __init__(self, PriceLevel = [], Date = [], Exchange='NYSE', Holidays='NY', SecurityName='S&P500'):
        
        self.PriceLevel = PriceLevel
        self.Date = Date
        super(Index, self).__init__(Exchange, Holidays, SecurityName)
    
class Stock(Security):
    
    def __init__(self, PriceLevel = [], Date = [], Exchange='NYSE', Holidays='NY', SecurityName='S&P500'):
        
        self.PriceLevel = PriceLevel
        self.Date = Date
        super(Stock, self).__init__(Exchange, Holidays, SecurityName)
    
    
class Commodity(Security):
    
    def __init__(self):
        return    

    
class Derivatives:
    def __init__(self):
        return
 

# One day returns of Prices    
def DailyReturn(SecurityObj):

    return pd.DataFrame(data=SecurityObj.PriceLevel).pct_change(1)    
    
# rolling standard devition
def VolCal(SecurityObj, NumDays=30):
    
    ReturnData = DailyReturn(SecurityObj)
    data = ReturnData.rolling(NumDays).std().values.tolist()
    data = [d[0] for d in data]
    VolData = [SecurityObj.Date, data]
    
    return VolData