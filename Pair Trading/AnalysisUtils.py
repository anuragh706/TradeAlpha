import sys
import os
sys.path.append(os.path.abspath(r'../Commons'))

import Configuration as co
import matplotlib.pyplot as plt
import SecuritySelectionUtils as su
import PositionSizingUtils as pu
import pandas as pd

## kwargs = {securityno : {filename, Type, TermDates, SecurityName}}
#
#
#kwargs = {'mu':0.0647,'sigma':0.0526,'theta': 0,'rho': 0.3,'eta': 0.01,'kappa':0.01}
#
##Security selection
#Securities = su.DummyVanillaPT(kwargs)
#
#Positions = pu.VanillaPT(Securities.SecurityObjs, -1, 0, 10, 1/365, 1)
#
##data = pd.DataFrame(Securities.SecurityObjs[0].PriceLevel, Securities.SecurityObjs[1].PriceLevel)
##data.to_csv('test.csv')
#
#Positions.calibrate(0)
#
##Positions.calibrate(0)




# kwargs = {securityno : {filename, Type, TermDates, SecurityName}}


kwargs1 = {'1':{'filename': co.Nifty50Filename,
               'Type':'Index', 
               'TermDates':'None', 
               'SecurityName':'Nifty50'
               }, 

          '2':{'filename': co.BankNiftyFilename,
               'Type':'Index', 
               'TermDates':'None', 
               'SecurityName':'BankNifty'
               }
           }


kwargs2 = { 
          '2':{'filename': co.HaitongFilename,
               'Type':'Stock', 
               'TermDates':'None', 
               'SecurityName':'Haitong'
               },
          '1':{'filename': co.HuataiFilename,
               'Type':'Stock', 
               'TermDates':'None', 
               'SecurityName':'Huatai'
               }
           }
         
            
kwargs3 = {
            '1':{'filename': co.Nifty012019Filename,
               'Type':'Index', 
               'TermDates':'None', 
               'SecurityName':'Nifty012019'
               }, 

            '2':{'filename': co.Nifty022019Filename,
               'Type':'Index', 
               'TermDates':'None', 
               'SecurityName':'Nifty022019'
               }
           } 
            
            
kwargs4 = { 
          '1':{'filename': co.ICICIFilename,
               'Type':'Stock', 
               'TermDates':'None', 
               'SecurityName':'ICICI'
               },
          '2':{'filename': co.HDFCFilename,
               'Type':'Stock', 
               'TermDates':'None', 
               'SecurityName':'HDFC'
               }
           }
#Security selection
Securities = su.VanillaPT(kwargs4)
#Securities.FilterOnMaturity('CRSP Fixed Term Index - 10-Year (Nominal)')
N = 365
r = 0.05
lamda = -1.5
#PositionSizingData
Positions = pu.VanillaPT(Securities.SecurityObjs, lamda, r, N)
PositionsData = Positions.PositionSize(10)

#Save Benchmarking Data
PositionsData.to_csv(co.Pair_Trading_Strategy + '/' + Positions.__class__.__name__+'.csv')

#Plot data to see results
InitialCapital = 10
ReturnData = pd.DataFrame()
ReturnData['Date'] = PositionsData.loc[N:,'Date']
ReturnData['Capital'] = PositionsData.loc[N:, 'Capital']/InitialCapital

ylabels = ['Capital']

for sec in Securities.SecurityObjs:
    ReturnData['Price_'+str(sec)] = PositionsData.loc[N:, 'Price_'+str(sec)]/PositionsData.loc[N, 'Price_'+str(sec)]
    ylabels.append('Price_'+str(sec))
   
fig = ReturnData.plot(x = 'Date', y = ylabels, kind = 'line', grid = True).get_figure()
fig.set_size_inches(20,10)

fig.savefig('Return.jpeg')
