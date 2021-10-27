import sys
import os
sys.path.append(os.path.abspath(r'../Commons'))

import Configuration as co
import matplotlib.pyplot as plt
import SecuritySelectionUtils as su
import PositionSizingUtils as pu
import pandas as pd

# kwargs = {securityno : {filename, Type, TermDates, SecurityName}}


kwargs = {'1':{'filename': co.SNP500Filename,
               'Type':'Index', 
               'TermDates':'None', 
               'SecurityName':'SNP500'
               }, 

          '2':{'filename': co.Treasuries10Filename,
               'Type':'Bond', 
               'TermDates':'None', 
               'SecurityName':'Treasuries10'
               }
           }

#Security selection
Securities = su.VanillaRPS(kwargs)
#Securities.FilterOnMaturity('CRSP Fixed Term Index - 10-Year (Nominal)')

#PositionSizingData
Positions = pu.VanillaRPS(Securities.SecurityObjs, VolDays = 252, RebalancingDays = 21)
PositionsData = Positions.PositionsSize(InitialCapital = 100)

#Save Benchmarking Data
PositionsData.to_csv(co.Risk_Parity_Strategy + '/' + Positions.__class__.__name__+'.csv')

#Plot data to see results
InitialCapital = 100
ReturnData = pd.DataFrame()
ReturnData['Date'] = PositionsData['Date']
ReturnData['Capital'] = PositionsData['Capital']/InitialCapital

ylabels = ['Capital']

for sec in Securities.SecurityObjs:
    ReturnData['Price_'+str(sec)] = PositionsData['Price_'+str(sec)]/PositionsData.loc[0, 'Price_'+str(sec)]
    ylabels.append('Price_'+str(sec))
   
fig = ReturnData.plot(x = 'Date', y = ylabels, kind = 'line', grid = True).get_figure()
fig.set_size_inches(20,10)

fig.savefig('Return.jpeg')





