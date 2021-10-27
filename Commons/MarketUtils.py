import sys
import os
sys.path.append(os.path.abspath(r'../Commons'))
import Configuration as co
import BaseUtils as bu
import pandas as pd


# Dates in datetime fromat {MM/DD/YYYY}
def readData (filename, TermDates, SecurityDict): 
    
    # TermDate should be a list of [startdat, enddate] 
    #other than that it takes the whole data
        
    datecol = SecurityDict['Date']
    raw_data = pd.read_csv (filename, index_col = None).dropna()
    raw_data[datecol] = pd.to_datetime(raw_data[datecol], format='%Y-%m-%d')
    if isinstance(TermDates,list):   
        
        mask = (raw_data[datecol] > TermDates[0]) & (raw_data[datecol] < TermDates[1])
        return raw_data.loc[mask] 
    
    return raw_data


def SecurityData (filename, Type, TermDates, SecurityName):

    colsDict = getattr(co , SecurityName)
    colsDictMeta = getattr(co , SecurityName + 'Meta')
    raw_data = readData (filename, TermDates, colsDict)
    kwargs = {key : raw_data[colsDict[key]].tolist() for key in colsDict}
    kwargs['PriceLevel'] = [float(val) for val in kwargs['PriceLevel']]
    kwargs = {**kwargs, **colsDictMeta}
    kwargs['SecurityName'] = SecurityName
    security = getattr(bu, Type)(**kwargs)
    
    return security




def FundamentalData (filename, statDate, endDate, cols):
    return
