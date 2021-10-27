'''

File paths, module paths, and constants used in the rest of the project go here.
To resolve dependencies it is enough to import this file in code file

'''

import sys
import os

#  main folder path
mainfolder = os.path.abspath('..')


# Data directories
Result_attic = os.path.abspath(r'../DataFiles/Result_attic')
Result = os.path.abspath(r'../DataFiles/Result')
FinancialData = os.path.abspath(r'../DataFiles/FinancialData')
CommonFiles = os.path.abspath(r'../DataFiles/CommonFiles')

# Code directories
Commons = os.path.abspath(__file__)
Execution = os.path.abspath(r'../Execution')
Risk_Parity_Strategy = os.path.abspath(r'../Risk Parity Strategy')
Documentation = os.path.abspath(r'../Documentation')
Pair_Trading_Strategy = os.path.abspath(r'../Pair Trading')

# Adds module paths,makes it easier to import modules from other directories
sys.path.append(Commons)
sys.path.append(Execution)
sys.path.append(Risk_Parity_Strategy)
sys.path.append(Documentation)


# Doc gen paths
DocGenInput = CommonFiles + r'/docdata.csv'
DocGenOutput = Result + r'/Report.docx'



#Mapping of security  data columns and Meta data
SNP500 = {'PriceLevel' : 'spindx', 'Date': 'caldt'}
SNP500Meta = {'Exchange' : 'NYSE', 'Holidays' : 'NY'}
Treasuries10 = {'PriceLevel' : 'PriceLevel', 'Date':'DATE'}
Treasuries10Meta = {'Exchange' : 'NYSE', 'Holidays' : 'NY', 'Type': 'Sovereign', 'Maturity' : '10Y'}

SNP500Filename = FinancialData + r'/IndexData/S&P500.csv'
Treasuries10Filename = FinancialData + r'/BondData/Treasuries10.csv'

Nifty50 = {'PriceLevel' : 'Close', 'Date': 'Date'}
Nifty50Meta = {'Exchange' : 'NSE', 'Holidays' : 'Del'}
BankNifty = {'PriceLevel' : 'Close', 'Date': 'Date'}
BankNiftyMeta = {'Exchange' : 'NSE', 'Holidays' : 'Del'}

Nifty50Filename = FinancialData + r'/IndexData/NIFTY50.csv'
BankNiftyFilename = FinancialData + r'/IndexData/BANKNIFTY.csv'


Huatai = {'PriceLevel' : 'Close', 'Date': 'Date'}
HuataiMeta = {'Exchange' : 'HK', 'Holidays' : 'HK'}
Haitong = {'PriceLevel' : 'Close', 'Date': 'Date'}
HaitongMeta = {'Exchange' : 'HK', 'Holidays' : 'HK'}

HuataiFilename = FinancialData + r'/StockData/Huatai.csv'
HaitongFilename = FinancialData + r'/StockData/Haitong.csv'


Nifty012019 = {'PriceLevel' : 'Close', 'Date': 'Date'}
Nifty012019Meta = {'Exchange' : 'HK', 'Holidays' : 'HK'}
Nifty022019 = {'PriceLevel' : 'Close', 'Date': 'Date'}
Nifty022019Meta = {'Exchange' : 'HK', 'Holidays' : 'HK'}

Nifty012019Filename = FinancialData + r'/IndexData/Nifty012019.csv'
Nifty022019Filename = FinancialData + r'/IndexData/Nifty022019.csv'


HDFC = {'PriceLevel' : 'Close', 'Date': 'Date'}
HDFCMeta = {'Exchange' : 'HK', 'Holidays' : 'HK'}
ICICI = {'PriceLevel' : 'Close', 'Date': 'Date'}
ICICIMeta = {'Exchange' : 'HK', 'Holidays' : 'HK'}

HDFCFilename = FinancialData + r'/StockData/HDFC.csv'
ICICIFilename = FinancialData + r'/StockData/ICICI.csv'