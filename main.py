from cProfile import label
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA


from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()



df = pd.read_csv('airline_passengers.csv', parse_dates = ['Month'], index_col = ['Month'])
 
df.head()

'''# graphique de base
plt.xlabel('Date')
plt.ylabel('Nombre de passagers aériens')
plt.plot(df)
plt.show()
'''


output = seasonal_decompose(df)
plt.plot(df, label='Serie temporelle')
plt.plot(output.resid, label='Bruit')
plt.plot(output.trend, label='Tendance')
plt.plot(output.seasonal, label='Composante saisonniere')
plt.xlabel('Date')
plt.ylabel('Nombre de passagers aériens')
plt.title("Decomposition d'une serie temporelle")
plt.legend()
plt.show()






