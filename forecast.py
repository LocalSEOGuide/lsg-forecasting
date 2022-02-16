# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller

#v1 on jarvis v2 
  
# Read the dataset
filename = 'sisu_forecast.csv'
columnname = 'clicks'
futurism = '3 months' #current rate 
forecast_time = '12 months' #project 

df = pd.read_csv(filename, parse_dates=['date'], index_col=0, usecols=["date", columnname])



#series = df.groupby(by=['date'], dropna=False, as_index=False).sum()


#df.set_index('date',inplace=True)

series = pd.read_csv('sisu_forecast.csv', index_col=0, usecols=["date", columnname])

series = series.groupby(by=['date'], dropna=False, as_index=False).sum()

autocorrelation_plot(series[columnname])

print(df.head(20))

# display first few rows
print(series.head(20))
# line plot of dataset
series.plot()
plt.show()

#date report data - :we only work with date data - roll up to week or month? 
#adfuller time 

test_result=adfuller(series[columnname])

def adfuller_test(cols):
    result=adfuller(cols)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )

if test_result[1] <= 0.05:
    print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data is stationary")
else:
    print("weak evidence against null hypothesis,indicating it is non-stationary ")

adfuller_test(series[columnname])

print(df.head())



#split the dataset

split_point = len(series) - 7
dataset, validation = series[0:split_point], series[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv', index=False)
validation.to_csv('validation.csv', index=False)
  
# Print the first five rows of the dataset
print(series.head())


# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)
    
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
    
    
x = series.values
days_in_year = 365
differenced = difference(x, days_in_year)
# fit model
model = sm.tsa.arima.ARIMA(differenced, order=(7,0,1))
model_fit = model.fit()
# print summary of fit model
print(model_fit.summary())


df.plot()
plt.show()


forecast = model_fit.forecast()[0]

#forecast.plot()

#p=1, d=1, q=0 or 1

#model=sm.tsa.arima.ARIMA(df[0],order=(1,1,1))
#model_fit=model.fit()
#model_fit.summary()


# ETS Decomposition
#result = seasonal_decompose(df[columnname])
  
# ETS plot 
#result.plot()
