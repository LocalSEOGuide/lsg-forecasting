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
from datetime import datetime
from dateutil.relativedelta import relativedelta
import csv 
import pmdarima as pm
from datetime import date, timedelta

# set up variables
#client name
client_name = 'client'
#training set: should have dates. should be in order
training_set = 'first_half.csv'
#set used for comparing to the model. 
confirmation_set = 'third_quarter.csv'
# name of metric that will be predicted
metric = 'clicks'
# column of metric that will be predicted
columnname = 'f0_'
#name of column that includes date/time information (note: index should be zero. if index for date/time is not zero, you may need to make fixes below)
columntime = 'date'
days_in_year=365
#days, months or years
dateformat = 'days'


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

#read the training set into series data with dates as index
series = pd.read_csv(training_set, index_col=0, usecols=[columntime, columnname])
print(series.head())
#convert the
series.index = pd.to_datetime(series.index)

print(series.head(10))
# line plot of dataset
series.plot()
plt.ylabel(metric)
plt.show()

#plot out the second half/validation dataset using the same set up 
validation = pd.read_csv(confirmation_set, index_col=0, usecols=[columntime, columnname])
print(validation.head())

validation.index = pd.to_datetime(validation.index)

print(validation.head())

# show the difference between the validation and the dataset

print('Dataset %s, Validation %d' % (len(series), len(validation)))
#show full graph for validation set

validation.plot()

plt.ylabel(metric)
plt.show()



#create length of training data + set up variables

og = series.values
size = int(len(og) * 0.66)
train, test = og[0:size], og[size:len(og)]
history = [x for x in train]
predictions = list()

# fit model
model = sm.tsa.arima.ARIMA(history, order=(4,1,2))
model_fit = model.fit()
#print fit summary.
print(model_fit.summary())

#forecast one step ahead, print to console
forecast = model_fit.forecast()[0]
print(forecast)
#forecast = inverse_difference(og, forecast, days_in_year)
#print('Forecast: %f' % forecast)

#ensure index is din datetime format
validation.index = pd.DatetimeIndex(validation.index).to_period(dateformat[0])
#this was a pain in the ass but: set up index for validation

start_index = 1

print(start_index)
#end prediction for length of training set
end_index = len(series)-1

#forecast from the start index to just before the end of the length of the series
forecast = model_fit.predict(start=start_index, end=end_index)

#for x in the length of the training set
history = [x for x in og]
#list so we can visualize later
forecastResult = []
numberResult = []
#days to count off
day = 0
#datetime stuff
last_date = series.index[-1].date()
print(last_date)
#for x in forecast print out prediction and actual result
#this can definitely be cleaned up -_-
for yhat in forecast:
	#if is there for working with smaller datasets
    if day < 150:
	#inverted = inverse_difference(history, yhat, days_in_year)
        day_number = timedelta(days=day)
        dtd = last_date + day_number
	#datetime is hard
        a = int(dtd.strftime('%Y%m%d'))
        print('Day %g: %f' % (a, yhat))
        history.append(yhat)
        print('Day %d:'%(day))
        print("actual")
        print(validation.index[day])
        print(validation[columnname][day])
        print('prediction')
        print('Day %d: %f' % (day, yhat))
        history.append(yhat)
        forecastResult.append(yhat)
        numberResult.append(validation.index[day])
        day += 1

#forecast = model_fit.predict(start=start_index, end=end_index)

# one-step out-of sample forecast
print("#forecast result")
print(forecastResult)
# create dataframe for forecast result + date
d = {dateformat:numberResult, 'Forecast':forecastResult}
histdf = pd.DataFrame(d)
histdf['Forecast'] = pd.to_numeric(histdf['Forecast'])
print(histdf)
#create dataframe for actual result + date
print("e")
e = {dateformat:validation.index, 'Validation':validation[columnname]}
validSeriesdf = pd.DataFrame(e)
print(validSeriesdf)
validSeriesdf['Validation'] = pd.to_numeric(validSeriesdf['Validation'])

print(validSeriesdf)
print(histdf)

#plot both on the same axis
ax = histdf.plot(alpha=0.6, x=dateformat, y="Forecast", label="forecast", color="r")
validSeriesdf.plot(alpha=0.6, x= dateformat, y='Validation', ax = ax)
plt.ylabel(metric)
plt.show()

#Auto arima. we're using the demo that was in machinelearningmastery. might move this to the beginning or as its own thing

model = pm.auto_arima(series[columnname], start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())

model.plot_diagnostics(figsize=(7,5))
plt.show()
