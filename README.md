# lsg-forecasting
Forecasting work for SEO


This uses the ARIMA model; it can be expanded to use the SARIMA model for seasonal data. 

 data_ex.py is expecting a CSV file with at least two columns, including date at column[0]. This python program performs data exploration to suggest a model to the user.
 
 
forecast.py is expecting two csv files with at least two columns each. this python program actually does the prediction. you can change the suggested arima parameters based on what the ADF suggestion is. 


The information you should modify is as follows. 

client_name = 'sams'
training_set = 'first_half.csv'
confirmation_set = 'third_quarter.csv'
metric = 'clicks'
columnname = 'f0_'
columntime = 'date'
days_in_year=365
dateformat = 'days'
