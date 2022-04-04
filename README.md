# Forecasting work for SEO

This uses the ARIMA model; it can be expanded to use the SARIMA model for seasonal data. 

 data_ex.py is expecting a CSV file with at least two columns, including date at `column[0]`. This python program performs 
 data exploration to suggest a model to the user.
 
`forecast.py` is expecting two csv files with at least two columns each. this python program actually does the prediction. 
You can change the suggested ARIMA parameters based on what the ADF suggestion is. 


The information you should modify is as follows. 
```
client_name = 'client'
training_set = 'first_half.csv'
confirmation_set = 'second_half.csv'
metric = 'clicks'
columnname = 'clicks'
columntime = 'date'
days_in_year=365
dateformat = 'days'
```
