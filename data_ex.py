from matplotlib import pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller

import statsmodels.api as sm

df = pd.read_csv('sisu_forecast.csv', parse_dates=['date'], index_col='date')

df = df.groupby(by=['date'], dropna=False, as_index=False).sum()

series = df.loc[:, 'clicks']

df.plot()
plt.show()

# histogram

df.hist()
plt.show()

# calc mean + variance

x = series.values

print(series.values)

split = round(len(x) / 2)
x1, x2 = x[0:split], x[split:]
mean1 = x1.mean()
mean2 = x2.mean()
var1, var2 = x1.var(), x2.var()
print('mean1=' + str(mean1))
print('mean2=' + str(mean2))
print('variance1= ' + str(var1))
print('variance2= ' + str(var2))

# adfuller time

result = adfuller(x)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# if the p-value is obtained is greater than significance level of 0.05 and the ADF statistic is higher than any of
# the critical values there is no reason to reject the null hypothesis. So, the time series is in fact non-stationary.

print(df.head())

dates = df.index
value = df['clicks']

x_values = dates
y_values = value

plt.plot(x_values, y_values)

plt.show()
