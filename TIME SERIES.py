#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pandas import Grouper


# In[2]:


import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 15, 6
import seaborn as sns
sns.set_style("whitegrid")


# In[3]:


from scipy import stats
from math import sqrt
from sklearn.metrics import mean_squared_error


# In[4]:


from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[5]:


from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


# In[6]:


import warnings
warnings.filterwarnings('ignore')


# In[7]:


df = pd.read_csv('Electric_Production.csv', header=0, index_col=0, parse_dates=True)
split_point = len(df) - 12*3
train, test = df[0:split_point], df[split_point:]
print('train %d, test %d' % (len(train), len(test)))


# In[60]:


train.head()


# In[9]:


train.shape


# In[10]:


train.describe()


# In[11]:


train.info()


# In[12]:


train.isnull().sum()


# In[40]:


fig,axes =plt.subplots(1,2)
# time series line plot on the
axes[0].plot(train['VALUES'])
axes[0].set(xlabel='Date',ylabel='Electric Values')

# histogram plot
sns.histplot(train['VALUES'], ax=axes[1], stat='density', kde=True)
axes[1].set(title='histogram plus estimated density', xlabel='Electric Values')

# plot corresponding normal curve
mu, std = stats.norm.fit(train['VALUES'])
xmin, xmax=plt.xlim() # the maximum x values from the histogram above
x=np.linspace(xmin, xmax, 100)
p=stats.norm.pdf(x, mu, std) #calculate the y values for the normal curve
axes[1].plot(x, p, color='orange',label='norm pdf')
axes[1].legend(loc='best')
plt.show()


# In[16]:


# Create the stack of 5 line plots from 2000 to 2004 to see the seasonality and contrast the data for each year.
groups = train['VALUES']['2001':'2003'].groupby(Grouper(freq='A')) # calendar year end
i = 1
n = len(groups)
rcParams['figure.figsize'] = 15, 6
for name, group in groups:
    plt.subplot((n*100) + 10 + i)
    i += 1
    plt.plot(group)
plt.show()


# In[17]:


# Creates multiple box and whisker plots
groups = train['VALUES']['1985':'2010'].groupby(Grouper(freq='A'))
years = pd.DataFrame()
for name, group in groups:
    years[name.year] = group.values
    plt.figure(figsize=(18,6))
    years.boxplot()
plt.show()


# In[32]:


transformed, lam = stats.boxcox(train['VALUES'].values)
print('Lambda: %f' % lam)


# In[18]:


decomposed1 = seasonal_decompose(train['VALUES'], model='multiplicative')
decomposed1.plot()
plt.show()


# In[19]:


decomposed = seasonal_decompose(train['VALUES'], model='additive')
decomposed.plot()
plt.show()


# In[20]:


def adfuller_test(data):
    result = adfuller(data)
    labels = ['ADF Statistic', 'p-value', '#lags uesd', 'Number of Observations Used']
    for value, labels in zip(result, labels):
        print(labels+':'+str(value))
    if result[1]<=0.05:
        print('Strong evidence against the null hypothesis, reject the null hypothesis. Data is stationary')
    else:
        print('Weak evidence against null hypothesis, time series has a unit root. It is non stationarity')


# In[21]:


adfuller_test(train['VALUES'])


# In[22]:


train['Seasonal First Difference']=train['VALUES']-train['VALUES'].shift(12)
train.head(13)


# In[23]:


adfuller_test(train['Seasonal First Difference'].dropna())


# In[24]:


plt.figure()
plt.subplot(121)
plot_acf(train['Seasonal First Difference'].diff().dropna(), lags=36, ax=plt.gca())
plt.title('Seasonal First Difference', fontsize=12)


# In[25]:


plt.subplot(122)
plot_pacf(train['Seasonal First Difference'].diff().dropna(), lags=36, ax=plt.gca())
plt.show()


# In[26]:


history = [x for x in train['VALUES']]
predictions = []
for i in range(len(test)):
    # predict in the persistence model generated from the previous time
    yhat = history[-1]
    predictions.append(yhat)
    # observations added into the training set
    obs = test['VALUES'][i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(test['VALUES'], predictions))
print('RMSE: %.3f' % rmse)


# # Evaluate a manually configured SARIMA model

# In[27]:


training_window=len(train['VALUES'])
history = [x for x in train['VALUES']]
history = history[(-training_window):] # fix training set window


# In[28]:


predictions = []


# In[29]:


order = (2,0,2)
seasonal_order =(0,1,1,12)


# In[30]:


for i in range(len(test)):
    model = SARIMAX(history, order=order, seasonal_order=seasonal_order)
    # predict
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    # move the training window
    obs = test['VALUES'][i]
    history.append(obs)
    history.pop(0)
    print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(test['VALUES'], predictions))
print('RMSE: %.3f' % rmse)


# # Diagnose Residuals
# 

# In[32]:


residuals = [test['VALUES'][i]-predictions[i] for i in range(len(test['VALUES']))]
results_manual = pd.DataFrame({'Expected':test['VALUES'], 'Prediction':predictions,'Residuals':residuals}, index=test.index)
results_manual.head()


# In[33]:


print(results_manual['Residuals'].describe())


# In[37]:


fig,axes =plt.subplots(2,2)
# autocorrelation plot
pd.plotting.autocorrelation_plot(residuals, ax=axes[0,0])
axes[0,0].set(title='Residual Autocorrelation Plot')

# line plot
axes[0,1].plot(residuals)
axes[0,1].set(title='Residual Line Plot', ylabel='residual')

# Q-Q Plot
import statsmodels.api as sm
sm.qqplot(np.array(residuals),stats.t, fit=True, line="45", ax=axes[1,0])
axes[1,0].set(title='Residual Q-Q Plot', xlabel='residual')

# histogram plot with corresponding normal curve
sns.histplot(residuals, ax=axes[1,1], stat='density', kde=True)
axes[1,1].set(title='Histogram with normal curve', xlabel='residual')
# plot corresponding normal curve
xmin, xmax=plt.xlim() # the maximum x values from the histogram above
x=np.linspace(xmin, xmax, 100)
mu, std = stats.norm.fit(residuals)
p=stats.norm.pdf(x, mu, std) #calculate the y values for the normal curve
axes[1,1].plot(x, p, color='orange',label='norm curve')
axes[1,1].legend(loc='best')
plt.show()


# In[39]:


get_ipython().system('pip install pmdarima')
from pmdarima.arima import auto_arima
model_sarima= auto_arima(train['VALUES'],trace=True, error_action='ignore',
start_p=0,start_q=0,max_p=5,max_q=2,max_d=1,max_D=1,m=12,
suppress_warnings=True,stepwise=True,seasonal=True)
model_sarima_fit = model_sarima.fit(train['VALUES'])
print(model_sarima_fit.summary())


# In[43]:


plt.figure(figsize=(10,10))
#PLOT residual errors
residuals = pd.DataFrame(model_sarima_fit.resid())
model_sarima_fit.plot_diagnostics()
plt.show()


# # CALCULATE RMSE on the dataset

# In[48]:


prediction_auto=model_sarima_fit.predict(len(test['VALUES']))
#ERRORS
residuals_auto = [test['VALUES'][i]-prediction_auto[i] for i in range(len(test['VALUES']))]
results_auto = pd.DataFrame({'Expected':test['VALUES'],'Prediction':prediction_auto,'Residuals':residuals_auto},index=test.index)
results_auto.head()


# In[49]:


RMSE_auto_optimal = sqrt(np.mean(np.square(residuals)))
print('The RMSE of the best SARIMA model =%.3f' % (RMSE_auto_optimal))


# # Compare RMSE
# 

# In[53]:


Compare_RMSE = pd.DataFrame({'RMSE':[10.018, 3.689,4.486]},index=['persistence model','Manually configured model SARIMA(2,0,2)(0,1,1,12)','Automated selected model ARIM'])
Compare_RMSE


# # Model forecast

# In[55]:


def forecast(model,predict_steps):
    pred_uc = model.get_forecast(steps=predict_steps, dynamic=True)
    #print(pred_uc.__dict__)
    #print(dir(pred_uc))
    #SARIMAXResults.conf_int,the default alpha = .05 returns a 95% confidence interval.
    pred_mean = pred_uc.predicted_mean
    pred_ci = pred_uc.conf_int()
    index = pd.date_range('2018-02-01', '2020-1-01', freq='MS')
    prediction_table = pd.DataFrame({'Predicted_Mean':pred_mean,'Lower Bound':pred_ci[:,0],'Upper Bound':pred_ci[:,1]},index=index)
    return prediction_table


prediction_table = forecast(model_fit,24)
prediction_table.head()


# In[58]:


plt.plot(df['VALUES'].loc['1995':], color = "blue", label='observed')
results_manual['Prediction'].plot(color = "red", label='testing_predicted')
prediction_table['Predicted_Mean'].plot(color = "green", label='forecast with CI')
plt.fill_between(prediction_table.index, prediction_table['Lower Bound'],prediction_table['Upper Bound'], color='k', alpha=.25)
plt.xlabel('Date')
plt.ylabel('Electric Production Values')
plt.legend(loc='upper left', fontsize=12)
sns.set()
plt.show()


# In[ ]:




