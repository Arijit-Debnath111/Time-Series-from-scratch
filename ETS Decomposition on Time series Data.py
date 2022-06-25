#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


airline=pd.read_csv('AirPassengers.csv',index_col=0,parse_dates=True)
airline.head()


# In[5]:


airline.tail()


# In[6]:


airline[['#Passengers']].plot()


# In[7]:


from statsmodels.tsa.seasonal import seasonal_decompose
result=seasonal_decompose(airline['#Passengers'],model='multiplicative')
result.plot()


# In[8]:


from statsmodels.tsa.stattools import adfuller
print('Augumented Dickey-Fuller Test on Airline Data')
dftest=adfuller(airline['#Passengers'],autolag='AIC')
dftest


# In[10]:


print('Augumented Dickey-Fuller Test on Airline Data')

dfout=pd.Series(dftest[0:4],index=['ADF test statistic','p-value','# lags used','# observations'])
for key,val in dftest[4].items():
    dfout[f'critical value ({key})']=val
print(dfout)


# In[11]:


df1=pd.read_csv('AirPassengers.csv',index_col='Month',parse_dates=True)
df1.index.freq='MS'


# In[12]:


df1.head()


# In[14]:


df1.index


# In[15]:


from statsmodels.tsa.stattools import acf,pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
acf(df1['#Passengers'])


# In[16]:


title='Autocorrelation:Airline Passengers'
lags=40
plot_acf(df1,title=title,lags=lags)


# In[21]:


from statsmodels.tsa.statespace.tools import diff

df1['d1']=diff(df1['#Passengers'],k_diff=1)
df1['d1'].plot(figsize=(12,5))


# In[22]:


title='PACF:Airline Passengers First Difference'
lags=40
plot_pacf(df1['d1'].dropna(),title=title,lags=np.arange(lags));


# In[ ]:




