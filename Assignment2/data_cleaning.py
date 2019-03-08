#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats


# In[2]:


DATA_PATH = "data.csv"


# In[3]:


raw_data = pd.read_csv("data.csv", index_col = 0)


# In[4]:


drop_list = ["lat", "lon", "wsnm", "prov"]
useful_data = raw_data.drop(drop_list, axis = 1)


# We find that null values occur in [prcp, gbrd, wdsp, gust]. We use different approaches to clean them up

# For gbrd, as suggested in the assignment document, just imput with 0

# In[5]:


useful_data['gbrd'] = useful_data['gbrd'].fillna(0)


# use the most recent value to fill missing wind speed data and wind gust data

# In[6]:


useful_data["wdsp"] = useful_data["wdsp"].fillna(method = "ffill")
useful_data["gust"] = useful_data["gust"].fillna(method = "ffill")


# In[7]:


print(useful_data.isnull().sum().sum())


# In[8]:


z_score_cols = ["stp", "smax", "smin"]


# In[10]:


tolerance = 0.5
reserve_index = np.full(useful_data.shape[0], True)
print(reserve_index.sum())
for col_name in z_score_cols:
    z_score = stats.zscore(useful_data[col_name])
    reserve_index = reserve_index & (z_score < tolerance) & (z_score > - tolerance)
    print(reserve_index.sum())


# In[11]:


data = useful_data.iloc[reserve_index, :]


# In[12]:


print(data.shape)


# In[13]:


data.to_csv("cleaned_data.csv")


# In[ ]:




