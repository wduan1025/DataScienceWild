
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib as plot


# In[45]:


churn_data = pd.read_csv("/Users/nissani/Desktop/WA_Fn_UseC_Telco_Customer_Churn.csv")


# In[46]:


churn_data.head()


# In[47]:


workable_data = churn_data.values


# In[48]:


workable_data


# In[49]:


for i in range(len(workable_data)):
    for j in range(len(workable_data.T)):
        if workable_data[i][j] == 'Male':
            workable_data[i][j] = 0
        if workable_data[i][j] == 'Female':
            workable_data[i][j] = 1
        if workable_data[i][j] == 'No':
            workable_data[i][j] = 0
        if workable_data[i][j] == 'Yes':
            workable_data[i][j] = 1
        if workable_data[i][j] == 'No phone service':
            workable_data[i][j] = 2
        if workable_data[i][j] == 'DSL':
            workable_data[i][j] = 1
        if workable_data[i][j] == 'Fiber optic':
            workable_data[i][j] = 2
        if workable_data[i][j] == 'No internet service':
            workable_data[i][j] = 2
        if workable_data[i][j] == 'Month-to-month':
            workable_data[i][j] = 0
        if workable_data[i][j] == 'One year':
            workable_data[i][j] = 1
        if workable_data[i][j] == 'Two year':
            workable_data[i][j] = 2
        if workable_data[i][j] == 'Electronic check':
            workable_data[i][j] = 0
        if workable_data[i][j] == 'Mailed check':
            workable_data[i][j] = 1
        if workable_data[i][j] == 'Bank transfer (automatic)':
            workable_data[i][j] = 2
        if workable_data[i][j] == 'Credit card (automatic)':
            workable_data[i][j] = 3


# In[50]:


workable_data


# In[51]:


churn_data_new = pd.DataFrame(workable_data)


# In[53]:


churn_data.hist()


# In[56]:


churn_data['Churn'].value_counts().plot(kind = 'bar')

