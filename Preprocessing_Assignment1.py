
# coding: utf-8

# In[6]:


get_ipython().system('pip install --upgrade pip')


# In[4]:


get_ipython().system('pip install imblearn')


# In[5]:


get_ipython().system('pip install scikit-learn')


# In[8]:


import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib as plot
from imblearn.over_sampling import SMOTE


# In[46]:


churn_data = pd.read_csv("/Users/nissani/Desktop/WA_Fn_UseC_Telco_Customer_Churn.csv")


# In[47]:


churn_data.head()


# In[99]:


list(churn_data)


# In[100]:


column_headers = np.array(list(churn_data))[1:-1]


# In[101]:


column_headers


# In[48]:


workable_data = churn_data.values


# In[49]:


workable_data


# In[50]:


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


# In[51]:


workable_data


# In[52]:


churn_data_new = pd.DataFrame(workable_data)


# In[53]:


churn_data['Churn'].value_counts().plot(kind = 'bar')


# In[64]:


labels = workable_data[:, -1]


# In[74]:


labels = list(labels)


# In[75]:


labels


# In[66]:


customer_id = workable_data[:, 0]


# In[67]:


customer_id


# In[68]:


workable_data = workable_data[:, 1:-1]


# In[69]:


workable_data


# In[70]:


for i in range(len(workable_data)):
    if workable_data[i, -1] == ' ':
        workable_data[i,-1] = 0
    else:
        workable_data[i] = workable_data[i].astype(np.float)


# In[71]:


workable_data


# In[72]:


sm = SMOTE(random_state = 28)


# In[86]:


new_data, new_labels = sm.fit_resample(workable_data, labels)


# In[80]:


len(new_data)


# In[79]:


from collections import Counter
Counter(new_labels)


# In[87]:


new_data -= np.mean(new_data)


# In[89]:


new_data = new_data/(np.std(new_data))**2


# In[90]:


new_data


# In[105]:


printable_data = pd.DataFrame(new_data, columns = list(column_headers))


# In[106]:


printable_data


# In[107]:


printable_data.to_csv("/Users/nissani/Desktop/Preprocessed.csv")

