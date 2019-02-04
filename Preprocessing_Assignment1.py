
# coding: utf-8

# In[1]:


get_ipython().system('pip install --upgrade pip')


# In[2]:


get_ipython().system('pip install imblearn')


# In[3]:


get_ipython().system('pip install scikit-learn')


# In[4]:


import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib as plot
from imblearn.over_sampling import SMOTE


# In[5]:


churn_data = pd.read_csv("/Users/nissani/Desktop/WA_Fn_UseC_Telco_Customer_Churn.csv")


# In[6]:


churn_data.head()


# In[7]:


list(churn_data)


# In[8]:


column_headers = np.array(list(churn_data))[1:-1]


# In[9]:


column_headers


# In[10]:


workable_data = churn_data.values


# In[11]:


workable_data


# In[12]:


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


# In[13]:


workable_data


# In[14]:


churn_data_new = pd.DataFrame(workable_data)


# In[15]:


churn_data['Churn'].value_counts().plot(kind = 'bar')


# In[16]:


labels = workable_data[:, -1]


# In[17]:


labels = list(labels)


# In[18]:


labels


# In[19]:


customer_id = workable_data[:, 0]


# In[20]:


customer_id


# In[21]:


workable_data = workable_data[:, 1:-1]


# In[22]:


workable_data


# In[23]:


for i in range(len(workable_data)):
    if workable_data[i, -1] == ' ':
        workable_data[i,-1] = 0
    else:
        workable_data[i] = workable_data[i].astype(np.float)


# In[24]:


workable_data


# In[25]:


sm = SMOTE(random_state = 28)


# In[26]:


new_data, new_labels = sm.fit_resample(workable_data, labels)


# In[27]:


len(new_data)


# In[28]:


from collections import Counter
Counter(new_labels)


# In[29]:


new_data -= np.mean(new_data)


# In[30]:


new_data = new_data/(np.std(new_data))**2


# In[31]:


new_data


# In[32]:


printable_data = pd.DataFrame(new_data, columns = list(column_headers))


# In[33]:


printable_labels = pd.DataFrame(new_labels, columns = ["Churn"])


# In[34]:


printable_data


# In[35]:


printable_labels


# In[36]:


final_data = pd.concat([printable_labels, printable_data], axis = 1)


# In[37]:


final_data.to_csv("/Users/nissani/Desktop/Preprocessed.csv")

