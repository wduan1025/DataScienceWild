
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
from sklearn.preprocessing import StandardScaler


# In[5]:


churn_data = pd.read_csv("/Users/nissani/Desktop/WA_Fn_UseC_Telco_Customer_Churn.csv")


# In[6]:


churn_data.head()


# In[7]:


list(churn_data)


# In[8]:


column_headers = np.array(list(churn_data))[1:-1]


# In[9]:


column_headers = list(column_headers)


# In[10]:


column_headers_2 = np.array(column_headers)


# In[11]:


workable_data = churn_data.values


# In[12]:


workable_data


# In[13]:


columns = [column_headers.index('gender'), column_headers.index('Partner'), column_headers.index('Dependents'), column_headers.index('PhoneService'), column_headers.index('MultipleLines'), column_headers.index('InternetService'), column_headers.index('OnlineSecurity'), column_headers.index('OnlineBackup'), column_headers.index('DeviceProtection'), column_headers.index('TechSupport'), column_headers.index('StreamingTV'), column_headers.index('StreamingMovies'), column_headers.index('Contract'), column_headers.index('PaperlessBilling'), column_headers.index('PaymentMethod')]


# In[14]:


churn_data = pd.concat([churn_data, pd.get_dummies(churn_data[column_headers_2[columns]])], axis = 1)


# In[15]:


churn_data.shape


# In[16]:


churn_data = churn_data.drop(column_headers_2[columns], axis = 1)


# In[17]:


churn_data.shape


# In[18]:


churn_data


# In[19]:


churn = churn_data['Churn']


# In[20]:


churn = pd.DataFrame(churn)


# In[21]:


churn = churn.values


# In[22]:


c1 = 0
c2 = 0
for el in churn:
    if el == 'Yes':
        c1 += 1
    if el == 'No':
        c2 += 1


# In[23]:


c1


# In[24]:


c2


# In[25]:


churn_data = churn_data.drop('Churn', axis = 1)


# In[26]:


churn_data


# In[27]:


workable_data = pd.concat([churn_data, pd.DataFrame(churn, columns = ['Churn'])], axis = 1)


# In[28]:


workable_data


# In[29]:


new_columns = list(workable_data)


# In[30]:


workable_data = workable_data.values


# In[31]:


for i in range(len(workable_data)):
    if workable_data[i][-1] == 'No':
        workable_data[i][-1] = 0
    if workable_data[i][-1] == 'Yes':
        workable_data[i][-1] = 1


# In[32]:


workable_data


# In[33]:


churn_data_new = pd.DataFrame(workable_data, columns = new_columns)


# In[34]:


churn_data_new


# In[36]:


churn_data_new['Churn'].value_counts().plot(kind = 'bar')


# In[37]:


workable_data = workable_data[:, 1:-1]


# In[38]:


workable_data


# In[39]:


for i in range(len(workable_data)):
    if workable_data[i, 3] == ' ':
        workable_data[i,3] = 0
    else:
        workable_data[i] = workable_data[i].astype(np.float)


# In[40]:


workable_data.shape


# In[41]:


scaler = StandardScaler()


# In[42]:


scaler.fit(workable_data)


# In[43]:


data = scaler.transform(workable_data)


# In[44]:


data


# In[45]:


sm = SMOTE(random_state = 28)


# In[46]:


columns_without_churn = new_columns[1:-1]


# In[47]:


columns_without_churn = np.array(columns_without_churn)


# In[48]:


data.shape


# In[49]:


new_data, new_labels = sm.fit_resample(data, churn)


# In[50]:


len(new_data)


# In[51]:


from collections import Counter
Counter(new_labels)


# In[53]:


printable_data = pd.DataFrame(data, columns = list(new_columns[1:-1]))


# In[55]:


printable_data


# In[58]:


churn_data_new['Churn']


# In[57]:


final_data = pd.concat([churn_data_new['Churn'], printable_data], axis = 1)


# In[59]:


final_data.to_csv("/Users/nissani/Desktop/Preprocessed.csv")

