
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib as plot
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import seaborn
import matplotlib.pyplot as plt


# In[2]:


churn_data = pd.read_csv("/Users/nissani/Desktop/WA_Fn_UseC_Telco_Customer_Churn.csv")


# In[3]:


#Collect headers of the dataframe and put data into a numpy array for use.
column_headers = np.array(list(churn_data))[1:-1]
column_headers = list(column_headers)
column_headers_2 = np.array(column_headers)
workable_data = churn_data.values

#Create an array of one-hot encoded features to properly remove them database
columns = [column_headers.index('gender'), 
           column_headers.index('Partner'), 
           column_headers.index('Dependents'), 
           column_headers.index('PhoneService'), 
           column_headers.index('MultipleLines'), 
           column_headers.index('InternetService'), 
           column_headers.index('OnlineSecurity'), 
           column_headers.index('OnlineBackup'), 
           column_headers.index('DeviceProtection'), 
           column_headers.index('TechSupport'), 
           column_headers.index('StreamingTV'), 
           column_headers.index('StreamingMovies'), 
           column_headers.index('Contract'), 
           column_headers.index('PaperlessBilling'), 
           column_headers.index('PaymentMethod')]


# In[4]:


#Concatenate one-hot encoded features with other features and then drop the original features.
churn_data = pd.concat([churn_data, pd.get_dummies(churn_data[column_headers_2[columns]])], axis = 1)
churn_data = churn_data.drop(column_headers_2[columns], axis = 1)


# In[5]:


#Collect the churn data for future use.
churn = churn_data['Churn']
churn = pd.DataFrame(churn)
churn = churn.values


# In[6]:


#Swap the place of the churn labels
churn_data = churn_data.drop('Churn', axis = 1)
workable_data = pd.concat([churn_data, pd.DataFrame(churn, columns = ['Churn'])], axis = 1)


# In[7]:


#Collect the column headers for the database and make the labels 1 or 0.
new_columns = list(workable_data)
workable_data = workable_data.values
for i in range(len(workable_data)):
    if workable_data[i][-1] == 'No':
        workable_data[i][-1] = 0
    if workable_data[i][-1] == 'Yes':
        workable_data[i][-1] = 1


# In[8]:


churn_data_new = pd.DataFrame(workable_data, columns = new_columns)


# In[9]:


churn_data_new['Churn'].value_counts().plot(kind = 'bar')


# In[10]:


#Replace unknown values with 0.
workable_data = workable_data[:, 1:-1]
for i in range(len(workable_data)):
    if workable_data[i, 3] == ' ':
        workable_data[i,3] = 0
    else:
        workable_data[i] = workable_data[i].astype(np.float)


# In[11]:


#Collect data on the range and variance of each feature.
variance_data = workable_data.T
variance = []
for i in range(len(variance_data)):
    variance.append(np.var(variance_data[i]))

range_data = []
for i in range(len(variance_data)):
    single_range = np.amax(variance_data[i])-np.amin(variance_data[i])
    range_data.append(single_range)


# In[12]:


#Create box plot to justify standardization.
variance_data = workable_data
variance_data = pd.DataFrame(variance_data, columns = [i for i in range(len(new_columns[1:-1]))])
variance_data.plot(kind = 'box', figsize = (18, 5))


# In[13]:


#Standardize data.
scaler = StandardScaler()
scaler.fit(workable_data)
data = scaler.transform(workable_data)


# In[14]:


#Create synthetic data to balance dataset.
sm = SMOTE(random_state = 28)
columns_without_churn = new_columns[1:-1]
columns_without_churn = np.array(columns_without_churn)
new_data, new_labels = sm.fit_resample(data, churn)


# In[15]:


for i in range(len(new_labels)):
    if new_labels[i] == 'No':
        new_labels[i] = 0
    if new_labels[i] == 'Yes':
        new_labels[i] = 1


# In[16]:


#Create final cleaned database
printable_data = pd.DataFrame(new_data, columns = list(new_columns[1:-1]))
printable_labels = pd.DataFrame(new_labels, columns = ['Churn'])
final_data = pd.concat([printable_labels, printable_data], axis = 1)


# In[17]:


#Create csv
final_data.to_csv("/Users/nissani/Desktop/Preprocessed.csv")

