
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn as sk


# In[2]:


data = pd.read_csv("/Users/nissani/Desktop/DataScience/cleaned_data.csv")


# In[3]:


data = data.drop(['Unnamed: 0'], axis = 1)


# In[4]:


data.head()


# In[5]:


list(data)


# In[6]:


data['wsid'].value_counts()


# In[7]:


data['elvt'].value_counts()


# In[8]:


data['inme'].value_counts()


# In[9]:


#Make a list of labels where 0 means it is not raining and 1 means that it is raining.
raining = []
for i in range(len(data['prcp'])):
    if data['prcp'][i] > 0:
        raining.append(1)
    else:
        raining.append(0)


# In[10]:


raining.index(1)


# In[11]:


len(raining)


# In[12]:


raining = np.array([raining]
                  )


# In[13]:


helper = data.values


# In[14]:


raining.shape


# In[15]:


helper.shape


# In[16]:


new_data = np.concatenate((helper,raining.T)
                          
                          , axis = 1)


# In[17]:


new_data


# In[18]:


columns = list(data)
columns.append('rainingLabel')


# In[19]:


labeled_data = pd.DataFrame(new_data, columns = columns)


# In[20]:


labeled_data.head()


# In[21]:


labeled_data['rainingLabel'].value_counts()


# In[22]:


9094/136441


# In[23]:


labeled_data['mdct'].value_counts()


# In[24]:


labeled_data["city"].value_counts()


# In[25]:


labeled_data


# In[26]:


vit_data = labeled_data.loc[labeled_data['city'] == 'Vitória']


# In[27]:


vit_data.head()


# In[28]:


vit_data['rainingLabel'].value_counts()[1]/vit_data['rainingLabel'].value_counts()[0]


# In[29]:


final_data = vit_data.drop(['city', "date", "mdct", 'wsid', 'elvt', 'inme'], axis=1)


# In[30]:


final_data.head()


# In[31]:


final_data['prcp'].value_counts()


# In[32]:


raining = pd.DataFrame(final_data['rainingLabel'])


# In[33]:


raining


# In[34]:


prcp = pd.DataFrame(final_data['prcp'])


# In[35]:


labels = pd.concat([prcp, raining], axis = 1)


# In[36]:


labels


# In[37]:


final_data = final_data.drop(['prcp','rainingLabel'], axis = 1)


# In[38]:


def featurizer(column, num):
    tmp = [0]*num + list(final_data[column])
    for i in range(1,num):
        tmp.remove(tmp[-i])
    new_column = column + '%s'%num
    return tmp, new_column


# In[39]:


column_headers = list(final_data)


# In[40]:


column_headers = column_headers[4:]


# In[41]:


column_headers


# In[42]:


new_data = []
headers = []

for el in column_headers:
    for i in range(3):
        tmp, new = featurizer(el, i+1)
        new_data.append(tmp)
        headers.append(new)


# In[43]:


headers


# In[44]:


new_data = np.array(new_data)


# In[45]:


featured_data = pd.DataFrame.from_records(new_data.T, columns = headers)


# In[46]:


featured_data


# In[47]:


column_names = list(final_data)
final_data = final_data.values


# In[48]:


final_data = pd.DataFrame(final_data, columns=column_names)


# In[49]:


new_final_data = pd.concat([final_data, featured_data], axis = 1)


# In[50]:


new_final_data


# In[54]:


new_final_data = new_final_data.drop(81837)


# In[55]:


len(new_final_data)


# In[56]:


len(labels)

