
# coding: utf-8

# In[146]:


import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


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


# In[51]:


new_final_data = new_final_data.drop(81837)


# In[52]:


len(new_final_data)


# In[53]:


len(labels)


# In[54]:


new_final_data = new_final_data.drop(['yr', 'mo','da','hr'], axis = 1)


# In[55]:


X = new_final_data.values
y = labels.values[:,1]
y = y.astype(int)


# In[56]:


class Model:
    def __init__(self, model_specification, X, y, n_folds):
        self.model_specification = model_specification
        self.X = X
        self.y = y
        self.n_samples = X.shape[0]
        # create cross validation folds
        randperm = np.random.permutation(range(self.n_samples)).tolist()
        self.folds = []
        fold_size = (int)(self.n_samples / n_folds)
        for i in range(n_folds):
            start = i * fold_size
            end = (int)((i + 1) * fold_size)
            self.folds.append(randperm[(i * fold_size):((i+1) * fold_size)])
        self.models = []
        self.aucs = []
        self.accuracies = []
        self.models_balanced = []
        self.aucs_balanced = []
        self.accuracies_balanced = []
        self.predictions = []
        self.test_indices = []
        self.data = []
        self.labels = []
        
    def train(self, balanced = True):
        if balanced:
            model_list = self.models_balanced
            acc_list = self.accuracies_balanced
            auc_list = self.aucs_balanced
        else:
            model_list = self.models
            acc_list = self.accuracies
            auc_list = self.aucs
        for fold in self.folds:
            test_indexes = fold
            train_indexes = [i for i in range(self.n_samples) if not i in test_indexes]
            current_X = self.X[train_indexes, :]
            current_y = self.y[train_indexes]
            if balanced:
                sm = SMOTE(random_state = (int)(time.time()))
                current_X, current_y = sm.fit_resample(current_X, current_y)
            self.data.append(current_X)
            self.labels.append(current_y)
            model = self.model_specification()
            model.fit(current_X, current_y)
            model_list.append(model)
            test_X = self.X[test_indexes, :]
            test_y = self.y[test_indexes]
            pred_y = model.predict(test_X)
            self.predictions.append((pred_y, balanced))
            self.test_indices.append(test_indexes)
            fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_y)
            auc_list.append(metrics.auc(fpr, tpr))
            acc_list.append(metrics.accuracy_score(test_y, pred_y, normalize=True))
    def show_evaluations(self):
        avg_acc_balanced = np.amax(self.accuracies_balanced)
        avg_acc_balanced_index = np.argmax(self.accuracies_balanced)
        avg_auc_balanced = np.mean(self.aucs_balanced)
        avg_acc = np.amax(self.accuracies)
        avg_acc_index = np.argmax(self.accuracies)
        avg_auc = np.mean(self.aucs)
        print("In balanced data, highest test accuracy is %.4f at %.4f, average AUC is %.4f"%(avg_acc_balanced, avg_acc_balanced_index, avg_auc_balanced))
        print("In original data, highest test accuracy is %.4f at %.4f, average AUC is %.4f"%(avg_acc, avg_acc_index, avg_auc))
        return avg_acc, avg_auc


# In[57]:


lr = Model(LogisticRegression, X, y, 10)
lr.train()
lr.train(balanced = False)


lr.show_evaluations()


# In[102]:


amount = np.array(lr.data[19])


# In[103]:


raining = np.array(lr.labels[19])


# In[104]:


raining_predictions = np.array(lr.predictions[19][0])


# In[105]:


indices = np.array(lr.test_indices[19])


# In[106]:


len(raining_predictions)


# In[107]:


len(indices)


# In[108]:


len(amount)


# In[109]:


len(raining)


# In[87]:


prcp_labels = labels.values[:,0]


# In[88]:


prcp_labels


# In[89]:


prcp_labels = prcp_labels.astype(float)


# In[90]:


prcp_labels


# In[91]:


len(prcp_labels)


# In[93]:


new_final_data = new_final_data.values


# In[95]:


new_final_data = new_final_data.astype(float)


# In[96]:


new_final_data


# In[118]:


raining_index = []
for i in range(len(raining_predictions)):
    if raining_predictions[i] == 1:
        raining_index.append(i)


# In[119]:


len(raining_index)


# In[120]:


index1 = indices[raining_index]


# In[122]:


prcp_labels_nonzero = prcp_labels[index1]


# In[132]:


len(prcp_labels_nonzero)


# In[124]:


test_data = new_final_data[index1]


# In[128]:


len(test_data)


# In[142]:


train_data = [new_final_data[i] for i in range(len(new_final_data)) if i not in index1]


# In[143]:


len(train_data)


# In[144]:


train_labels = [prcp_labels[i] for i in range(len(prcp_labels)) if i not in index1]


# In[145]:


len(train_labels)


# In[167]:


linear = LinearRegression()


# In[168]:


linear.fit(train_data, train_labels)


# In[169]:


linear.score(test_data, prcp_labels_nonzero)


# In[170]:


linear.predict(test_data)


# In[164]:


prcp_labels_nonzero

