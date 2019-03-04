#!/usr/bin/env python
# coding: utf-8

# In[302]:


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

RAW_DATA_FNAME = "data/Preprocessed.csv"


# In[253]:


raw_data = pd.read_csv(RAW_DATA_FNAME, index_col = [0])
# balanced_data = pd.read_csv(BALANCE_DATA_FNAME, index_col = [0])


# In[254]:


X = raw_data.drop(["Churn"], axis = 1)
feature_names = X.columns
X = X.values
y = raw_data["Churn"]
y = y.values


# In[297]:


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
            model = self.model_specification()
            model.fit(current_X, current_y)
            model_list.append(model)
            test_X = self.X[test_indexes, :]
            test_y = self.y[test_indexes]
            pred_y = model.predict(test_X)
            fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_y)
            auc_list.append(metrics.auc(fpr, tpr))
            acc_list.append(metrics.accuracy_score(test_y, pred_y, normalize=True))
    def show_evaluations(self):
        avg_acc_balanced = np.mean(self.accuracies_balanced)
        avg_auc_balanced = np.mean(self.aucs_balanced)
        avg_acc = np.mean(self.accuracies)
        avg_auc = np.mean(self.aucs)
        print("In balanced data, average test accuracy is %.4f, average AUC is %.4f"%               (avg_acc_balanced, avg_auc_balanced))
        print("In original data, average test accuracy is %.4f, average AUC is %.4f"%               (avg_acc, avg_auc))
        return avg_acc, avg_auc


# In[298]:


lr = Model(LogisticRegression, X, y, 10)
lr.train()
lr.train(balanced = False)
rf = Model(RandomForestClassifier, X, y, 10)
rf.train()
rf.train(balanced = False)


# In[299]:


lr.show_evaluations()


# In[300]:


rf.show_evaluations()


# analyze model for business solution

# In[328]:


lr_coefs = np.zeros(len(lr.models[0].coef_[0]))

for model in lr.models:
    lr_coefs += model.coef_.ravel()
print(lr_coefs)
fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
plt.bar(range(len(lr_coefs)), lr_coefs)
plt.xticks(range(len(lr_coefs)), feature_names, rotation = "vertical")
plt.ylabel("coefficient")
plt.title('logistic regression coefficients')
plt.show()

