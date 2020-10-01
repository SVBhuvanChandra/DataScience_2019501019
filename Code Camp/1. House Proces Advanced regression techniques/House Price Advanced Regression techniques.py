#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Need to import required python libraries

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mserr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[2]:


# TRAIN data

train_x = pd.read_csv("C:\\Users\\Bhuvan PC\\Downloads\\codecamptrain.csv",index_col=0)
train_x


# In[3]:


# Target variable

train_y = train_x['SalePrice']
train_x.drop('SalePrice',axis=1,inplace=True)
train_x


# In[4]:


# TEST data

test_x = pd.read_csv("C:\\Users\\Bhuvan PC\\Downloads\\codecamptest.csv",index_col=0)
test_x


# In[5]:


# As a first step of pre-processing remove columns with null value ratio greater than provided limit
sample_size = len(train_x)
sample_size


# In[6]:


# on Train dataset
train_col_with_nullvalues=[[col,float(train_x[col].isnull().sum())/float(sample_size)] for col in train_x.columns if train_x[col].isnull().sum()]
train_col_with_nullvalues


# In[7]:


# on Test dataset
test_col_with_nullvalues=[[col,float(test_x[col].isnull().sum())/float(sample_size)] for col in test_x.columns if test_x[col].isnull().sum()]
test_col_with_nullvalues


# In[8]:


print(len(train_col_with_nullvalues))
print(len(test_col_with_nullvalues))


# In[9]:


train_col_to_drop=[x for (x,y) in train_col_with_nullvalues if y>0.3]
train_col_to_drop


# In[10]:


test_col_to_drop=[x for (x,y) in test_col_with_nullvalues if y>0.3]
test_col_to_drop


# In[11]:


train_x.drop(train_col_to_drop,axis=1,inplace=True)
train_x


# In[12]:


test_x.drop(test_col_to_drop,axis=1,inplace=True)
test_x


# As a second pre-processing step find all categorical columns and one hot  encode them. 
# Before one hot encode fill all null values with dummy in those columns.  
# Some categorical columns in train_x may not have null values in train_x but have null values in test_x.
# To overcome this problem we will add a row to the train_x with all dummy values for categorical values. 
# Once one hot encoding is complete drop the added dummy column

# In[13]:


# on Train dataset
train_categorical_columns=[col for col in train_x.columns if train_x[col].dtype==object]
train_categorical_columns
print(len(train_categorical_columns))
train_ordinal_columns=[col for col in train_x.columns if col not in train_categorical_columns]
train_ordinal_columns
print(len(train_ordinal_columns))


# In[14]:


# on Test datset
test_categorical_columns=[col for col in test_x.columns if test_x[col].dtype==object]
test_categorical_columns
print(len(test_categorical_columns))
test_ordinal_columns=[col for col in test_x.columns if col not in test_categorical_columns]
test_ordinal_columns
print(len(test_ordinal_columns))


# In[15]:


dummy_row=list()
for col in train_x.columns:
    if col in train_categorical_columns:
        dummy_row.append("dummy")
    else:
        dummy_row.append("")
new_row = pd.DataFrame([dummy_row],columns=train_x.columns)
train_x = pd.concat([train_x,new_row],axis=0, ignore_index=True)
train_x


# In[16]:


test_x.shape


# In[17]:


for col in train_categorical_columns:
    train_x[col].fillna(value="dummy",inplace=True)
    test_x[col].fillna(value="dummy",inplace=True)
    
enc = OneHotEncoder(drop='first',sparse=False)
enc.fit(train_x[train_categorical_columns])
trainx_enc=pd.DataFrame(enc.transform(train_x[train_categorical_columns]))
trainx_enc.columns=enc.get_feature_names(train_categorical_columns)

testx_enc=pd.DataFrame(enc.transform(test_x[train_categorical_columns]))
testx_enc.columns=enc.get_feature_names(train_categorical_columns)

train_x = pd.concat([train_x[train_ordinal_columns],trainx_enc],axis=1,ignore_index=True)
test_x = pd.concat([test_x[train_ordinal_columns],testx_enc],axis=1,ignore_index=True)

train_x.drop(train_x.tail(1).index,inplace=True)

train_x


# In[18]:


train_x.tail(3)


# In[19]:


test_x.tail(3)


# In[20]:


print(train_x.shape)
print(test_x.shape)


# In[21]:


test_x.head(3)


# In[22]:


test_x.tail(3)


# In[23]:


imputer = KNNImputer(n_neighbors=2)
imputer.fit(train_x)
trainx_filled = imputer.transform(train_x)
trainx_filled=pd.DataFrame(trainx_filled,columns=train_x.columns)
trainx_filled


# In[24]:


testx_filled = imputer.transform(test_x)
testx_filled=pd.DataFrame(trainx_filled,columns=test_x.columns)
testx_filled.reset_index(drop=True,inplace=True)

testx_filled


# In[25]:


test_x.shape


# In[26]:


# Applying Standard Scalar.

scaler = preprocessing.StandardScaler().fit(train_x)
train_x=scaler.transform(trainx_filled)
test_x=scaler.transform(testx_filled)


# In[27]:


train_x = pd.DataFrame(train_x)
train_x


# In[28]:


test_x = pd.DataFrame(test_x)

test_x.drop(test_x.tail(1).index,inplace=True)
test_x


# In[29]:


# pca = PCA().fit(train_x)
# itemindex = np.where(np.cumsum(pca.explained_variance_ratio_)>0.999)
# # print('np.cumsum(pca.explained_variance_ratio_)', np.cumsum(pca.explained_variance_ratio_))

# #Plotting the Cumulative Summation of the Explained Variance
# plt.figure(np.cumsum(pca.explained_variance_ratio_)[0])
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Number of Components')
# plt.ylabel('Variance (%)') #for each component
# plt.title('Principal Components Explained Variance')
# plt.show()

# pca_std = PCA(n_components=itemindex[0][0]).fit(train_x)
# train_x = pca_std.transform(train_x)
# test_x = pca_std.transform(test_x)


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(train_x, train_y.values.ravel(), test_size=0.3, random_state=42)


# In[41]:


# Linear Regression

Linreg = LinearRegression()
Linreg.fit(X_train, y_train)

print(sqrt(mean_squared_error(y_test, Linreg.predict(X_test))))
print('R2 Value/Coefficient of Determination: {}'.format(Linreg.score(X_test,y_test)))
  


# In[32]:


# Ridge Regression

from math import sqrt
from sklearn.metrics import r2_score, mean_squared_error
Ridgereg = Ridge(alpha = 0.5,tol = 0.1)
Ridgereg = Ridgereg.fit(X_train,y_train)

print(sqrt(mean_squared_error(y_test, Ridgereg.predict(X_test))))
print('R2 Value/Coefficient of Determination: {}'.format(Ridgereg.score(X_test,y_test)))
                                                                       


# In[33]:


# Elastic Regression

from sklearn.linear_model import ElasticNet
Elas = ElasticNet(alpha=0.001, normalize=True)
Elas.fit(X_train, y_train)

# print(sqrt(mean_squared_error(ytrain, Elas.predict(xtrain))))
print(sqrt(mserr(y_test, Elas.predict(X_test))))
print('R2 Value/Coefficient of Determination: {}'.format(Elas.score(X_test, y_test)))



# In[34]:


# Lassoreg = Lasso(alpha = 0.5,tol = 0.1)
# Lassoreg = Lassoreg.fit(X_train,y_train)
# print(Ridgereg.score(X_train,y_train))
# print(Ridgereg.score(X_test,y_test))

from sklearn.linear_model import Lasso
from math import sqrt
from sklearn.metrics import r2_score, mean_squared_error


lassoreg = Lasso(alpha=0.001, normalize=True)
lassoreg.fit(X_train, y_train)
# lassoreg.predict(X_train)
print(sqrt(mean_squared_error(y_test, lassoreg.predict(X_test))))
print('R2 Value/Coefficient of Determination: {}'.format(lassoreg.score(X_test, y_test)))



# In[40]:


test_prediction = pd.DataFrame(Elas.predict(test_x),columns=['SalePrice'])
test_prediction.index.name = 'Id'
test_prediction.to_csv("C:\\Users\\Bhuvan PC\\Downloads\\final_test_pred.csv")

