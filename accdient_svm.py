#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


dataset = pd.read_csv('F:/accident_final.csv')
x = dataset.iloc[:, :-1].values
y= dataset.iloc[:, 5].values


# In[3]:


from sklearn.preprocessing import LabelEncoder
labelencoder_x=LabelEncoder()
x[:, 0]=labelencoder_x.fit_transform(x[:, 0])
x[:, 1]=labelencoder_x.fit_transform(x[:, 1])
x[:, 2]=labelencoder_x.fit_transform(x[:, 2])
x[:, 3]=labelencoder_x.fit_transform(x[:, 3])
x[:, 4]=labelencoder_x.fit_transform(x[:, 4])


# In[4]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=0)


# In[5]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


# In[6]:


from sklearn.svm import SVC
classifier= SVC(kernel='linear', random_state=0)
classifier.fit(x_train, y_train)


# In[7]:


y_pred = classifier.predict(x_test)


# In[9]:


from sklearn.metrics import classification_report, confusion_matrix
cr=classification_report(y_test, y_pred)


# In[10]:


cr


# In[11]:


from sklearn.metrics import classification_report, confusion_matrix


# In[12]:


print(classification_report(y_test, y_pred))


# In[ ]:




