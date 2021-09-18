# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 20:36:05 2020

@author: zerin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('F:/accident_2020.csv')
x = dataset.iloc[:, :-1].values
y= dataset.iloc[:, 6].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x=LabelEncoder()
x[:, 0]=labelencoder_x.fit_transform(x[:, 0])
x[:, 1]=labelencoder_x.fit_transform(x[:, 1])
x[:, 2]=labelencoder_x.fit_transform(x[:, 2])
x[:, 3]=labelencoder_x.fit_transform(x[:, 3])
x[:, 4]=labelencoder_x.fit_transform(x[:, 4])
ct = ColumnTransformer([("Place", OneHotEncoder(), [0])], remainder = 'passthrough')
ct = ColumnTransformer([("Month", OneHotEncoder(), [1])], remainder = 'passthrough')
ct = ColumnTransformer([("Time", OneHotEncoder(), [2])], remainder = 'passthrough')
ct = ColumnTransformer([("Junction", OneHotEncoder(), [3])], remainder = 'passthrough')
ct = ColumnTransformer([("Vehicle", OneHotEncoder(), [4])], remainder = 'passthrough')
x = ct.fit_transform(x)
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)


from sklearn.model_selection import train_test_split, cross_val_score
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=0)

'''from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)'''

#set svm to traing
from sklearn.svm import SVC
classifier= SVC(kernel='linear', random_state=0)
classifier.fit(x_train, y_train)

#result
y_pred = classifier.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test, y_pred)

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())




   