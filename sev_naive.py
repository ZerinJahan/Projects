# -*- coding: utf-8 -*-
"""
Created on Mon May 18 19:31:19 2020

@author: zerin
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('F:/severity_2020.csv')
x = dataset.iloc[:, :-1].values
y= dataset.iloc[:, 6].values

from sklearn.preprocessing import LabelEncoder
labelencoder_x=LabelEncoder()
x[:, 0]=labelencoder_x.fit_transform(x[:, 0])
x[:, 1]=labelencoder_x.fit_transform(x[:, 1])
x[:, 2]=labelencoder_x.fit_transform(x[:, 2])
x[:, 3]=labelencoder_x.fit_transform(x[:, 3])
x[:, 4]=labelencoder_x.fit_transform(x[:, 4])
x[:, 5]=labelencoder_x.fit_transform(x[:, 5])
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
kf = KFold(n_splits=20, random_state=None, shuffle=True)
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.5)



'''from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)'''

from sklearn.naive_bayes import CategoricalNB
classifier = CategoricalNB()
classifier.fit(x_train,y_train)


y_pred = classifier.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test, y_pred)





