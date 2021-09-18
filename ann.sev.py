# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 23:02:12 2020

@author: zerin
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from itertools import cycle

dataset = pd.read_csv('F:/severity_2020_(1).csv')
x = dataset.iloc[:, :-1].values
y= dataset.iloc[:, 5].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x=LabelEncoder()
x[:, 0]=labelencoder_x.fit_transform(x[:, 0])
x[:, 1]=labelencoder_x.fit_transform(x[:, 1])
x[:, 2]=labelencoder_x.fit_transform(x[:, 2])
x[:, 3]=labelencoder_x.fit_transform(x[:, 3])
x[:, 4]=labelencoder_x.fit_transform(x[:, 4])

onehotencoder = OneHotEncoder()
x = onehotencoder.fit_transform(x).toarray()

labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)



from sklearn.model_selection import train_test_split, cross_val_score
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


'''from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)'''


#creating model with keras
 
from keras.models import Sequential
from keras.layers import Dense

#initial ann

classifier = Sequential()

#adding input & hiddenlayer

classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'relu', input_dim = 60))

#addin 2nd hidden layer

classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'relu'))

#output layer

classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'softmax'))

#compile ann

classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#fit to train set
from sklearn.metrics import roc_curve
classifier.fit(x_train, y_train, batch_size = 100, epochs = 20)

y_pred = classifier.predict(x_test)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)

'''from sklearn.metrics import classification_report, confusion_matrix
cm=confusion_matrix(y_test,np.argmax(y_pred, axis = 1))
cr=classification_report(y_test,np.argmax(y_pred, axis = 1))'''

from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()



