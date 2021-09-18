import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import pandas as pd

# Import some data to play with
dataset = pd.read_csv('F:/4.2/thesis/weather prediction/springer_fahim_weathr/2-1_HeatIndex_preprocessed.csv')
x = dataset.iloc[:, 0:5].values
y= dataset.iloc[:, 6].values
new_data = pd.read_csv('F:/4.2/thesis/weather prediction/springer_fahim_weathr/input_csv_final.csv')


from sklearn.preprocessing import   LabelEncoder



labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)


# Binarize the output
from sklearn.preprocessing import label_binarize
y = label_binarize(y, classes=[0,1,2,3,4])
n_classes = 5


from sklearn.model_selection import train_test_split, cross_val_score
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state =  0)

y_train = np.argmax(y_train, axis=1)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
new_data= sc_x.transform(new_data)


#creating model with keras
 
from keras.models import Sequential
from keras.layers import Dense

#initial ann

classifier = Sequential()

#adding input & hiddenlayer

classifier.add(Dense(output_dim = 24, init = 'uniform', activation = 'relu', input_dim = 5))

#addin 2nd hidden layer

classifier.add(Dense(output_dim = 24, init = 'uniform', activation = 'relu'))


#output layer

classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'softmax'))

#compile ann

classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#fit to train set

classifier.fit(x_train, y_train, batch_size = 100, epochs = 30)

y_pred = classifier.predict(x_test)
y_pred_2 = classifier.predict(new_data)


'''from sklearn.metrics import classification_report, confusion_matrix
cm=confusion_matrix(y_test,np.argmax(y_pred, axis = 1))
cr=classification_report(y_test,np.argmax(y_pred, axis = 1))'''

from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
roc_auc[i] = auc(fpr[i], tpr[i])


# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


plt.figure()
lw = 2

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=2)

colors = cycle(['red', 'aqua', 'green', 'darkorange', 'purple'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()
