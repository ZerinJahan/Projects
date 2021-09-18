import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle

dataset = pd.read_csv('F:/accident_2020_(1).csv')
x = dataset.iloc[:, :-1].values
y= dataset.iloc[:, 5].values

from sklearn.preprocessing import  OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x=LabelEncoder()
x[:, 0]=labelencoder_x.fit_transform(x[:, 0])
x[:, 1]=labelencoder_x.fit_transform(x[:, 1])
x[:, 2]=labelencoder_x.fit_transform(x[:, 2])
x[:, 3]=labelencoder_x.fit_transform(x[:, 3])
x[:, 4]=labelencoder_x.fit_transform(x[:, 4])

labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

# Binarize the output
from sklearn.preprocessing import label_binarize
y = label_binarize(y, classes=[0,1,2,3,4])
n_classes = 5

from sklearn.model_selection import train_test_split, cross_val_score
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state =  0)



'''from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)'''

from sklearn.naive_bayes import CategoricalNB
from sklearn.multiclass import OneVsRestClassifier
classifier =OneVsRestClassifier (CategoricalNB())
y_pred= classifier.fit(x_train,y_train).predict(x_test)


'''from sklearn.metrics import classification_report, confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test, y_pred)'''

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())


from sklearn.metrics import roc_curve, auc
from scipy import interp

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
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