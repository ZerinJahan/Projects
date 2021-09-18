import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import pandas as pd

# Import some data to play with
dataset = pd.read_csv('F:/accident_2020_(1).csv')
x = dataset.iloc[:, :-1].values
y= dataset.iloc[:, 5].values



from sklearn.preprocessing import  OneHotEncoder, LabelEncoder

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


# Binarize the output
from sklearn.preprocessing import label_binarize
y = label_binarize(y, classes=[0,1,2,3,4])
n_classes = 5


from sklearn.model_selection import train_test_split, cross_val_score
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state =  0)




'''#classifier random forest
from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(n_estimators = 30, criterion = 'entropy')
classifier.fit(x_train, y_train)'''

'''#classifier decision tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_train,y_train)'''


'''#classifier knn
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=20, metric='minkowski', p=2)
classifier.fit(x_train,y_train)'''


from sklearn.naive_bayes import CategoricalNB
classifier= CategoricalNB()
classifier.fit(x_train, y_train)
y_pred = classifier.fit(x_train,y_train)



'''from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(activation='relu', max_iter = 1500, learning_rate_init = 0.1, hidden_layer_sizes=(100,2))
classifier.fit(x_train,y_train)'''


y_pred = classifier.predict(x_test)

'''from sklearn.metrics import classification_report, confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)'''


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, y_pred,average='micro')
print('Precision_macro: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, y_pred,average='micro')
print('Recall_macro: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, y_pred,average='micro')
print('F1 score_macro: %f' % f1)


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
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'])
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
plt.savefig('F:/pics/rf.png', dpi=400)
plt.show()