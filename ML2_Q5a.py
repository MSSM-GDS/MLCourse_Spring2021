#!/usr/local/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
plt.style.use('ggplot')
from sklearn import svm, metrics, model_selection
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import plot_roc_curve



wdbc_train = pd.read_csv("./WDBC_dat/WDBC_train.csv", sep=',')
print("training set:")
print(wdbc_train.head(5))

wdbc_test = pd.read_csv("./WDBC_dat/WDBC_test.csv", sep=',')
print("test set:")
print(wdbc_test.head(5))

counts = wdbc_train['diagnosis'].value_counts()
print(counts)

###commented out because they take too long to print! ###
#pairplot = sns.pairplot(wdbc_train, hue = 'diagnosis', vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension'])
#pairplot.savefig("Pairplot_WDBC_train.png")
#plt.figure(figsize=(20,20))
#heatmap_wdbc = sns.heatmap(wdbc_train.corr(), annot = True, linewidths=.5)
#fig = heatmap_wdbc.get_figure()
#fig.savefig("heatmap_WDBC_train.png")
#countplot = sns.countplot(wdbc_train['diagnosis'], label = "Count")
#countplot.get_figure().savefig("Countplot_WDBC_train.png")
###### end of commented figure/plot ###

##create training set
X_train = wdbc_train.drop(['diagnosis'], axis=1)
print(X_train.head(5))
print(X_train.shape)

Y_train = wdbc_train['diagnosis']
print(Y_train.head(5))
print(Y_train.shape)

##create test set:
X_test = wdbc_test.drop(['diagnosis'], axis=1)
print(X_test.head(5))
print(X_test.shape)

Y_test = wdbc_test['diagnosis']
print(Y_test.head(5))
print(Y_test.shape)


##normalize trainig data
X_train_min = X_train.min()
X_train_range = (X_train - X_train_min).max()
X_train_scaled = (X_train - X_train_min)/X_train_range

##normalize test data
X_test_min = X_test.min()
X_test_range = (X_test - X_test_min).max()
X_test_scaled = (X_test - X_test_min)/X_test_range

##model fitting
print("Model fitting pre-norm")
svc_model = svm.SVC()
svc_model.fit(X_train, Y_train)
Y_predict = svc_model.predict(X_test)
cm = np.array(confusion_matrix(Y_test, Y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_benign'],
                         columns=['predicted_cancer','predicted_benign'])
print(confusion)
heatmap_confusion = sns.heatmap(confusion, annot = True, linewidths=.5)
fig = heatmap_confusion.get_figure()
fig.savefig("heatmap_Confusion.png")
print(classification_report(Y_test,Y_predict))

# plot ROC AUC prenorm
metrics.plot_roc_curve(svc_model, X_test, Y_test)
plt.show()

###model fitting postnorm
print("Model fitting postnorm")
svc_model_postnorm = svm.SVC()
svc_model_postnorm.fit(X_train_scaled, Y_train)
Y_predict_postnorm = svc_model_postnorm.predict(X_test_scaled)
cm_postnorm = confusion_matrix(Y_test, Y_predict_postnorm)
confusion_postnorm = pd.DataFrame(cm_postnorm, index=['is_cancer', 'is_benign'],
                         columns=['predicted_cancer','predicted_benign'])
print(confusion_postnorm)
heatmap_confusion_postnorm = sns.heatmap(confusion_postnorm, annot = True, linewidths=.5)
fig = heatmap_confusion_postnorm.get_figure()
fig.savefig("heatmap_Confusion_postnorm.png")
print(classification_report(Y_test,Y_predict_postnorm))

#plot ROC AUC post norm
metrics.plot_roc_curve(svc_model_postnorm, X_test_scaled, Y_test)
plt.show()
