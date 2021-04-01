#!/usr/local/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
plt.style.use('ggplot')

from sklearn import svm, metrics, model_selection, preprocessing
from sklearn.metrics import classification_report, confusion_matrix, plot_precision_recall_curve, plot_roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import make_pipeline

import seaborn as sns




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
print(X_train.describe())

Y_train = wdbc_train['diagnosis']
print(Y_train.head(5))
print(Y_train.shape)

##create test set:
X_test = wdbc_test.drop(['diagnosis'], axis=1)
print(X_test.head(5))
print(X_test.shape)
print(X_test.describe())
#X_test.plot()

Y_test = wdbc_test['diagnosis']
print(Y_test.head(5))
print(Y_test.shape)

##scaling so that me
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled_features = scaler.transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled_features, index=X_train.index, columns=X_train.columns)
print(X_train_scaled.shape)
print(X_train_scaled)


X_test_scaled_features = scaler.transform (X_test)
X_test_scaled = pd.DataFrame(X_test_scaled_features, index=X_test.index, columns=X_test.columns)
print(X_test_scaled.shape)
print(X_test_scaled)

###model fitting postnorm: tried both linear models.
print("Model fitting postnorm")
param_grid = {'C': [0.1, 1, 2, 3, 5, 10, 100], 'gamma': [1, 0.1, 0.05, 0.03, 0.01, 0.005, 0.001], 'kernel': ['rbf']}
grid = GridSearchCV(svm.SVC(),param_grid,refit=True,verbose=4, cv=5)
search = grid.fit(X_train_scaled,Y_train)
print("Best Params:")
print(search.best_params_)
print(search.best_estimator_)
gridsearch_predictions = search.predict(X_test_scaled)
cm_postnorm = confusion_matrix(Y_test, gridsearch_predictions)
confusion_postnorm = pd.DataFrame(cm_postnorm, index=['is_cancer', 'is_benign'],
                         columns=['predicted_cancer','predicted_benign'])
print(confusion_postnorm)
heatmap_confusion_postnorm = sns.heatmap(confusion_postnorm, annot = True, linewidths=.5)
fig = heatmap_confusion_postnorm.get_figure()
fig.savefig("heatmap_Q5b_Confusion_GridSearchCV_postnorm.png")
print(classification_report(Y_test,gridsearch_predictions))

#plot ROC AUC post norm
metrics.plot_roc_curve(grid, X_test_scaled, Y_test)
plt.show()
ROC = roc_auc_score(Y_test, grid.decision_function(X_test_scaled))
print(ROC)
