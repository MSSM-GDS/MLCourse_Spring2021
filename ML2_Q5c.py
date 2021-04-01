#!/usr/local/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
plt.style.use('ggplot')

from sklearn import svm, metrics, model_selection, preprocessing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, plot_precision_recall_curve, plot_roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, LabelEncoder
from sklearn.neural_network import MLPClassifier

import seaborn as sns




wdbc_train = pd.read_csv("./WDBC_dat/WDBC_train.csv", sep=',')
print("training set:")
print(wdbc_train.head(5))

wdbc_test = pd.read_csv("./WDBC_dat/WDBC_test.csv", sep=',')
print("test set:")
print(wdbc_test.head(5))

counts = wdbc_train['diagnosis'].value_counts()
print(counts)


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
pipe = Pipeline(steps=[
    ('preprocess', StandardScaler()),
    ('classification', MLPClassifier())
])


# mlp_activation = ['identity', 'logistic', 'tanh', 'relu']
# mlp_solver = ['sgd', 'adam']
# mlp_alpha = [1e-4, 1e-3, 0.01, 0.1, 1]
# preprocess = [MinMaxScaler(), StandardScaler()]
# hidden_layer_sizes = [10, 100, 200]
# #learning_rate = ['constant', 'invscaling', 'adaptive']
# momentum = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
# random_state = 0
#
# mlp_param_grid = [
#     {
#         'preprocess': preprocess,
#         'classification__activation': mlp_activation,
#         'classification__solver': mlp_solver,
#         'classification__hidden_layer_sizes': hidden_layer_sizes,
#         'classification__alpha': mlp_alpha,
#         #'classifiation__learning_rate': learning_rate,
#         'classification__momentum': momentum,
#         'classification__random_state': [random_state]
#     }
# ]
#
# strat_k_fold = StratifiedKFold(
#     n_splits=5
# )
#
# mlp_grid = GridSearchCV(
#     pipe,
#     param_grid=mlp_param_grid,
#     cv=strat_k_fold,
#     verbose=2,
#     n_jobs=-1
# )
#
#
# print("params")
# print(mlp_grid.get_params().keys())
#
#
# mlp_grid.fit(X_train_scaled, Y_train)
# print("mlp grid values")
# # Best MLPClassifier parameters
# print(mlp_grid.best_params_)
# # Best score for MLPClassifier with best parameters
# print('\nBest F1 score for MLP: {:.2f}%'.format(mlp_grid.best_score_ * 100))
#
# best_params = mlp_grid.best_params_
# print(best_params)

## after finding best params:

mlp = MLPClassifier(
    max_iter=1000,
    alpha=1,
    activation='relu',
    solver='adam',
    random_state=0,
    momentum=0.1,
    learning_rate='constant',
    hidden_layer_sizes = 100
)

mlp.fit(X_train_scaled, Y_train)
mlp_predict = mlp.predict(X_test_scaled)
mlp_predict_proba = mlp.predict_proba(X_test_scaled)[:, 1]

print('MLP Accuracy: {:.2f}%'.format(accuracy_score(Y_test, mlp_predict) * 100))
print('MLP AUC: {:.2f}%'.format(roc_auc_score(Y_test, mlp_predict_proba) * 100))
print('MLP Classification report:\n\n', classification_report(Y_test, mlp_predict))
print('MLP Training set score: {:.2f}%'.format(mlp.score(X_train_scaled, Y_train) * 100))
print('MLP Testing set score: {:.2f}%'.format(mlp.score(X_test_scaled, Y_test) * 100))
#plot ROC AUC post norm
metrics.plot_roc_curve(mlp, X_test_scaled, Y_test)
plt.show()
