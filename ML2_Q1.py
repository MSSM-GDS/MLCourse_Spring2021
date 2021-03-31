#!/usr/local/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
plt.style.use('ggplot')
from sklearn import svm

###Class 1: (0, 2), (1.5, 3), (2, 0)
###Class 2: (-1, 1), (-1, -0.5), (1, -1)

#x = [0, 1.5, 2, -1, -1, 1]
#y = [2, 3, 0, 1, -0.5, -1]
#plt.scatter(x,y)
#plt.show()

X = np.array([[0,2],
             [1.5,3],
             [2,0],
             [-1,1],
             [-1,-0.5],
             [1,-1]])

y = [0,0,0,1,1,1]
clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(X,y)
print("print SVM model CLF")
print(clf)

print(clf.predict([[-0.5,0]]))

w = clf.coef_[0]
print("this is w:")
print(w)

a = -w[0] / w[1]

xx = np.linspace(-2,3)
yy = a * xx - clf.intercept_[0] / w[1]

print("value of b:")
b_val = clf.intercept_
print(b_val)

b = clf.support_vectors_[0]
yy_up = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_down = a * xx + (b[1] - a * b[0])


plt.plot(xx, yy, 'k-', label="non weighted div")
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.scatter(X[:, 0], X[:, 1], c = y)
plt.legend()
plt.show()
