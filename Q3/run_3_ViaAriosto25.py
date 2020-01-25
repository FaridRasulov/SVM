import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
from scipy.optimize import minimize
import csv
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
from scipy import stats
import time
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import numpy as np

import func_3_ViaAriosto25 as f

X_all_labels, y_all_labels = f.load_mnist('Data', kind='train')

indexLabel2 = np.where((y_all_labels==2))
xLabel2 =  X_all_labels[indexLabel2][:1000,:].astype('float64') 
yLabel2 = y_all_labels[indexLabel2][:1000].astype('float64') 

indexLabel4 = np.where((y_all_labels==4))
xLabel4 =  X_all_labels[indexLabel4][:1000,:].astype('float64') 
yLabel4 = y_all_labels[indexLabel4][:1000].astype('float64') 

indexLabel6 = np.where((y_all_labels==6))
xLabel6 =  X_all_labels[indexLabel6][:1000,:].astype('float64')
yLabel6 = y_all_labels[indexLabel6][:1000].astype('float64') 


X = np.concatenate([xLabel4,xLabel2])
y = np.concatenate([yLabel4,yLabel2])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1870543)

y_train = np.array([1 if i == 4 else -1 for i in y_train])
y_test = np.array([1 if i == 4 else -1 for i in y_test])

y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

q = 2
C = 5
maxit = 10000
tol = 10e-8
gamma = 0.015
kernel = 'gauss'
L_ = np.zeros(y_train.shape[0]).reshape(-1,1)

L_, b, NI, FE, mMdiff, obj, tr_time = f.fit_mvp(X_train, y_train, kernel, C, gamma, q, maxit, tol,L_)

y_train_pred = f.predict(b, L_, X_train, X_train, y_train, kernel, gamma).T
y_test_pred = f.predict(b, L_, X_train, X_test, y_train, kernel, gamma).T

conf_mat = confusion_matrix(y_test, y_test_pred)
train_acc = np.round(accuracy_score(y_train,y_train_pred),2)
test_acc = np.round(accuracy_score(y_test, y_test_pred),2)

f.writetxt('output_question_3.txt', C, gamma, kernel, 'default', obj, conf_mat, test_acc, train_acc, NI, FE, mMdiff, q, tr_time)