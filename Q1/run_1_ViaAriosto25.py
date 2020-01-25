import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
from scipy import stats
import time
from sklearn.model_selection import KFold
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from sklearn.metrics import confusion_matrix
import numpy as np

import func_1_ViaAriosto25 as f

np.random.seed(1870543)

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

y_train = np.array([1 if i == 4 else -1 for i in y_train])
y_test = np.array([1 if i == 4 else -1 for i in y_test])

y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1)) 

scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# C = [1,5,10,25,50]
# Gamma = [0.015,0.1,1,2,5]
# kernel = ['gauss','poly']

# kf = KFold(n_splits=3)
# kf.get_n_splits(X_train)
# grid_dict={}

# for k in kernel:
#     for c in C:
#         for g in Gamma:
#             val_score = []
#             for train_index, val_index in kf.split(X_train):
#                 X_tn, X_val = X_train[train_index], X_train[val_index]
#                 y_tn, y_val = y_train[train_index], y_train[val_index]
#                 lambda_star, b, NI, obj, tr_time = f.fit(X_tn,y_tn,k,c,g)
#                 y_train_val = f.predict(b,lambda_star, X_tn,X_tn,y_tn,k,g).T
#                 y_test_val = f.predict(b,lambda_star, X_tn,X_val,y_tn,k,g).T

#                 val_acc = accuracy_score(y_val,y_test_val)
#                 val_score.append(val_acc)
#             grid_dict[np.mean(val_score)]=(c,g,k)

# val_acc = max(grid_dict, key=float)

C = 5 #grid_dict[val_acc][0]
gamma = 0.015 #grid_dict[val_acc][1]
kernel = 'gauss' #grid_dict[val_acc][2]

t = time.time()
lambda_star, b, NI, obj, tr_time = f.fit(X_train,y_train,kernel,C,gamma)
trainingComputingTime = time.time() - t

y_train_pred = f.predict(b,lambda_star, X_train,X_train,y_train,kernel,gamma).T
y_test_pred = f.predict(b,lambda_star, X_train,X_test,y_train,kernel,gamma).T

conf_mat = confusion_matrix(y_test,y_test_pred)
val_acc = 0.87 #np.round(max(grid_dict, key=float), 2)
train_acc = np.round(accuracy_score(y_train,y_train_pred), 2)
test_acc = np.round(accuracy_score(y_test,y_test_pred), 2)

f.writetxt('output_question_1.txt', C, gamma, kernel, 'default', obj, conf_mat, val_acc, test_acc, train_acc, NI, tr_time)