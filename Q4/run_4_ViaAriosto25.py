import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler 
import numpy as np 
from sklearn.metrics import accuracy_score 
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel,euclidean_distances 
from scipy import stats 
import time
from cvxopt import matrix as cvxopt_matrix 
from cvxopt import solvers as cvxopt_solvers 
from sklearn.metrics import confusion_matrix
import numpy as np 

import func_4_ViaAriosto25 as f

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

X = np.concatenate([xLabel2,xLabel4,xLabel6]) 
y = np.concatenate([yLabel2,yLabel4,yLabel4])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1870543)

y_train2vsall = np.array([1 if i == 2 else -1 for i in y_train]) 
y_test2vsall = np.array([1 if i == 2 else -1 for i in y_test]) 
y_train2vsall = y_train2vsall.reshape((-1, 1)) 
y_test2vsall = y_test2vsall.reshape((-1, 1))  

y_train4vsall = np.array([1 if i == 4 else -1 for i in y_train]) 
y_test4vsall = np.array([1 if i == 4 else -1 for i in y_test]) 
y_train4vsall = y_train4vsall.reshape((-1, 1)) 
y_test4vsall = y_test4vsall.reshape((-1, 1))  

y_train6vsall = np.array([1 if i == 6 else -1 for i in y_train]) 
y_test6vsall = np.array([1 if i == 6 else -1 for i in y_test]) 
y_train6vsall = y_train6vsall.reshape((-1, 1)) 
y_test6vsall = y_test6vsall.reshape((-1, 1))  

scaler = MinMaxScaler(feature_range=(0, 1)) 
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test) 


C = 3
gamma = 0.015
kernel = 'gauss' 

lambda_star2vsall, b, NI2, tr_time2 = f.fit(X_train,y_train2vsall,kernel,C,gamma) 
y_train_pred2vsall = f.predict(b,lambda_star2vsall, X_train,X_train,y_train2vsall,kernel,gamma) 
y_test_pred2vsall = f.predict(b,lambda_star2vsall, X_train,X_test,y_train2vsall,kernel,gamma) 

lambda_star4vsall, b, NI4, tr_time4 = f.fit(X_train,y_train4vsall,kernel,C,gamma) 
y_train_pred4vsall = f.predict(b,lambda_star4vsall, X_train,X_train,y_train4vsall,kernel,gamma) 
y_test_pred4vsall = f.predict(b,lambda_star4vsall, X_train,X_test,y_train4vsall,kernel,gamma) 

lambda_star6vsall, b, NI6, tr_time6 = f.fit(X_train,y_train6vsall,kernel,C,gamma)
y_train_pred6vsall = f.predict(b,lambda_star6vsall, X_train,X_train,y_train6vsall,kernel,gamma) 
y_test_pred6vsall= f.predict(b,lambda_star6vsall, X_train,X_test,y_train6vsall,kernel,gamma) 

finalResulttrain = np.array([]) 
finalResulttest = np.array([]) 

NI = NI2+NI4+NI6
tr_time = tr_time2 + tr_time4 + tr_time6

for i in range(0,len(y_train_pred2vsall)): 
    if(y_train_pred2vsall[i]>=y_train_pred4vsall[i] and y_train_pred2vsall[i]>=y_train_pred6vsall[i]): 
        finalResulttrain=np.append(finalResulttrain,2)
    elif(y_train_pred4vsall[i]>=y_train_pred6vsall[i]): 
        finalResulttrain=np.append(finalResulttrain,4)
    else: 
        finalResulttrain=np.append(finalResulttrain,6) 

for i in range(0,len(y_test_pred2vsall)): 
    if(y_test_pred2vsall[i]>=y_test_pred4vsall[i] and y_test_pred2vsall[i]>=y_train_pred6vsall[i]): 
        finalResulttest=np.append(finalResulttest,2) 
    elif(y_test_pred4vsall[i]>=y_train_pred6vsall[i]): 
        finalResulttest=np.append(finalResulttest,4) 
    else: 
        finalResulttest=np.append(finalResulttest,6) 

conf_mat = confusion_matrix(finalResulttest,y_test)
train_acc = np.round(accuracy_score(finalResulttrain,y_train),2)
test_acc = np.round(accuracy_score(finalResulttest,y_test),2)

f.writetxt('output_question_4.txt', C, gamma, kernel, 'default', conf_mat, test_acc, train_acc, NI, tr_time)