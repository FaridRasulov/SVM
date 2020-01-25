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

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,offset=16).reshape(len(labels), 784)
    return images, labels

def get_Wk(grad, y_train, L_, sel_q, C, tol, Wk_prev=None):

    grad_y = (- grad / y_train)[:,0]

    Lp = ((L_ < tol) & (y_train == 1))
    Lm = ((L_ < tol) & (y_train == -1))
    Up = ((L_ > C-(tol)) & (y_train == 1))
    Um = ((L_ > C-(tol)) & (y_train == -1))

    R_indexes = np.where(Lp|Um|((L_ > tol) & (L_ < C-tol)))[0]
    S_indexes = np.where(Lm|Up|((L_ > tol) & (L_ < C-tol)))[0]
    
    compl = np.concatenate([S_indexes,Wk_prev])
    np.put(grad_y, compl.astype(int), [-np.inf]*len(compl))
    I = np.argpartition(grad_y, -sel_q)[-sel_q:]
    m = np.around(np.max(grad_y[I]), decimals=3)

    grad_y = (- grad / y_train)[:,0]
    
    compl = np.concatenate([R_indexes,Wk_prev])
    np.put(grad_y, compl.astype(int), [np.inf]*len(compl))
    J = np.argpartition(grad_y, sel_q)[:sel_q]
    M = np.around(np.min(grad_y[J]), decimals=3)

    Wk = np.append(I, J)
    Wc = np.setdiff1d(np.concatenate([R_indexes , S_indexes]), Wk)
    mMdiff = m - M
    return Wk, Wc, mMdiff

def analytic_solver(Wk, y_train, L_Wk, grad, QWk, C):
    i = Wk[0]
    j = Wk[1]
    I = 0
    J = 1

    d = np.array([y_train[i,0],-y_train[j,0]]).reshape(2, 1)
    
    if (d[I] > 0) and (d[J] > 0): beta_max = min(C - L_Wk[I], C - L_Wk[J])
    elif (d[I] < 0) and (d[J] < 0): beta_max = min(L_Wk[I], L_Wk[J])
    elif (d[I] > 0) and (d[J] < 0): beta_max = min(C - L_Wk[I], L_Wk[J])
    elif (d[I] < 0) and (d[J] > 0): beta_max = min(L_Wk[I], C - L_Wk[J])
    
    grad_d = np.matmul(np.array([grad[i],grad[j]]).T, d)
    dsQds = QWk[I][I]+QWk[J][J]+2*QWk[I][J]*d[I]*d[J]
    
    if dsQds == 0:
        beta_star = beta_max 
    else:
        beta_nv = -grad_d/dsQds
        beta_star = np.min([beta_max[0],beta_nv[0]])

    lambda_star = L_Wk + beta_star * d
    return lambda_star

def objective_function_mat_Q(params, Q):
    L_ = params.reshape(-1, 1)
    L_Q = np.matmul(L_.T, Q)
    L_Q_L = np.matmul(L_Q, L_)
    objective_value = (1 / 2) * L_Q_L - L_.sum()
    return objective_value[0][0]
    
def fit_mvp(X_train, y_train, kernel, C, gamma, r, maxit, tol, L_):
    sel_q = int(r / 2)
    K_mat = rbf_kernel(X_train, gamma = gamma)
    Q = np.multiply(np.multiply(K_mat, y_train).T, y_train).T
        
    grad = np.ones(Q.shape[0])*-1.
    L_old = np.zeros(y_train.shape[0]).reshape(-1, 1)
    Wk_prev = np.array([])
    
    t = time.time()
    for i in range(maxit):
        
        Wk, Wc, mMdiff = get_Wk(grad, y_train, L_, sel_q, C, tol, Wk_prev)
        Wk_prev = np.concatenate([Wk_prev,Wk])
        
        if mMdiff < tol : break
        
        QWk = Q[Wk,:][:,Wk]
        L_Wk = L_[Wk,:]
        L_Wc = L_[Wc,:]
        QcWk = Q[Wc,:][:,Wk]
        
        y_train_Wk = y_train[Wk, 0]
        y_train_Wc = y_train[Wc, 0]     
        
        sol = analytic_solver(Wk, y_train, L_Wk, grad, QWk, C)
        L_Wk_star = np.around(sol.reshape(-1,1), decimals=6)
        
        np.put(L_, Wk, L_Wk_star)
        grad = grad + np.matmul(Q,L_ - L_old)
        np.put(L_old, Wk, L_Wk_star)
    tr_time = time.time() - t
    
    obj = objective_function_mat_Q(L_, Q)
    NI = i
    FE = 0
    b = (1/y_train)-np.matmul(np.multiply(L_, y_train).T,K_mat).T
    return L_, np.mean(b), NI, FE, mMdiff, obj, tr_time

def predict(b, l, X_train, X_test, y_train, kernel, kernel_param):
    if kernel == 'poly':
        z = np.matmul(np.multiply(l, y_train).T,polynomial_kernel(X_train, X_test, degree=kernel_param)) + b
    else:
        z = np.matmul(np.multiply(l, y_train).T,rbf_kernel(X_train, Y=X_test, gamma=kernel_param)) + b
    return np.sign(z).astype(int)

def writetxt(file, C, gamma, kernel, method, obj, conf_mat, test_acc, train_acc, NI, FE, mMdiff, r, tr_time):
    print('\nSaving results in output_question_3.txt')
    output = open(file,"w")
    output.write("This is homework 2: question 3")
    output.write("\nValue of C: " + "%i" % C)
    output.write("\nValue of Gamma: " + "%.3f" % gamma)
    output.write("\nKernel: " + kernel)
    #output.write("\nMethod:" + method)
    #output.write("\nDual objective: " + "%.2f" % obj)
    output.write("\nClassification rate on the test set: " + "%.2f" % test_acc)
    output.write("\nClassification rate on the training set: " + "%.2f" % train_acc)
    output.write("\nConfusion Matrix 1st row: " + str(conf_mat[0]))
    output.write("\nConfusion Matrix 2nd row: " + str(conf_mat[1]))
    output.write("\nNumber of iterations: " + "%i" % NI)
    #output.write("\nNumber of Function Evaluations: " + "%i" % FE)
    output.write("\n m - M : " + "%.5f" % mMdiff)
    output.write("\n q : " + "%i" % r)
    output.write("\nTraining time: " + "%.2f" % tr_time + "sec")
    output.close()