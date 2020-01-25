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

def load_mnist(path, kind='train'): 
    import os
    import gzip
    import numpy as np

    """Load MNIST data from path""" 
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind) 
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind) 
    with gzip.open(labels_path, 'rb') as lbpath: 
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8) 
    with gzip.open(images_path, 'rb') as imgpath: 
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,offset=16).reshape(len(labels), 784) 
    return images, labels 

def fit(X_train, y_train, kernel, C, kernel_param): 
    if kernel == 'gauss': 
        K_mat = rbf_kernel(X_train, Y = None, gamma = kernel_param) 
    elif kernel == 'poly': 
        K_mat = polynomial_kernel(X_train, Y = None, degree = kernel_param) 
     
    m,n = X_train.shape 
    y_train = y_train.reshape(-1,1) * 1. 
    Q = np.multiply(np.multiply(K_mat, y_train).T, y_train).T 
     
    P = cvxopt_matrix(Q) 
    q = cvxopt_matrix(-np.ones((m, 1))) 
    G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m)))) 
    h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C))) 
    A = cvxopt_matrix(y_train.reshape(1, -1)) 
    b = cvxopt_matrix(np.zeros(1))
    
    cvxopt_solvers.options['show_progress'] = False
    t = time.time()
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    tr_time = time.time()-t
    NI = sol['iterations']
    lambda_star = np.array(sol['x']).reshape(-1, 1) 
    lambda_star = np.around(lambda_star, decimals=4) 
    b = (1/y_train)-np.matmul(np.multiply(lambda_star, y_train).T,K_mat).T 
    return lambda_star, np.mean(b), NI, tr_time

def predict(b, l, X_train, X_test, y_train, kernel, kernel_param): 
    if kernel == 'poly': 
        z = np.matmul(np.multiply(l, y_train).T,polynomial_kernel(X_train, X_test, degree=kernel_param)) + b 
    else: 
        z = np.matmul(np.multiply(l, y_train).T,rbf_kernel(X_train, X_test, gamma=kernel_param)) + b
    return z[0]

def writetxt(file, C, gamma, kernel, method, conf_mat, test_acc, train_acc, NI, tr_time):
    print('\nSaving results in output_question_bonus.txt')
    output = open(file,"w")
    output.write("This is homework 2: question bonus")
    output.write("\nValue of C: " + "%i" % C)
    output.write("\nValue of Gamma: " + "%.3f" % gamma)
    output.write("\nKernel: " + kernel)
    #output.write("\nMethod:" + method)
    output.write("\nClassification rate on the test set: " + "%.2f" % test_acc)
    output.write("\nClassification rate on the training set: " + "%.2f" % train_acc)
    output.write("\nConfusion Matrix 1st row: " + str(conf_mat[0]))
    output.write("\nConfusion Matrix 2nd row: " + str(conf_mat[1]))
    output.write("\nNumber of iterations: " + "%i" % NI)
    #output.write("\nNumber of Function Evaluations: " + "%i" % NI)
    output.write("\nTraining time: " + "%.2f" % tr_time + "sec")
    output.close()