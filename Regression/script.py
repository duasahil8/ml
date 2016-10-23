import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

np.set_printoptions(threshold=np.inf)


def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix

    # IMPLEMENT THIS METHOD

    X = np.array(X)
    y = np.array(y)
    y = np.reshape(y,y.size)
    classes = np.unique(y)

    i = 0
    for c in classes:
        X_c = X[y==c]
        mean_c = np.array(X_c.mean(0))
        mean_c = np.reshape(mean_c,(mean_c.size,1))
        if i==0:
            mean_t = mean_c
        else:
            mean_t = np.hstack((mean_t,mean_c))
        i = i+1

    covmat = np.cov(np.transpose(X))
    #print mean_t.shape , covmat.shape

    return mean_t, covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    # IMPLEMENT THIS METHOD
    X = np.array(X)
    y = np.array(y)
    y = np.reshape(y,y.size)
    classes = np.unique(y)
    covmats = []

    i = 0
    for c in classes:
        X_c = X[y==c]
        covmat_c = np.cov(np.transpose(X_c))
        covmats.append(covmat_c)
        #print covmats
        mean_c = np.array(X_c.mean(0))
        mean_c = np.reshape(mean_c,(mean_c.size,1))
        if i==0:
            means = mean_c
        else:
            means = np.hstack((means,mean_c))

        i = i+1

    #print 'qda ' , means.shape , covmat.shape
    return means,covmats



def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value

    match_count = 0
    y_pred = np.array([])
    p_y = 1/5.0
    p_y_x = np.array([])
    d = Xtest.shape[1]
    pi_term = (2*np.pi)**(d/2)
    #classes = np.unique(ytest)
    #num_classes = int(np.unique(ytest).size)
    num_classes = means.shape[1]
    data_size = int(Xtest.shape[0])
    for x_i in range(data_size):
        for i in range(num_classes):
            a = Xtest[x_i,:] - means[:,i]
            b = np.dot(np.transpose(a),inv(covmat))
            mahalanobis = np.dot(b,a)
            mult_const = (p_y)/(pi_term*np.sqrt(det(covmat)))
            p_y_x = np.append(p_y_x,mult_const*np.exp(-0.5*mahalanobis))

        true = ytest[x_i]
        predicted = np.argmax(p_y_x)+1
        y_pred = np.append(y_pred,predicted)
        if (true==predicted):
            match_count += 1;

        p_y_x = np.array([])
    acc = (float(match_count)/float(data_size))*100.0



    return acc , y_pred





def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    y_pred = np.array([])
    match_count = 0
    p_y = 1/5.0
    d = Xtest.shape[1]
    p_y_x = np.array([])
    pi_term = (2*np.pi)**(d/2)
    #classes = np.unique(ytest)
    classes = means.shape[1]

    for x_i in range(Xtest.shape[0]):
        for i in range(classes):
            a = Xtest[x_i,:] - means[:,i]
            b = np.dot(np.transpose(a),inv(covmats[i]))
            mahalanobis = np.dot(b,a)
            mult_const = (p_y)/(pi_term*np.sqrt(det(covmats[i])))
            p_y_x = np.append(p_y_x,mult_const*np.exp(-0.5*mahalanobis))
        true = ytest[x_i]
        predicted = np.argmax(p_y_x)+1
        y_pred = np.append(y_pred,predicted)
        if (true==predicted):
            match_count += 1;

        p_y_x = np.array([])

    acc = (match_count/float(Xtest.shape[0]))*100.0

    return acc , y_pred

def learnOLERegression(X,y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1
    # IMPLEMENT THIS METHOD
    xx_t = np.dot(np.transpose(X),X)
    w = np.dot(inv(xx_t), np.dot(np.transpose(X),y))
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d
    # y = N x 1
    # lambd = ridge parameter (scalar)
    # Output:
    # w = d x 1

    # IMPLEMENT THIS METHOD
    I = np.identity(X.shape[1])
    xt_x = np.dot(np.transpose(X),X)
    reg_term = lambd*I*X.shape[0]
    inv_term = inv(xt_x+reg_term)
    w = np.dot(inv_term,np.dot(np.transpose(X),y))
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse

    # IMPLEMENT THIS METHOD


    a = ytest - np.dot(Xtest,w)
    a_sq = np.dot(np.transpose(a),a)
    sme = (a_sq)/Xtest.shape[0]
    rmse = np.sqrt(sme)
    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda

    # IMPLEMENT THIS METHOD

    #X_t(Xw - y) + lambd*w
    y = np.array(y)
    w = np.reshape(w,(w.size,1))
    N = X.shape[0]
    a = y-np.dot(X,w)
    a_sq = np.dot(np.transpose(a),a)
    w_sq = np.dot(np.transpose(w),w)

    error = a_sq/(2*N) + lambd*w_sq/2
    error_grad = np.dot(np.transpose(X),-a)/N + lambd*w


    error = error.flatten()
    error_grad = error_grad.flatten()
    return  error , error_grad

def mapNonLinear(x,p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xd - (N x (d+1))
    # IMPLEMENT THIS METHOD

    N = x.size
    Xd = np.empty((N,p+1))
    for i in range(p+1):
        Xd[:,i] = np.power(x,i, out=None)

    return Xd

# Main script

# Problem 1
# load the sample data
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc , y_pred_lda = ldaTest(means,covmat,Xtest,ytest)

print('LDA Accuracy = '+str(ldaacc))

# QDA
means,covmats = qdaLearn(X,y)
qdaacc, y_pred_qda = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))

plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.show()

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

# Problem 2

if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses3)

# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    rmses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses4)


# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
