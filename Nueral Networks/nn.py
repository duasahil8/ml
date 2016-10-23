from __future__ import division
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time


np.set_printoptions(threshold=np.inf)

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W

start_time = time.time()

def sigmoid(z):

    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return 1/(1+np.exp(-z))


def derivative_sigmoid(z):
    return np.exp(-z)/((1+np.exp(-z))**2)

import random

def preprocess():
   mat = loadmat('mnist_all.mat')

   train_data = np.array([])
   train_l = []
   validation_data = np.array([])
   validation_l = []
   validation_label = np.array([])
   train_label = np.array([])
   test_data = np.array([])
   test_l = []
   test_label = np.array([])
   j = 0
   k = 0
   l = 0
   keys = mat.keys()
   train_data = np.array([]).astype('float').reshape(0,784)
   validation_data = np.array([]).astype('float').reshape(0,784)
   test_data = np.array([]).astype('float').reshape(0,784)
   for i in range(0,10):
       str_train = 'train' + str(i)

       train_i = mat.get(str_train)
       train_i_np = np.array(train_i)

       str_test = 'test' + str(i)
       test_i = mat.get(str_test)
       test_i_np = np.array(test_i)

       validation_i = train_i_np[-1000:,:]
       train_i_np = train_i_np[:-1000,:]

       train_data = np.vstack((train_data,train_i_np))
       validation_data = np.vstack((validation_data, validation_i))
       test_data = np.vstack((test_data,test_i_np))

       for p in range(train_i_np.shape[0]):
           train_l.append(i)
       for q in range(validation_i.shape[0]):
           validation_l.append(i)
       for r in range(test_i_np.shape[0]):
           test_l.append(i)

   validation_label = np.array(validation_l)
   train_label = np.array(train_l)
   test_label = np.array(test_l)

   train_data = np.divide(train_data,255)
   validation_data = np.divide(validation_data,255)
   test_data = np.divide(test_data,255)

   validation_label = np.reshape(validation_label,(validation_label.size,1))
   train_label = np.reshape(train_label,(train_label.size,1))
   test_label = np.reshape(test_label,(test_label.size,1))


   train_data = np.append(train_data,np.ones([len(train_data),1]),1)
   test_data  = np.append(test_data,np.ones([len(test_data),1]),1)
   validation_data = np.append(validation_data,np.ones([len(validation_data),1]),1)

   return train_data, train_label, validation_data, validation_label, test_data, test_label


def feed_forward(X,w1,w2):

    a2 = np.dot(X,np.transpose(w1))
    #a2 = np.reshape(a2,(a2.size,1))
    z2 = sigmoid(a2)
    bias = np.ones(X.shape[0])
    bias = np.reshape(bias,(bias.size,1))
    print bias.shape
    z2 = np.append(z2,bias,1)
    print 'z2' , z2.shape
    #z2 = np.reshape(z2,(z2.size,1))
    a3 = np.dot(z2,np.transpose(w2))
    y_out = sigmoid(a3)
    print y_out.shape
    return y_out,a2, z2 , a3



def nnObjFunction(params, *args):


    n_input, n_hidden, n_class, train_data, train_label, lambdaval = args
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    J_total = 0
    gradient_total = np.array([])

    n = train_data.shape[0]




    #w1[:w2.shape[1]-1] = 1
    #i = random.randint(0,49999)
    y_out,a2,z2,a3 = feed_forward(train_data,w1,w2)
    #print y_out.shape , a2.shape , z2.shape , z3.shape




    #true
    #add loop
    y_true_total = np.array([]).astype('float').reshape(0,10)

    y_true = (np.zeros(10))
    y_true = np.tile(y_true,n)
    y_true = np.reshape(y_true,(n,10))
    for i in range(n):
        index = train_label[i]
        y_true[i][int(index)] = 1
        #y_true_total = np.vstack((y_true_total,y_true))



    #error fn
    J = ((y_true-y_out)**2)
    J_scalar = 0.5*np.sum(J,1)
    J_total = np.sum(J_scalar)


    #backprop
    delta3 = np.array(np.multiply((y_true-y_out),y_out,(1-y_out)))
    gradient_w2 = -np.dot(np.transpose(delta3),z2)
    #w2_prime = np.array(w2[:,:-1])
    #z2 = z2[:-1,:]
    #remove after dot product
    delta2 = np.dot(delta3,w2)*(np.multiply((1-z2),z2))
    delta2 = delta2[:,:-1]
    #train_data_prime  = np.reshape(train_data,(train_data.size,1))
    gradient_w1 = -np.dot(np.transpose(delta2),train_data)


    # #############write to file
    #
    # file.write('\n\n======================================================\n\n')
    # file.write(str(i))
    # file.write('\nOutput -   ')
    # file.write(str(y_out))
    # file.write('\nTrue Vector  ')
    # file.write(str(y_true))
    # file.write('\nTrue Value  ')
    # file.write(str(train_label[i][0]))
    # file.write('\nPredicted Value  ')
    # file.write(str(np.argmax(y_out)))
    #
    # file.write('\n')
    #
    # if(np.argmax(y_out) == train_label[i][0]):
    #    file.write('Matched! :) \n')
    #    file.write('Confidence  \n')
    #    file.write(str(np.max(y_out)*100))
    # else:
    #    file.write('Not Matched :(')


    gradient_w1 = np.divide((gradient_w1 + lambdaval*w1),n)
    gradient_w2 = np.divide((gradient_w2 + lambdaval*w2),n)

    gradient_total = np.concatenate((gradient_w1.flatten(), gradient_w2.flatten()),0)
    J_total = J_total/n
    w1_s = np.sum(w1*w1)
    w2_s = np.sum(w2*w2)
    weight_reg = w1_s + w2_s
    J_total = J_total + weight_reg*(0.5)*lambdaval*(1/n)



    print 'J_total ' , J_total

    return (J_total,gradient_total)



def nnPredict(w1,w2,data):
    data = np.array(data)
    v = open('val.txt', 'w')
    y_out,a,b,c = feed_forward(data,w1,w2)
    labels = np.argmax(y_out,1)
    #true = validation_label[0]
    v.write('\n\n====================================\n\n')
    v.write(str(labels))

    return labels




"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

n_input = train_data.shape[1]-1;
n_hidden = 4;
n_class = 10;

initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

lambdaval = 0.1;

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

opts = {'maxiter' : 50}

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)


w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

f = open('finalwts.txt','w')
f.write(str(w1))
f.write(str(w2))

labels = nnPredict(w1,w2,validation_data)

print 'Running time  ' , (time.time()-start_time)/60 , ' mins'
#
#
#print 'Train Data' , train_data[343][734]
# print 'Train Label' , train_label.shape
# print 'Test Data '  , test_data.shape
# print 'Test Label' , test_label.shape
# print 'Validation Data ' , validation_data.shape
# print 'Validation Label' , validation_label.shape
#print 'W1 ' , initial_w1.shape
#print 'W2' , initial_w2.shape

#nnObjFunction(initial_w1,initial_w2)



#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
