import time
import pandas as pd
import numpy as np
import csv
import sys


L = len(sys.argv)

def read_data(st):
    with open(st) as csvfile:
        spamreader = csv.reader(csvfile)
        T = []
        for row in spamreader:
            l = [float(x) for x in row]
            T.append(l)
    return np.asarray(T)

if L == 1:
    tr_img = read_data('train_image.csv')
    tr_label = read_data('train_label.csv')
    test_img = read_data('test_image.csv')
else:
    tr_img = read_data(sys.argv[1])
    tr_label = read_data(sys.argv[2])
    test_img = read_data(sys.argv[3])
    print("used data")



N_training = tr_img.shape[0]
N_testing = test_img[0]



def predict_label(P):
    if P.shape[1] == 1:
        return (P > 0).astype(int)
    else:
        return np.argmax(P, axis=1).astype(int).reshape((P.shape[0], -1))


def layer_update(w,b, x, previous, lr):

        ### Update
        w = w - lr * np.matmul(np.transpose(x), previous) * (1 / x.shape[0])
        b = b - lr * np.mean(np.transpose(previous), axis=1)

        return w, b, np.matmul(previous, np.transpose(w))


def sigmoid_update(x, grad):
    ### Update
    L = 1 / (1 + np.exp(-x))
    L2 = 1 - L
    temp = np.multiply(L, L2)

    return np.multiply(grad, temp)

def weight_bias(input_size, output_size):
    return np.random.normal(0, 0.1, [input_size, output_size]), np.random.normal(0, 0.1, [1, output_size])

def Forwad(x, w1, b1, w2, b2, w3, b3):
    a1 = np.matmul(x, w1) + b1
    a2 = 1 / (1 + np.exp(-a1))
    a3 = np.matmul(a2, w2) + b2
    a4 = 1 / (1 + np.exp(-a3))
    a5 = np.matmul(a4, w3) + b3

    return a1, a2, a3, a4, a5

def Forwad_loss(X, Y):
    Y_10 = np.zeros([BS , 10])

    for i in range(BS):
        Y_10[i, int(Y[i]) ] = 1.0

    normilze = np.exp(X - np.amax(X, axis=1).reshape([BS, 1]))

    prob = normilze / (np.sum(normilze, axis=1).reshape([BS, 1]))
    loss = (1 / BS) * - np.sum(np.multiply(Y_10, np.log(prob)))

    return Y_10, prob, loss

def Backward(y, a5, a4, a3, a2, a1, BS, w1,b1,w2,b2,w3,b3, lr):


        expanded_data, prob,_ = Forwad_loss(a5, y)
        grad_a2 = - (1/BS) *  (expanded_data - prob)

        w3, b3, grad_d0 =  layer_update(w3, b3,a4, grad_a2, lr)
        grad_d1 = sigmoid_update(a3, grad_d0)

        w2, b2, grad_h1 =  layer_update(w2, b2,a2, grad_d1, lr)
        grad_a1 = sigmoid_update(a1, grad_h1)
        w1, b1, grad_x  = layer_update(w1, b1,x, grad_a1, lr)
        return w3, b3, w2, b2, w1, b1




################### Tuning paramters
num_epoch = 100
step = 10 ## tuning learning rate frequency
learning_rate = 10 ## Learning rate #5: 95 10: 97



layer_1_size = 1000
layer_2_size = 120 ### 120
layer_3_size = 10
d = 28 * 28
BS = 50
w1, b1 = weight_bias(d, layer_1_size)
w2, b2 = weight_bias(layer_1_size, layer_2_size)
w3, b3 = weight_bias(layer_2_size, layer_3_size)


a = time.perf_counter()

for t in range(num_epoch):

    if (t % step == 0) and (t != 0):
        learning_rate = learning_rate * 0.5


    for i in range(int(np.floor(N_training/BS))):

        x = tr_img[(i * BS): (i + 1) * BS]
        y = tr_label[(i * BS): (i + 1) * BS]
        a1, a2, a3, a4, a5 = Forwad(x, w1, b1, w2, b2, w3, b3)
        w3, b3, w2, b2, w1, b1 = Backward(y, a5, a4, a3, a2, a1, BS, w1, b1, w2, b2, w3, b3, learning_rate)

x = test_img
_, _, _, _, L = Forwad(x, w1, b1, w2, b2, w3, b3)
Z = predict_label(L)
pd.DataFrame(Z).to_csv("./test_predictions.csv",header=None, index=None)

print(time.perf_counter() - a)




