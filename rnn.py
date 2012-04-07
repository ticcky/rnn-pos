#!/usr/bin/env python
from numpy import zeros, ones, dot, exp, mat, array, random

class RNN:
    n_inputs = None
    n_outputs = None
    n_hidden = None
    
    W = None # row matrix of output layer weights
    V = None # row matrix of input layer weight
    U = None # row matrix of hidden layer weights

    def __init__(self, n_inputs = 100, n_outputs = 10, n_hidden = 40, p_time = 3):
        self.n_inputs, self.n_outputs, self.n_hidden = n_inputs, n_outputs, n_hidden
        self.p_time = 3
        
        self.U = random.random((self.n_hidden, self.n_hidden))
        self.V = random.random((self.n_hidden, self.n_inputs))
        self.W = random.random((self.n_outputs, self.n_hidden))
        self.Theta_j = zeros(self.n_hidden)
        self.Theta_k = zeros(self.n_outputs)
        self.y_p = zeros(self.n_hidden)
        self.x_p = zeros(self.n_inputs)
        self.x_pp = [zeros(self.n_inputs) for n in range(p_time)]
        self.y_pp = [zeros(self.n_hidden) for n in range(p_time)]

    def g(self, x):
        return 1/(1 + exp(-x))

    def g_prime(self, x):
        return x*(1 - x)

    f = g
    f_prime = g_prime

    def reset_hidden(self):
        self.y_p = zeros(self.n_hidden)
        self.y_pp = [zeros(self.n_hidden) for n in range(self.p_time)] #zeros(self.n_hidden)
        

    def feed(self, x):
        self.y_pp.insert(0, self.y_p)
        self.y_pp = self.y_pp[:self.p_time + 1]
        self.x_pp.insert(0, self.x_p)
        self.x_pp = self.x_pp[:self.p_time]
        #self.y_pp = self.y_p
        self.x_p = x
        self.y_p = self.f(dot(self.V, x) + dot(self.U, self.y_p) + self.Theta_j)
        y_k = self.g(dot(self.W, self.y_p) + self.Theta_k)
        return y_k

    def train(self, train_set, ni = 0.1):
        dW = zeros((self.n_outputs, self.n_hidden))
        dV = zeros((self.n_hidden, self.n_inputs))
        dU = zeros((self.n_hidden, self.n_hidden))
        self.reset_hidden()
        cntr = 0
        for x, d in train_set:
            cntr += 1
            if cntr % 100 == 0:
                print cntr
            if type(x) == int:
                x_v = zeros(self.n_inputs)
                d_v = zeros(self.n_outputs)
                x_v[x] = 1.0
                d_v[d] = 1.0
            else:
                x_v = x
                d_v = d
            
            y = self.feed(x_v)
            d_k = (d_v - y) * self.g_prime(y)
            y_j = self.y_p
            d_j = dot(d_k, self.W) * self.f_prime(self.y_p)
            y_pp = self.y_pp[0]
            
            dW += ni * dot(mat(d_k).transpose(), mat(y_j))
            dV += ni * dot(mat(d_j).transpose(), mat(x_v))
            dU += ni * dot(mat(d_j).transpose(), mat(y_pp))

            for i in range(self.p_time):
                d_j = dot(d_j, self.U)*self.f_prime(self.y_pp[i])
                dV += ni * dot(mat(d_j).transpose(), mat(self.x_pp[i]))
                dU += ni * dot(mat(d_j).transpose(), mat(self.y_pp[i + 1]))
                

        self.W += dW
        self.V += dV
        self.U += dU

if __name__ == "__main__":
    train_data =         [
            (array([1, 0, 0 , 0, 0]), array([1])),
            (array([0, 1, 0 , 0, 0]), array([1])),
            (array([1, 0, 1 , 0, 0]), array([1])),
            (array([1, 0, 0 , 1, 0]), array([1])),
            (array([1, 0, 0 , 0, 0]), array([1])),
            (array([0, 1, 0 , 0, 0]), array([1])),
            (array([1, 0, 1 , 0, 0]), array([1])),
            (array([1, 0, 0 , 1, 0]), array([1])),
            (array([1, 0, 0 , 0, 0]), array([1])),
            (array([0, 1, 0 , 0, 0]), array([1])),
            (array([1, 0, 1 , 0, 0]), array([1])),
            (array([1, 0, 0 , 1, 0]), array([1])),
            (array([1, 0, 0 , 0, 0]), array([1])),
            (array([0, 1, 0 , 0, 0]), array([1])),
            (array([1, 0, 1 , 0, 0]), array([1])),
            (array([1, 0, 0 , 1, 0]), array([1])),
            (array([1, 0, 0 , 0, 0]), array([1])),
            (array([0, 1, 0 , 0, 0]), array([1])),
            (array([1, 0, 1 , 0, 0]), array([1])),
            (array([1, 0, 0 , 1, 0]), array([1])),
            (array([1, 0, 0 , 0, 0]), array([1])),
            (array([0, 1, 0 , 0, 0]), array([1])),
            (array([1, 0, 1 , 0, 0]), array([1])),
            (array([1, 0, 0 , 1, 0]), array([1])),
            (array([1, 0, 0 , 0, 0]), array([1])),
            (array([0, 1, 0 , 0, 0]), array([1])),
            (array([1, 0, 1 , 0, 0]), array([1])),
            (array([1, 0, 0 , 1, 0]), array([1])),
            (array([1, 0, 1 , 0, 1]), array([0])),
            (array([1, 0, 0 , 0, 0]), array([0])),
            (array([1, 1, 1 , 0, 0]), array([0])),
            (array([1, 0, 0 , 1, 1]), array([0])),
            (array([1, 0, 0 , 0, 0]), array([0])),
            (array([1, 0, 1 , 0, 1]), array([0])),
        ]

    rnn = RNN(5, 1, 2)
    for i in range(1000):
        rnn.train(train_data)
    rnn.reset_hidden()
    print rnn.feed(array([1, 0, 0, 0, 0]))
    print rnn.feed(array([0, 1, 0, 0, 0]))
    print rnn.feed(array([0, 0, 1, 0, 0]))
    print rnn.feed(array([0, 0, 0, 1, 0]))
    rnn.reset_hidden()
    print rnn.feed(array([0, 0, 1, 0, 0]))
    rnn.reset_hidden()
    print rnn.feed(array([1, 0, 1, 0, 1]))
    rnn.reset_hidden()
    print rnn.feed(array([1, 1, 1, 1, 1]))
