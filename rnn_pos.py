#!/usr/bin/env python
from numpy import zeros, ones, dot, exp, mat, array

class RNN:
    n_inputs = None
    n_outputs = None
    n_hidden = None
    
    W = None # row matrix of output layer weights
    V = None # row matrix of input layer weight
    U = None # row matrix of hidden layer weights

    def __init__(self, n_inputs = 100, n_outputs = 10, n_hidden = 40):
        self.n_inputs, self.n_outputs, self.n_hidden = n_inputs, n_outputs, n_hidden
        
        self.U = zeros((self.n_hidden, self.n_hidden))
        self.V = zeros((self.n_hidden, self.n_inputs))
        self.W = zeros((self.n_outputs, self.n_hidden))
        self.Theta_j = zeros(self.n_hidden)
        self.Theta_k = zeros(self.n_outputs)
        self.y_p = zeros(self.n_hidden)

    def g(self, x):
        return 1/(1 + exp(-x))

    def g_prime(self, x):
        return x*(1 - x)

    f = g
    f_prime = g_prime

    def reset_hidden(self):
        self.y_p = zeros(self.n_hidden)
        self.y_pp = zeros(self.n_hidden)
        

    def feed(self, x):
        self.y_pp = self.y_p
        self.y_p = self.f(dot(self.V, x) + dot(self.U, self.y_p) + self.Theta_j)
        y_k = self.g(dot(self.W, self.y_p) + self.Theta_k)
        return y_k

    def train(self, train_set, ni = 0.1):
        dW = zeros((self.n_outputs, self.n_hidden))
        dV = zeros((self.n_hidden, self.n_inputs))
        dU = zeros((self.n_hidden, self.n_hidden))
        self.reset_hidden()
        for x, d in train_set:
            y = self.feed(x)
            d_k = (d - y) * self.g_prime(y)
            y_j = self.y_p
            d_j = dot(d_k, self.W) * self.f_prime(self.y_p)
            y_pp = self.y_pp
            
            dW += ni * dot(mat(d_k).transpose(), mat(y_j))
            dV += ni * dot(mat(d_j).transpose(), mat(x))
            dU += ni * dot(mat(d_j).transpose(), mat(y_pp))

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

    rnn = RNN(5, 1, 40)
    for i in range(5000):
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
