#!/usr/bin/env python
from numpy import zeros, ones, dot, exp

class RNN:
    n_inputs = 100
    n_outputs = 10
    n_hidden = 40
    
    W = None # row matrix of output layer weights
    V = None # row matrix of input layer weight
    U = None # row matrix of hidden layer weights

    def __init__(self):
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

    def feed(self, x):
        self.y_p = self.f(dot(self.V, x) + dot(self.U, self.y_p) + self.Theta_j)
        y_k = self.g(dot(self.W, self.y_p) + self.Theta_k)
        return y_k
        

if __name__ == "__main__":
    rnn = RNN()
    print rnn.feed(ones(rnn.n_inputs))
