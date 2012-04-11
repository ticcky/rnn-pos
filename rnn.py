#!/usr/bin/env python
from numpy import zeros, ones, dot, exp, mat, array, random, abs, multiply

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
        self.y_p = mat(zeros(self.n_hidden))
        self.x_p = mat(zeros(self.n_inputs))
        self.x_pp = [mat(zeros(self.n_inputs)) for n in range(p_time)]
        self.y_pp = [mat(zeros(self.n_hidden)) for n in range(p_time)]

    def f(self, x):
        return 1/(1 + exp(-x))

    def f_prime(self, x):
        return multiply(x, (1 - x))

    def get_hidden_state_matrix(self):
        return mat(zeros((self.n_hidden, 1)))
    
    def g(self, x):
        t = exp(x)
        denom = t.sum()
        return t / denom #exp(x) #/ denom

    def g_primex(self, x):
        denom = sum([exp(i) for i in x])
        return array([1 - sum([a/denom for a in x if a != i]) for i in x])
    
    def feed(self, x, y_p):
        y_p = self.f(dot(self.V, x) + dot(self.U, y_p)) # + dot(self.y_p, self.U))
        y_k = self.g(dot(self.W, y_p))

        return y_k, y_p #.transpose()

    def train(self, train_set, ni = 0.1):
        W = self.W
        Wt = self.W.transpose()
        # init delta weights
        dW = zeros((self.n_outputs, self.n_hidden))
        dV = zeros((self.n_hidden, self.n_inputs))
        dU = zeros((self.n_hidden, self.n_hidden))
        
        cntr = 0
        x_v = mat(zeros((self.n_inputs, 1)))
        d_v = mat(zeros((self.n_outputs, 1)))
        y_p = self.get_hidden_state_matrix()
        def zero(vector):
            for i in range(len(vector)):
                vector[i] = 0.0

        # go through the training set and accumulate the weight changes
        for x, d in train_set:
            cntr += 1
            if cntr % 100 == 0:
                print cntr

            # reset hidden layer (e.g. to be able to account for the sentence end)
            if x is None:
                y_p = self.get_hidden_state_matrix()
                continue

            # prepare the input and output vectors
            if type(x) == int:
                # we don't have to zero the vector here, because we always zero it when finishing the iteration
                #zero(x_v)
                #zero(d_v)
                x_v[x,0] = 1.0
                d_v[d,0] = 1.0
            else:
                x_v = x
                d_v = d

            # get the network's output for the current input, and it's hidden layer output
            y_k, y_j = self.feed(x_v, y_p)
            #y_k, y_j = mat(zeros(self.n_outputs)).transpose(), mat(zeros(self.n_hidden))
            # y_k - column vector
            # y_j - row vector

            # get gradient for function g (sotfmax)
            d_k = (d_v - y_k)  
            # d_k - column vector

            # get gradient for function f (sigmoid)          
            d_j = multiply(self.f_prime(y_j),  Wt * d_k) #multiply(dot(d_k.transpose(), self.W), self.f_prime(y_j))

            dW += d_k * y_j.transpose()
            #dV += dot(d_j.transpose(), x_v)
            dV[:,x] = d_j.transpose()
            dU += d_j * y_p.transpose()

            y_p = y_j

            # clean up
            x_v[x,0] = 0.0
            d_v[d,0] = 0.0

            # BPTT
            #for i in range(self.p_time):
            #    d_j = dot(d_j, self.U)*self.f_prime(self.y_pp[i])
            #    dV += ni * dot(mat(d_j).transpose(), mat(self.x_pp[i]))                
            #    dU += ni * dot(mat(d_j).transpose(), mat(self.y_pp[i + 1]))
                
            #import pdb; pdb.set_trace()
        self.W += ni * dW
        self.V += ni * dV
        self.U += ni * dU

        return sum(abs([sum(sum(dW)),sum(sum(dV)), sum(sum(dU))]))

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
