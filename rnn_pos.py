#!/usr/bin/env python
from numpy import zeros, ones, dot, exp, mat, array
import json
import sys

from rnn import RNN

OOV_TAG = "OOV"

def swap_tuple(x):
    return (x[1], x[0])

def matrix_to_list(m):
    return [list(x) for x in m]
    

class RNNPOSTagger:
    def __init__(self, vocab, tags):
        self.vocab = vocab
        self.tags = tags
        self.tag_ndx = dict(map(swap_tuple, enumerate(self.tags)))
        self.vocab_ndx = dict(map(swap_tuple, enumerate(self.vocab)))

    def save_to_file(self, filename):
        f = open(filename, 'wb')

        context = {}
        context['vocab'] = self.vocab
        context['tags'] = self.tags
        context['n_inputs'] = self.rnn.n_inputs
        context['n_outputs'] = self.rnn.n_outputs
        context['n_hidden'] = self.rnn.n_hidden        
        context['U'] = matrix_to_list(self.rnn.U)
        context['V'] = matrix_to_list(self.rnn.V)
        context['W'] = matrix_to_list(self.rnn.W)
                
        f.write(json.dumps(context))
        f.close()

    def load_from_file(self, filename):
        f = open(filename, 'rb')

        context = json.loads(f.read())
        self.vocab = context['vocab']
        self.tags = context['tags']
        self.__init__(self.vocab, self.tags)
        
        self.rnn = RNN(context['n_inputs'], context['n_outputs'], context['n_hidden'])
        self.rnn.U = array([array(x) for x in context['U']])
        self.rnn.V = array([array(x) for x in context['V']])
        self.rnn.W = array([array(x) for x in context['W']])

        f.close()

    def train(self, data, hidden_layer_cnt = 10):
        n_input = len(self.vocab)
        n_output = len(self.tags)
        n_hidden = hidden_layer_cnt

        self.rnn = RNN(n_input, n_output, n_hidden)

        training_set = self.prepare_training_set(data)
        for epoch in range(10000):
            print "Running epoch #%d" % epoch
            self.rnn.train(training_set[:10])
            if epoch % 20 == 0:
                self.save_to_file('_tmp_save')

    def get_tag(self, word):
        input_vector = array([0.0 for i in range(len(self.vocab))])
        if self.vocab_ndx.has_key(word):
            input_vector[self.vocab_ndx[word]] = 1.0
            self.rnn.reset_hidden()
            res = self.rnn.feed(input_vector)
            print res
            res_ndx = res.argmax()
            return self.tags[res_ndx]
        else:
            return ""

    def prepare_training_set(self, data):
        vocab_size = len(self.vocab)
        tag_count = len(self.tags)
        res = []        
        for word, pos in data:
            #x = zeros(vocab_size)
            #y = zeros(tag_count)
            #x[self.vocab_ndx[word]] = 1
            #y[self.tag_ndx[pos]] = 1
            #res += [(x, y)]
            res += [(self.vocab_ndx[word], self.tag_ndx[pos])]

        return res

def train_tagger(input_file, output_file):

    f_in = open(input_file, 'r')
    
    data = []
    word_dict = {}
    tag_dict = {}
    for ln in f_in:
        word, pos = ln.strip().split('\t')
        word_dict[word] = None
        tag_dict[pos] = None
        data += [(word, pos)]
    
    tagger = RNNPOSTagger(word_dict.keys(), tag_dict.keys())
    #tagger.load_from_file(output_file)
    #test_in = array([0.0 for i in range(len(word_dict))])
    #print tagger.rnn.feed(test_in)
    #exit(0)

    tagger.train(data)

    tagger.save_to_file(output_file)

    
def run_tagger(input_file, output_file):
    tagger = RNNPOSTagger({}, {})
    tagger.load_from_file(output_file)

    good = 0
    bad = 0

    f_in = open(input_file, 'r')
    for ln in f_in:
        word, pos = ln.strip().split('\t')
        mytag = tagger.get_tag(word)

        if mytag == pos:
            good += 1
        else:
            bad += 1
        
        print "%s\t%s\t%s" % (word, mytag, pos)
    print good, bad


if __name__ == "__main__":
    argv = sys.argv
    if argv[1] == "train":
        train_tagger(argv[2], argv[3])
    elif argv[1] == "run":
        run_tagger(argv[2], argv[3])
        
