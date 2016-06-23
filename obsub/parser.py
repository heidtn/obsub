"""
    This file parses tsv files with properly formatted data into training sets.  It assumes the format: "source (tab) sentence (tab) score (tab) (newline)
"""

import urllib2
from pysqlite2 import dbapi2 as sqlite
from BeautifulSoup import *
from urlparse import urljoin

import re
import string

import nltk
import itertools

import numpy as np

"""
this serves to pull articles down to parse into the RNN, can also pull articles down to create training sets.  Some parsing is inspired by the O'Reilly book Collective Intelligence.
"""

class Parser:
    def __init__(self, dbname):
        self.con = sqlite.connect(dbname)

    def __del__(self):
        self.con.close()

    def dbcommit(self):
        self.con.commit()

    def createTables(self):
        self.con.execute('create table wordlist(url, word, count)')
        self.con.execute('create index wordidx on wordlist(word)')
        self.dbcommit()

    def destroyTables(self):
        self.con.execute('drop table is exists wordlist')





""" create training data from tab seperated files from here: http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/"""
vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

def createTrainingData(filenames):    
    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print "Reading TSV file..."
    reader = []
    for filename in filenames:
        with open(filename, 'rb') as f:
            for line in f:
                els = line.split('\t')
                reader.append(els) #array of arrays [site, sentence, score]
    
    # Split full comments into sentences
    sentences = []
    for line in reader:
        for sents in nltk.sent_tokenize(line[1].decode('utf-8').lower()):
            sentences.append((sents, line[2]))

    #sentences = list(itertools.chain(*[[nltk.sent_tokenize(x[1].decode('utf-8').lower()), x[2]] for x in reader]))
    
    # Append SENTENCE_START and SENTENCE_END
    #sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print "Parsed %d sentences." % (len(sentences))
    
    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(x[0]) for x in list(sentences)]
     
    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print "Found %d unique words tokens." % len(word_freq.items())
     
    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
     
    print "Using vocabulary size %d." % vocabulary_size
    print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])
     
    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
     
    print "\nExample sentence: '%s'" % sentences[0][0]
    print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]
     
    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([float(x[1]) for x in sentences], dtype=float)

    return X_train, y_train, index_to_word, word_to_index

def main(): 
   xt, yt, index_to_word = createTrainingData(['../datasets/RT_train.txt'])

if __name__ == "__main__":
    main()