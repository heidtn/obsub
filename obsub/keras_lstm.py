import parser

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

import numpy as np

max_features = 8000
maxlen = 100  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

def getmodel():
	print('Build model...')
	model = Sequential()
	model.add(Embedding(max_features, 256, input_length=maxlen, dropout=0.4))
	model.add(LSTM(128, dropout_W=0.4, dropout_U=0.4))  # try using a GRU instead, for fun
	model.add(Dropout(0.4))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	print("compiling model")
	# try using different optimizers and different optimizer configs
	model.compile(loss='binary_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])
	return model


def fitmodel(model, X_train, X_test, y_train, y_test, epoch=30):
	X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
	X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
	score = model.evaluate(X_test, y_test, batch_size=32)
	print "score before eval: ", score
	model.fit(X_train, y_train, batch_size=32, nb_epoch=epoch)
	score = model.evaluate(X_test, y_test, batch_size=32)
	print "score after eval: ", score


unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

def testsentence(model, sentence, word_to_index):
	sent = []
	for word in sentence.split():
		if word in word_to_index:
			sent.append(word_to_index[word])
		else:
			sent.append(word_to_index[unknown_token])

	sent = np.array([sent])
	sent = sequence.pad_sequences(sent, maxlen=maxlen)
	toprint = model.predict_classes(sent, batch_size=32, verbose=1)
	return toprint


def main():
	X_train, y_train, index_to_word, word_to_index = parser.createTrainingData(['../datasets/RT_train.txt'])

	X_test = X_train[:2000]
	X_train = X_train[2000:]

	y_test = y_train[:2000]
	y_train = y_train[2000:]

	print "getting model"
	model = getmodel()
	print "fitting model"
	fitmodel(model, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
	main()