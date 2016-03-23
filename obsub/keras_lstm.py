import parser

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM


def getmodel():

	model = Sequential()
	model.add(Embedding(8000, 8))
	#model.regularizers = []
	model(LSTM(128, activation='sigmoid', 
	               inner_activation='hard_sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam')

def fitmodel(model, X_train, y_train, epoch=10):
	score = model.evaluate(X_test, Y_test, batch_size=16)
	print "score before eval: ", score
	model.fit(X_train, y_train, batch_size=16, nb_epoch=epoch)
	score = model.evaluate(X_test, Y_test, batch_size=16)
	print "score after eval: ", score

def main():
    X_train, y_train = parser.createTrainingData(['../datasets/RT_train.txt'])

    print "getting model"
	model = getmodel()
	print "fitting model"
	fitmodel(model, X_train, y_train)


if __name__ == "__main__":
	main()