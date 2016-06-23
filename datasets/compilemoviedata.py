import random

"""this is the program I wrote to 'normalize' the rotten tomatoes data into one document"""

p = open('plot.tok.gt9.5000', 'r') #objective
q = open('quote.tok.gt9.5000', 'r') #subjective

objective = [line.decode('utf-8','ignore').encode("utf-8") for line in p]
subjective = [line.decode('utf-8','ignore').encode("utf-8") for line in q]

print("there are %d objective lines and %d subjective lines" % (len(objective), len(subjective)) )

p.close()
q.close()

train = len(objective) - len(objective)/5
test = len(objective) - train


with open('RT_train.txt', 'w') as o:
	trainset = []
	for i in xrange(train):
		s = ""
		s += 'www.rottentomatoes.com'
		s += '\t'
		s += objective[i].strip().replace('\t', ' ')
		s += '\t'
		s += '1.0'
		s += '\n'
		trainset.append(s)
	for j in xrange(train):
		s = ""
		s += 'www.rottentomatoes.com'
		s += '\t'
		s += subjective[j].strip().replace('\t', ' ')
		s += '\t'
		s += '0.0'
		s += '\n'
		trainset.append(s)

	random.shuffle(trainset)
	for item in trainset:
		o.write(item)

with open('RT_cv.txt', 'w') as o:
	for i in xrange(test):
		o.write('www.rottentomatoes.com')
		o.write('\t')
		o.write(objective[i + train].strip().replace('\t', ' '))
		o.write('\t')
		o.write('1.0')
		o.write('\n')
	for j in xrange(test):
		o.write('www.rottentomatoes.com')
		o.write('\t')
		o.write(subjective[j + train].strip().replace('\t', ' '))
		o.write('\t')
		o.write('0.0')
		o.write('\n')

