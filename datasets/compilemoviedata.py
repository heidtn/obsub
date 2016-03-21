"""this is the program I wrote to 'normalize' the rotten tomatoes data into one document"""

p = open('plot.tok.gt9.5000', 'r') #objective
q = open('quote.tok.gt9.5000', 'r') #subjective

objective = [line for line in p]
subjective = [line for line in q]

p.close()
q.close()

train = len(objective) - len(objective)/5
test = len(objective) - train


with open('RT_train.txt', 'w') as o:
	for i in xrange(train):
		o.write('www.rottentomatoes.com')
		o.write('\t')
		o.write(objective[i].strip().replace('\t', ' '))
		o.write('\t')
		o.write('1.0')
		o.write('\n')
	for j in xrange(train):
		o.write('www.rottentomatoes.com')
		o.write('\t')
		o.write(subjective[i].strip().replace('\t', ' '))
		o.write('\t')
		o.write('-1.0')
		o.write('\n')

with open('RT_test.txt', 'w') as o:
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
		o.write(subjective[i + train].strip().replace('\t', ' '))
		o.write('\t')
		o.write('-1.0')
		o.write('\n')

