
import urllib2
from pysqlite2 import dbapi2 as sqlite
from BeautifulSoup import *
from urlparse import urljoin

import re

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



"""
	parse articles to create a training set from human inputs
"""
def createTrainingSet(urls):
	for url in urls:
		try:
			headers = { 'User-Agent' : 'Mozilla/5.0' }
			req = urllib2.Request(url, None, headers)
			c = urllib2.urlopen(req)
		except Exception, e:
			print ("couldn't open %s" % url), " Error: ", str(e)
			continue
		soup = BeautifulSoup(c.read())
		[s.extract() for s in soup('script')] #remove all script tags
		[s.extract() for s in soup('head')] #remove all script tags
		onlytext = gettextonly(soup)
		print onlytext

#gets only the text portion of a website
def gettextonly(soup):		
	v = soup.string #purely text portion between tags
	if v == None:
		c = soup.contents #nested elements of a tag
		resulttext = ''
		for t in c:
			subtext = gettextonly(t) #...recursion...
			resulttext += subtext + '\n'
		return resulttext
	else:
		return v.strip()


def main():
	parser = Parser('wordvectors.db')
	#open file with test data
	#parse each test data element
	#train the RNN on the test data
	testarticles = ["https://medium.com/@ErinBrockovich/we-are-all-flint-42fb50e700fe#.hyftjcv9o"]
	createTrainingSet(testarticles)

if __name__ == "__main__":
	main()