"""
	This is used to manually create new datasets from websites.
"""

import urllib2
from BeautifulSoup import *
from urlparse import urljoin


"""
    parse website articles manually to create a training set from human inputs
"""
def createTrainingSet(urls):
    sites = {}
    for url in urls:
        try:
            headers = { 'User-Agent' : 'Mozilla/5.0' }
            req = urllib2.Request(url, None, headers)
            c = urllib2.urlopen(req)
        except Exception, e:
            print ("couldn't open %s" % url), " Error: ", str(e)
            continue
        soup = BeautifulSoup(c.read())
        print "parsing: ", url
        sentences = splitElements(soup)
        #for sentence in sentences:
        #   print sentence
        ratings = defineElements(sentences)
        sites[url] = []
        for i in xrange(len(sentences)):
           sites[url].append((sentences[i], rating[i]))

    return sites


def splitElements(soup):
    [s.extract() for s in soup('script')] #remove all script tags
    [s.extract() for s in soup('head')] #remove all script tags
    onlytext = gettextonly(soup)
    onlytext.replace('\r', '').replace('\t', ' ').replace('\n', '')

    onlytext = ''.join([i if ord(i) < 128 else ' ' for i in onlytext])

    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in onlytext])
    #raw_sentences = re.split('\\.|\\?|\\!', onlytext) #split by sentences.  Can cause issues i.e. U.S. etc.
    #sentences = [sentence.strip().replace('\n', '') for sentence in raw_sentences 
    #                if not sentence.isspace() and sentence.find(' ') > 1]
    print list(sentences)
    return sentences

def defineElements(sentences):
    ratings = []
    print """Rate each sentence by how objective (1.0) or subjective (-1.0) it is.  If it is neither rate it at 0.  
If the sentence isn't a sentence or doesn't makes sense rate it as n\n\n"""
    for sentence in sentences:
        printable = set(string.printable)
        success = False
        request = str("rate this sentence: " + filter(lambda x: x in printable, sentence) + " ")

        while not success:
            rating = raw_input(request)
            if(rating != 'n'):
                try:
                    rating = float(rating)
                    if rating <= 1 and rating >= -1:
                        success = True
                    else:
                        print "rating must be between -1 and 1"
                except Exception:
                    print "input can only be the letter n or a number from -1 to 1"
            else:
                success = True
        ratings.append(rating)

    return ratings

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

def saveData(sites, filename, append=True):
    filemode = 'w'
    if append: filemode = 'a'
    with open(filename, filemode) as f:
        for site in sites:
            for sentence, rating in sites[site]:
                f.write(site)
                f.write("\t")
                f.write('"' + sentence + '"')
                f.write("\t")
                f.write(rating)
                f.write('\n')

def main()
    parser = Parser('wordvectors.db')
    #open file with test data
    #parse each test data element
    #train the RNN on the test data
    testarticles = ["https://medium.com/@ErinBrockovich/we-are-all-flint-42fb50e700fe#.hyftjcv9o"]
    sites = createTrainingSet(testarticles)
    saveData(sites, 'testdata.txt', append=False)

if __name__ == "__main__":
    main()