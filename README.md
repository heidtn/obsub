### Name

# What does this do?
NAME is a system based on RNN (recurrent neural networks) with LSTM (long short-term memory) that parses sentences to reveal if they are an opinion or a fact.

# How do I set it up?
run pip install -r pip_dependencies to install the necessary libraries

run python serve.py for an interactive page
run python compute.py 'http://www.website.com/pathtoarticle'

# Why is this hard?
Often the hardest part of designing any machine learning program is determining which algorithm to use.  There's 100's and each has it's place. RNN's are the new hotness right now as far as machine learning goes.  They're based on a concept call Hierarchical Temporal Memory that emulates the neo-cortex.  In short they've been done a lot.  I've chosen them for this project as it seems to be a good fit for parsing sentences and it will be interesting to see what they can do.

# Why is this useful? 
I intend to run this on news articles to determine how opinionated a piece is.  There's a lot of potential for something like this in terms of fact checking articles as well.  It's mostly a non-specific curiosity and could potentiall provide some interesting results.  

# Disclaimer
Please note that I am in no implying that the objectivity of a piece or article has any specific merit or detriment.  Also note that in training sets things may errantly be marked as opinion when they are really a fact.  It is not an exact science.  Personal experience can be very difficult to separate fact from opinion.  I pick pieces I think have a good combination of facts and opinions.  The point isn't to prove the validity of any statement.  There is no hypothesis here.  I only wish to see if the data shows anything new.