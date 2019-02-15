This is a general purpose sentence feature extraction algorithm called 
doc2vec("Document and sentence embeddings"-T.Mikolov and Li,2014).
I have also included a text preprocessing pipiline which includes tokenization, 
rare word removal, stopwords removal, html tags removal and detokenization, using
reges, nltk and some of my own solutions.
This pipeline could be used as a quick reference for preprocessing. 

It was trained on the Amazon Fine Foods corpus, which contains a lot of typing errors, random words and phrases.
Despite that, the model was able to generalise very well, for example it generalised phrases like "good food" with 
"tasty" and "big hit". Another example, "fast delivery" was generaalised with phrases like "arrived on time", "Amazon Prime delvery", etc.

The so obtained vectors after training can then be passed to a classifier for sentiment analysis, topic modelling or can even be used for 
semantic sentence similarity.
