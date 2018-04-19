# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 20:04:02 2018


"""

import re
import csv
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy import spatial
        

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

count=0
list1=[]
with open('Reviews.csv') as csvfile:
    csvreader= csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        count=count+1
        list1.append(row[-1])


    list2=[]
    sentences=[]
    sentence=''
    for row in list1:# sentence tokenizer
        sentences=sent_tokenize(row)
        for sentence in sentences:
            list2.append(sentence)

    list2temp=[]
    for sentence in list2:#clean html tags
        list2temp.append(cleanhtml(sentence))    
    list2=list2temp[:]


    list2temp=[]
    for sentence in list2:#remove very small sentences
        if(not(len(sentence) < 10)):
            list2temp.append(sentence)
    list2=list2temp[:]


    list2temp=[]
    for sentence in list2:#remove garbage
        if((sentence.find("Length::")==-1)):
            list2temp.append(sentence)
    list2=list2temp[:]
    

    list2temp=[]
    for sentence in list2:#convert to lowercase
        list2temp.append(sentence.lower())
    list2=list2temp[:]
   
    
    punctuation = '''''!()-[]{};:'"\,<>./?@#$%^&*_~'''#remove punctuations
    list2temp=[]
    no_punct = ""
    for sentence in list2:
        for char in sentence:  
           if char not in punctuation:  
               no_punct = no_punct + char  
        list2temp.append(no_punct)
        no_punct=""
    list2=list2temp[:]    

    list3=[]
    for row in list2:#tokenize
        words= nltk.word_tokenize(row)
        list3.append(words)
    
    
    #remove stopwords
    swords=['ourselves', 'hers', 'yourself' , 'there',  'having', 'with', 'they', 'an', 'be', 'for', 'do', 'its', 'yours', 'such', 'into',  'itself', 'other',  'is', 's', 'am',  'as',  'him', 'each', 'the', 'themselves',  'are', 'we', 'these', 'your', 'his',   'me', 'were', 'her', 'himself', 'this',  'should', 'our', 'their', 'while',    'to', 'ours', 'had', 'she',  'no', 'when', 'at', 'any', 'them',  'and',  'have',  'will',  'does', 'yourselves', 'then', 'that',  'what',  'why', 'so', 'can', 'did', 'he', 'you', 'herself', 'has', 'just',  'too',  'myself', 'which', 'those', 'i', 'whom', 't', 'being', 'if', 'theirs', 'my',  'a', 'by', 'doing', 'it', 'was', 'here', 'than']
    list3temp=[]
    sent3temp=[]
    for sentence in list3:
        for word in sentence:
            if(word not in swords):
                sent3temp.append(word)
        list3temp.append(sent3temp)
        sent3temp=[]
    list3=list3temp[:]
    
    temp1=""
    list3temp=[]
    #detokenize
    for line in list3:#detokenize
        for word in line:
            temp1=temp1+word+' '
        list3temp.append(temp1)
        temp1=""
    list3=list3temp[:]
    
    
    
    #gensim doc2vec
    from collections import namedtuple
    import re 
    import string
    
    doc1=list3[:]
    docs = []
    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
    for i, text in enumerate(doc1):
        words = text.split()
        tags = [i]
        docs.append(analyzedDocument(words, tags))
    
    # Train model (set min_count = 1, if you want the model to work with the provided example data set)
    import random
    from gensim.models import doc2vec
    from collections import namedtuple
    import nltk
    passes = 100             # Number of passes of one document during training
    
    
    model = doc2vec.Doc2Vec( size = 100 #model 
        , window = 10
        , min_count = 1
        , workers = 4
        ,alpha=0.025
        ,min_alpha=0.015
        ,dm=1)
    model.build_vocab(docs) # Build vocabulary
    for epoch in range(passes):
    
        # shuffle
    
        random.shuffle(docs)
        model.min_alpha = model.alpha
        # Train
        model.train(docs, total_examples=model.corpus_count, epochs=model.iter)
        
        print('Completed pass %i ' % (epoch + 1))
        from sklearn.metrics.pairwise import cosine_similarity
        # lower alpha
        model.alpha -= 0.002
        print('\n\n')
        infer_vector1=model.infer_vector("good food")
        infer_vector2=model.infer_vector("late delivery")
        print (1-spatial.distance.euclidean(infer_vector1, infer_vector2))
        print('\n\n')
        print(type(infer_vector1))
        sample="poor taste wo".split()
        infer_sample=model.infer_vector(sample)
        similar_doc=model.docvecs.most_similar([infer_sample], topn=5)
        for row in similar_doc:
            print(doc1[row[0]]+"\n")
        print("\n\n")
        sample="nutritious food good for health".split()
        infer_sample=model.infer_vector(sample)
        similar_doc=model.docvecs.most_similar([infer_sample], topn=5)
        for row in similar_doc:
            print(doc1[row[0]]+'\n')
        print("\n\n")    
    # Get the vectors
        
