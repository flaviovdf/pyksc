# -*- coding: utf8
'''This module contains the code used for data conversion'''
from __future__ import division, print_function

from collections import defaultdict

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import Vectorizer

import nltk

class NoopAnalyzer(BaseEstimator):
    '''
    Since we use NLTK to preprocess (more control) this
    class is used to bypass sklearns preprocessing
    '''
    def analyze(self, text_document):
        '''Does nothing'''
        return text_document

def __tokenize_and_stem(fpath):
    '''
    Tokenizes and stems the file, converting each line to 
    an array of words.
    
    Arguments
    ---------
    fpath: a path to a file 
        Each line is a song, tags are separated by space
    '''
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    stopwords = set(nltk.corpus.stopwords.words('english'))
    stemmer = nltk.stem.PorterStemmer()
    
    docs = []
    term_pops = defaultdict(int)
    with open(fpath) as tags_file:
        for line in tags_file:
            as_doc = []
            for term in tokenizer.tokenize(line)[1:]:
                term = term.lower().strip()
                if term not in stopwords and term != '':
                    stemmed = stemmer.stem(term)
                    as_doc.append(stemmed)
                    term_pops[stemmed] += 1
                   
            docs.append(as_doc)

    return docs, term_pops

def clean_up(fpath, bottom_filter=0.01):
    '''
    Converts a YouTube tag file to a series of tokens. This code
    stems the tags, removes stopwords and filters infrequent
    tags (whose probability is bellow `bottom_filter`).
    
    Arguments
    ---------
    fpath: a path to a file 
        Each line is a song, tags are separated by space
    bottom_filter: float (defaults to 0.01, one percent)
        Minimum probability for tags to be considered useful
    '''
    docs, term_pops = __tokenize_and_stem(fpath)
    for doc in docs:
        to_yield = []
        for term in doc:
            prob_term = term_pops[term] / len(term_pops)
            if prob_term > bottom_filter:
                to_yield.append(term)
        
        yield to_yield
    
def vectorize_videos(fpath, use_idf=False):
    '''
    Converts a YouTube tag file to a sparse matrix pondered. We can assign
    weights based on IDF if specified.
    
    Arguments
    ---------
    fpath: a path to a file 
        Each line is a song, tags are separated by space
    use_idf: bool (optinal, defaults to True)
        Indicates whether to use IDF.
    bottom_filter: float (defaults to 0.005, half of one percent)
        Minimum probability for tags to be considered useful
    '''
    #Vectorizes to TF-IDF
    vectorizer = Vectorizer(analyzer=NoopAnalyzer(), use_idf = use_idf)
    sparse_matrix = vectorizer.fit_transform(clean_up(fpath, bottom_filter=0))
    vocabulary = vectorizer.vocabulary
    return sparse_matrix, vocabulary