'''
Created on Mar 30, 2016
@author: anup
'''

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from collections import Counter

class Document(object):
    def __init__(self, topic, title, doc):
        """Initial the Document object with tokens and topics
        """
        if topic:
            self.terms = Counter()
            stopwrd = stopwords.words('english')
            self.topic = topic
            self.title = self.tokenize(title,stopwrd)
            self.tokens = self.tokenize(doc,stopwrd)
            
    def document_terms(self):
        return self.terms
    
    def tokenize(self,doc,stopwrd):
        """Tokenize using whitespace and words only"""
        
        result = []
        document = doc
        if type(doc) is list:
            document = '.'.join(doc)
            
        for sent in sent_tokenize(document):
            sent = sent.strip('.')
            for token in nltk.word_tokenize(sent):
                token = token.lower().strip('.').strip('\'')
                if token not in stopwrd and len(token) > 2 and re.match('^[0-9]*[a-zA-Z]+[0-9]*$', token):
                    self.terms[token] += 1
                    result.append(token)
        return result
