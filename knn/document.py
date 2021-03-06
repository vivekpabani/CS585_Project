#!/usr/bin/env python


"""
Problem Definition :


"""

__author__ = 'vivek'

from nltk.stem.porter import *
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import defaultdict


class Document(object):
    stop_words = stopwords.words('english')

    def __init__(self, f_path, topic=None):

        self.content = open(f_path, 'r').readlines()
        self.title = self.content[0]
        self.text = ' '.join(self.content[1:])
        self.topic = topic

        """
        # No stemmer or lemmatizer.
        self.title_tokens = self.tokenize(self.title)
        self.text_tokens = self.tokenize(self.text)

        # Only stemmer.
        self.title_tokens = self.stem(self.tokenize(self.title))
        self.text_tokens = self.stem(self.tokenize(self.text))

        """
        # Only lemmatizer.
        self.title_tokens = self.lemmatize(self.tokenize(self.title))
        self.text_tokens = self.lemmatize(self.tokenize(self.text))
        """

        # Both stemmer and lemmatizer.
        self.title_tokens = self.stem(self.lemmatize(self.tokenize(self.title)))
        self.text_tokens = self.stem(self.lemmatize(self.tokenize(self.text)))
        """

        self.tf = self.tf_index(self.text_tokens)

        self.tfidf = None
        self.tfidfie = None
        self.vector = None

    def tokenize(self, data):
        return [t.lower() for t in re.findall(r"\w+(?:[-']\w+)*", data) if t not in self.stop_words]

    def stem(self, tokens):
        stemmer = PorterStemmer()
        return [stemmer.stem(token) for token in tokens]

    def lemmatize(self, tokens):
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]

    def tf_index(self, token_list):
        tf = defaultdict(lambda: 0)

        for token in token_list:
            tf[token] = tf[token] + 1

        return tf
