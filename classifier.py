#!/usr/bin/env python


"""
Problem Definition :


"""

__author__ = 'vivek'

import time
import re
import os
import math
from collections import defaultdict, Counter
from nltk.stem.porter import *
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import operator


class TopicSet(object):

    def __init__(self, docs, topics):
        self.docs = docs
        self.topics = topics
        self.title_tokens = defaultdict(lambda: list())
        self.text_tokens = defaultdict(lambda: list())
        self.title_doc_freqs = defaultdict(lambda: list())
        self.text_doc_freqs = defaultdict(lambda: list())

        for doc in self.docs:
            self.title_tokens[doc.topic].append(doc.title_tokens)
            self.text_tokens[doc.topic].append(doc.text_tokens)

        for topic in self.topics:
            self.title_doc_freqs[topic] = self.count_doc_frequencies(self.title_tokens[topic])
            self.text_doc_freqs[topic] = self.count_doc_frequencies(self.text_tokens[topic])

        self.title_common_tokens = self.find_common_tokens(self.title_doc_freqs)
        self.text_common_tokens = self.find_common_tokens(self.text_doc_freqs)

    def count_doc_frequencies(self, token_l):
        """
        :param token_l: A list of lists of tokens, one per document. This is the output of the tokenize method.
        :return: A dict mapping from a term to the number of documents that contain it.
        """

        doc_freqs = defaultdict(lambda: 0)
        doc_count = len(token_l)

        for doc in token_l:
            for token in set(doc):
                doc_freqs[token] += 1

        for key, value in doc_freqs.items():
            doc_freqs[key] = value*1.0 / doc_count

        return doc_freqs

    def find_common_tokens(self, doc_freqs):

        token_count = defaultdict(lambda: 0)
        common_tokens = list()

        threshold = int(math.floor(len(doc_freqs.keys())/2.0))

        for topic in doc_freqs.keys():
            doc_freq = doc_freqs[topic]
            doc_tokens = filter(lambda x: doc_freq[x] > 0.1, doc_freq.keys())

            for token in doc_tokens:
                token_count[token] += 1

        for token, count in token_count.items():
            if count > threshold:
                common_tokens.append(token)

        return common_tokens


class Document(object):

    def __init__(self, f_path, topic=None):
        self.stop_words = stopwords.words('english')
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

    def tokenize(self, data):
        return [t.lower() for t in re.findall(r"\w+(?:[-']\w+)*", data) if t not in self.stop_words]

    def stem(self, tokens):
        stemmer = PorterStemmer()
        return [stemmer.stem(token) for token in tokens]

    def lemmatize(self, tokens):
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]


class Index(object):

    def __init__(self, docs, text_exclude_tokens=None, title_exclude_tokens=None):

        if not text_exclude_tokens:
            text_exclude_tokens = list()
        if not title_exclude_tokens:
            title_exclude_tokens = list()

        self.docs = docs

        self.title_tokens = [[t for t in d.title_tokens if t not in (title_exclude_tokens+text_exclude_tokens)] for d in self.docs]
        self.title_doc_freqs = self.count_doc_frequencies(self.title_tokens)
        self.title_index = self.create_tf_index(self.title_tokens)
        self.title_lengths, self.mean_title_length = self.compute_doc_lengths(self.title_tokens)
        self.title_tfidf = self.create_tfidf_index(self.title_tokens, self.title_index, self.title_lengths, self.title_doc_freqs)
        self.topic_title_tfidf = self.create_topic_tfidf_index(self.title_tokens, self.title_index, self.title_doc_freqs)

        self.text_tokens = [[t for t in d.text_tokens if t not in text_exclude_tokens] for d in self.docs]
        self.text_doc_freqs = self.count_doc_frequencies(self.text_tokens)
        self.text_index = self.create_tf_index(self.text_tokens)
        self.text_lengths, self.mean_text_length = self.compute_doc_lengths(self.text_tokens)
        self.text_tfidf = self.create_tfidf_index(self.text_tokens, self.text_index, self.text_lengths, self.text_doc_freqs)
        self.topic_text_tfidf = self.create_topic_tfidf_index(self.text_tokens, self.text_index, self.text_doc_freqs)

    def count_doc_frequencies(self, token_l):
        """
        :param token_l: A list of lists of tokens, one per document. This is the output of the tokenize method.
        :return: A dict mapping from a term to the number of documents that contain it.
        """

        doc_freqs = defaultdict(lambda: 0)

        for doc in token_l:
            for token in set(doc):
                doc_freqs[token] += 1

        return doc_freqs

    def create_tf_index(self, token_l):
        """
        Create an index in which each postings list contains a list of [doc_id, tf weight] pairs.
        :param token_l: list of lists, where each sublist contains the tokens for one document.
        :return:
        """

        index = defaultdict(lambda: list())

        for i in xrange(len(token_l)):
            doc = token_l[i]
            counter = Counter(doc)
            for token in counter.keys():
                index[token].append([i, counter[token]])
        
        return index

    def create_tfidf_index(self, token_l, tf_index, doc_lengths, doc_freqs):

        doc_count = len(token_l)

        tfidf = defaultdict(lambda: list())

        for token, freq_l in tf_index.items():
            token_idf = doc_freqs[token]*1.0/doc_count
            for token_freq in freq_l:
                score = (1 + token_freq[1]*1.0/doc_lengths[token_freq[0]]) * (1 + token_idf)
                tfidf[token_freq[0]].append([token, score])

        return tfidf

    def create_topic_tfidf_index(self, token_l, tf_index, doc_freqs):

        doc_count = len(token_l)
        all_token_count = sum(len(t) for t in token_l)

        tfidf = defaultdict(lambda: 0)

        for token, freq_l in tf_index.items():
            token_count = sum(i[1] for i in freq_l)
            token_idf = doc_freqs[token]*1.0/doc_count
            score = (1 + token_count*1.0/all_token_count) * (1 + token_idf)
            tfidf[token] = score

        return tfidf

    def compute_doc_lengths(self, token_l):

        doc_lengths = defaultdict(lambda: 0)
        total_len = 0

        for i in xrange(len(token_l)):
            doc = token_l[i]
            doc_len = len(doc)
            doc_lengths[i] = doc_len
            total_len += doc_len

        return doc_lengths, total_len/len(token_l)


class Classifier(object):

    def __init__(self, index_dict):
        self.index_dict = index_dict

    def classify(self, doc):
        score_dict = defaultdict(lambda: 0)
        for topic, index in self.index_dict.items():
            score_dict[topic] = self.cal_score(doc, index)

        print "Scores : ", score_dict.items()

        return max(score_dict.iteritems(), key=operator.itemgetter(1))[0]

    def cal_score(self, doc, index):
        text_tfidf = index.topic_text_tfidf
        title_tfidf = index.topic_title_tfidf

        title_score, text_score = 0.0, 0.0

        text_len = len(doc.text_tokens)
        title_len = len(doc.title_tokens)

        for token in doc.title_tokens:
            if token in title_tfidf:
                title_score += (2*title_tfidf[token])
            elif token in text_tfidf:
                title_score += (1.5*text_tfidf[token])

        for token in doc.text_tokens:
            if token in text_tfidf.keys():
                text_score += text_tfidf[token]

        total_score = (title_score/title_len) + (text_score/text_len)

        return total_score


def main():
    start_time = time.time()

    t_path = "./bbc/"

    train_docs = defaultdict(lambda: list())
    test_docs = list()

    topic_list = list()

    print "Reading all the documents. Separating Train and Test Documents...\n"

    for topic in os.listdir(t_path):
        d_path = t_path + topic + '/'
        topic_list.append(topic)
        temp_docs = list()

        for f in os.listdir(d_path):
            f_path = d_path + f
            temp_docs.append(Document(f_path, topic))

        train_docs[topic] = temp_docs[:-10]
        test_docs += temp_docs[-10:]

    all_docs = list()

    for key, value in train_docs.items():
        all_docs = all_docs + value

    print "Finding the common tokens within all the topics..\n"

    topic_set = TopicSet(all_docs, topic_list)

    print "Common Tokens - To Be Excluded : \n ", topic_set.text_common_tokens

    index_dict = defaultdict(lambda: None)

    print "\nTop 20 Tokens Per Topic : \n"
    for topic, docs in train_docs.items():
        print "Topic : ", topic
        print ""
        index_dict[topic] = Index(docs, topic_set.text_common_tokens)

        print "Tokens : ", sorted(index_dict[topic].topic_text_tfidf.items(), key=lambda t: t[1], reverse=True)[:20]
        print ''

    classifier = Classifier(index_dict)

    print "\nTest Classification: \n"
    print "If mismatch in prediction, you will see the title and text of the article for inspection.\n"

    for doc in test_docs:
        prediction = classifier.classify(doc)
        print "Original : " + doc.topic + " | Predicted : " + prediction

        # If mismatch in prediction, print the title and text of the article for inspection.
        if doc.topic != prediction:
            print "\nTitle Tokens : ", doc.title
            print "Text Tokens : ", doc.text

        print ''

    print "Run time...{} secs \n".format(round(time.time() - start_time, 4))


if __name__ == '__main__':
    main()