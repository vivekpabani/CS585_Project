from collections import defaultdict
import operator
from util import *
import math
import numpy as np


class NaiveBayes(object):

    def __init__(self):
        # stores (class, #documents)
        self.class_docs = defaultdict(lambda: 0)
        # stores (class, prior)
        self.class_priors = defaultdict(lambda: 0)
        # stores (class, (doc, [tokens]))
        self.class_token_list = defaultdict(lambda: defaultdict(lambda: list()))
        # stores (class, [tokens])
        self.class_tokens = defaultdict(lambda: list())
        # stores (class, (token, {"mean", "var"}))
        self.class_token_stat = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))

        self.confusion_matrix = None

    def train(self, documents):
        """
        Given a list of labeled Document objects, compute the class priors and class feature stats.
        """
        for document in documents:

            topic = document.topic
            vec = document.vector

            # count documents per class
            self.class_docs[topic] += 1

            for token in vec.keys():
                self.class_token_list[topic][token].append(vec[token])
                self.class_tokens[topic].append(token)

        for topic in self.class_token_list.keys():
            for token in self.class_token_list[topic].keys():
                zeros = [0] * (len(self.class_tokens[topic]) - len(self.class_token_list[topic][token]))
                self.class_token_list[topic][token] = self.class_token_list[topic][token] + zeros
                self.class_token_stat[topic][token]["mean"] = np.mean(self.class_token_list[topic][token])
                self.class_token_stat[topic][token]["var"] = np.var(self.class_token_list[topic][token])

        for key in self.class_docs.keys():
            self.class_priors[key] = float(self.class_docs[key])/float(len(documents))

    def classify(self, documents):
        """
        Classify the list of documents.
        :param documents: The list of documents.
        :return: a list of strings, the class topics, for each document.
        """

        predictions = list()
        scores = defaultdict(lambda: 0)

        for document in documents:
            vec = document.vector

            for topic, prior in self.class_priors.items():
                scores[topic] = math.log10(prior)

                for token in vec.keys():
                    t_mean, t_var = self.class_token_stat[topic][token]["mean"], \
                                    self.class_token_stat[topic][token]["var"]
                    if t_var != 0 and t_mean != 0:
                        token_score = (1/math.sqrt(2*math.pi*t_var**2)) * math.exp(-(t_var - t_mean)**2/(2*t_var**2))
                        scores[topic] += token_score

            predictions.append(max(scores.iteritems(), key=operator.itemgetter(1))[0])

            scores = dict.fromkeys(scores, 0)

        return predictions

