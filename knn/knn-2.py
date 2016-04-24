#!/usr/bin/env python


"""
Problem Definition :


"""

__author__ = 'vivek'

from util import *
import os
from tfidf import *
from document import *
import random


def euclidean_distance(doc1, doc2):
    """
    The euclidean distance between two docs
    :param doc1: First doc
    :param doc2: Second doc
    :return: the distance between docs.
    """

    distance = 0
    v1, v2 = doc1.vector, doc2.vector
    features = list(set(v1.keys()).union(v2.keys()))

    for feature in features:
        distance += pow((v1[feature] - v2[feature]), 2)

    return math.sqrt(distance)


def cosine_similarity(doc1, doc2):
    """
    The cosine_similarity between two docs
    :param doc1: First doc
    :param doc2: Second doc
    :return: the cosine_similarity between docs.
    """

    distance = 0
    v1, v2 = doc1.vector, doc2.vector

    # Choose the doc with less features to lessen the calculations.
    if len(v2.keys()) < len(v1.keys()):
        v1, v2 = v2, v1

    for feature in v1.keys():
        distance += (v1[feature] * v2[feature])

    return distance


def find_k_neighbours(docs, target, k):
    """
    Find K nearest neighbours of given doc
    :param docs: list of docs
    :param target: source doc
    :param k: parameter k
    :return: list of k nearest docs.
    """
    distance_list = list()

    # for each doc, find the similarity and update the distance list.
    for i in xrange(len(docs)):
        doc = docs[i]
        distance_list.append((i, cosine_similarity(doc, target)))

    # sort the list and pick top k results.
    sorted_dist_list = sorted(distance_list, key=lambda x: x[1], reverse=True)

    k_neighbours = list()

    for i in xrange(k):
        k_neighbours.append(docs[sorted_dist_list[k][0]])

    return k_neighbours


def classify(k_neighbours):
    """
    Classify the doc based on its k neghbours
    :param k_neighbours:
    :return: return the class with highest count.
    """

    class_list = [n.topic for n in k_neighbours]
    prediction = max(set(class_list), key=class_list.count)

    return prediction


def main():
    start_time = time.time()

    t_path = "../bbc/"


    all_docs = defaultdict(lambda: list())

    topic_list = list()

    print "Reading all the documents...\n"

    for topic in os.listdir(t_path):
        d_path = t_path + topic + '/'
        topic_list.append(topic)
        temp_docs = list()

        for f in os.listdir(d_path):
            f_path = d_path + f
            temp_docs.append(Document(f_path, topic))

        all_docs[topic] = temp_docs[:]

    fold_count = 10

    confusion_matrix, c_dict = init_confusion_matrix(topic_list)

    train_docs, test_docs = list(), list()

    for key, value in all_docs.items():
        random.shuffle(value)
        test_len = len(value)/fold_count
        train_docs += value[:-test_len]
        test_docs += value[-test_len:]

    index = Index(train_docs)

    test_topics = [d.topic for d in test_docs]

    for doc in train_docs:
        doc.vector = doc.tfidfie

    for doc in test_docs:
        doc.vector = doc.tf

    k = int(math.sqrt(len(train_docs)))

    prediction_list = list()

    print "Training and Testing together.. Finding K=" + str(k) + " neighbours, and classifying..\n"
    print "Train Length : ", len(train_docs)
    print "Test Length : ", len(test_docs)

    for doc in test_docs:
        k_neighbours = find_k_neighbours(train_docs, doc, k)
        prediction_list.append(classify(k_neighbours))

    confusion_matrix = update_confusion_matrix(test_topics, prediction_list, confusion_matrix, c_dict)

    accuracy = cal_accuracy(confusion_matrix)
    precision = cal_precision(confusion_matrix)
    recall = cal_recall(confusion_matrix)
    f_measure = cal_f_measure(precision, recall)

    print "\nConfusion Matrix"
    for item in confusion_matrix:
        print item

    result_table = list()

    result_table.append(["Measure", "Value"])

    result_table.append(["accuracy", str(accuracy)])
    result_table.append(["precision", str(precision)])
    result_table.append(["recall", str(recall)])
    result_table.append(["f_measure", str(f_measure)])

    print_table(result_table)

    print "Run time...{} secs \n".format(round(time.time() - start_time, 4))


if __name__ == '__main__':
    main()
