#!/usr/bin/env python


"""
Problem Definition :


"""

__author__ = 'vivek'

import time
from collections import defaultdict
import os
from nb import *
from rank_classifier import *
from knn import *
import random
from document import *
from tfidf import *


def recommendation(all_docs, test_docs, classifier_list):

    user_docs = random.sample(test_docs, 5)

    for i in xrange(len(user_docs)):
        print str(i+1) + ": " + user_docs[i].title

    choice = int(raw_input("Enter Document Number: ")) - 1
    selected_doc = user_docs[choice]

    classifier_list = sorted(classifier_list, key=lambda cl: cl.stats['f_measure'], reverse=True)

    prediction_list = list()
    for classifier in classifier_list:
        prediction_list.append(classifier.classify([selected_doc])[0])

    prediction_count = Counter(prediction_list)
    top_prediction = prediction_count.most_common(1)
    print "top_prediction", top_prediction
    if top_prediction[0][1] > 1:
        prediction = top_prediction[0]
    else:
        prediction = prediction_list[0]

    print "prediction_list", prediction_list
    print "prediction", prediction

    knn = KNN(all_docs[prediction])

    k_neighbours = knn.find_k_neighbours(selected_doc, 5)

    for i in xrange(len(k_neighbours)):
        print k_neighbours[i].title


def recommendation2(all_docs, test_docs, classifier_list):

    option_count = 5
    end = False

    while not end:
        user_docs = random.sample(test_docs, option_count)

        while True:
            print "\n---Available Choices For Articles(Titles)---\n"

            for i in xrange(len(user_docs)):
                print str(i+1) + ": " + user_docs[i].title

            print "r: Refresh List\n"
            print "q: Quit()\n"

            choice = raw_input("Enter Choice: ")

            if choice == 'q':
                end = True
                break
            elif choice == 'r':
                break
            else:
                try:
                    user_choice = int(choice) - 1
                except:
                    print "Invalid Choice.. Try Again.."
                    continue
                selected_doc = user_docs[user_choice]

                classifier_list = sorted(classifier_list, key=lambda cl: cl.stats['f_measure'], reverse=True)

                prediction_list = list()
                for classifier in classifier_list:
                    prediction_list.append(classifier.classify([selected_doc])[0])

                prediction_count = Counter(prediction_list)
                top_prediction = prediction_count.most_common(1)

                print "top_prediction", top_prediction
                if top_prediction[0][1] > 1:
                    prediction = top_prediction[0]
                else:
                    prediction = prediction_list[0]

                knn = KNN(all_docs[prediction])

                k_neighbours = knn.find_k_neighbours(selected_doc, 5)

                while True:
                    print "\n---Recommended Articles for : " + selected_doc.title + " ---\n"
                    for i in xrange(len(k_neighbours)):
                        print str(i+1) + ": " + k_neighbours[i].title
                    next_choice = raw_input("\nEnter Next Choice: [Article num to read the article. "
                                            "'o' to read the original article. "
                                            "'b' to go back to article choice list.]  ")

                    if next_choice == 'c':
                        break
                    elif next_choice == 'b':
                        text = selected_doc.text
                        print "\nArticle Text for original title : " + selected_doc.title
                        print text
                    else:
                        try:
                            n_choice = int(next_choice) - 1
                        except:
                            print "Invalid Choice.. Try Again.."
                            continue
                        text = k_neighbours[n_choice].text
                        print "\nArticle Text for recommended title : " + k_neighbours[n_choice].title
                        print text


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

    train_docs, test_docs = list(), list()

    for key, value in all_docs.items():
        random.shuffle(value)
        test_len = len(value)/fold_count
        train_docs += value[:-test_len]
        test_docs += value[-test_len:]

    index = Index(train_docs)

    test_topics = [d.topic for d in test_docs]

    for doc in train_docs:
        doc.vector = doc.tfidf

    for doc in test_docs:
        doc.vector = doc.tf

    nb = NaiveBayes()
    rc = RankClassifier()

    classifier_list = [rc]

    for i in xrange(len(classifier_list)):

        print "Classifier #" + str(i+1) + "\n"

        classifier = classifier_list[i]

        classifier.confusion_matrix, c_dict = init_confusion_matrix(topic_list)

        print "Training...\n"

        classifier.train(train_docs)

        print "Testing... Classifying the test docs...\n"

        predictions = classifier.classify(test_docs)

        # Update the confusion matrix with updated values.
        classifier.confusion_matrix = update_confusion_matrix(test_topics, predictions, classifier.confusion_matrix,
                                                              c_dict)

        classifier.stats = cal_stats(classifier.confusion_matrix)
        print_table(get_stats_table(classifier.stats))

    recommendation2(all_docs, test_docs, classifier_list)

    print "Run time...{} secs \n".format(round(time.time() - start_time, 4))

if __name__ == '__main__':
    main()