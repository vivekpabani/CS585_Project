#!/usr/bin/env python


"""
Problem Definition :

provides print table feature.
"""

__author__ = 'vivek'

from pylab import *


def print_table(table):
    """
    Prints the table data provided in list of lists format, where each list is a row.
    """
    col_width = [max(len(x) for x in col) for col in zip(*table)]
    lines = ['-'*(i + 2) + '-' for i in col_width]
    dash_line = ''.join(lines)

    print dash_line

    for index, line in enumerate(table):
        table_row = ""
        for i, x in enumerate(line):
            table_row += "{:{}}".format(x, col_width[i]) + " | "

        print "| " + table_row

        if index == 0:
            print dash_line

    print dash_line


def init_confusion_matrix(c_array):
    """
    initiate confusion matrix with 0 values.
    :param c_array: class array
    :return: 0 valued confusion matrix
    """

    class_list, counts = np.unique(np.array(c_array), return_counts=True)
    class_list = class_list.tolist()
    class_index_list = [(i, class_list.index(i)) for i in class_list]
    class_count = len(class_list)
    class_dict = dict(class_index_list)

    conf_mat = [[0 for i in xrange(class_count)] for j in xrange(class_count)]

    return conf_mat, class_dict


def update_confusion_matrix(true_y, pred_y, conf_mat, class_dict):
    """
    updates confusion matrix.
    :param true_y: true class values
    :param pred_y: predicted y values
    :param conf_mat: confusion matrix
    :param class_dict: class dictionary
    :return: updated confusion matrix
    """

    for t in xrange(len(true_y)):
        a = class_dict[true_y[t]]
        b = class_dict[pred_y[t]]
        conf_mat[a][b] = conf_mat[a][b] + 1

    return conf_mat


def cal_accuracy(conf_mat):
    """
    calculates accuracy based on confusion matrix
    :param conf_mat: confusion matrix
    :return: accuracy
    """

    numerator, denominator = 0, 0

    for i in xrange(len(conf_mat)):
        numerator = numerator + conf_mat[i][i]

        for j in xrange(len(conf_mat)):
            denominator = denominator + conf_mat[i][j]

    if denominator == 0:
        acc = 0
    else:
        acc = float(numerator)/denominator

    return acc


def cal_precision(conf_mat):
    """
    calculates precision for all classes based on confusion matrix
    :param conf_mat: confusion matrix
    :return: precision list
    """

    pre_list = list()

    for i in xrange(len(conf_mat)):
        numerator = conf_mat[i][i]
        denominator = 0

        for j in xrange(len(conf_mat)):
            denominator = denominator + conf_mat[i][j]

        if denominator == 0:
            pre = 0
        else:
            pre = float(numerator)/denominator
        pre_list.append(pre)

    return sum(pre_list)/len(pre_list)


def cal_recall(conf_mat):
    """
    calculates recall for all classes based on confusion matrix
    :param conf_mat: confusion matrix
    :return: recall list
    """

    rec_list = list()

    for i in xrange(len(conf_mat)):
        numerator = conf_mat[i][i]
        denominator = 0

        for j in xrange(len(conf_mat)):
            denominator = denominator + conf_mat[j][i]
        if denominator == 0:
            rec = 0
        else:
            rec = float(numerator)/denominator
        rec_list.append(rec)

    return sum(rec_list)/len(rec_list)


def cal_f_measure(prec, rec):
    """
    calculates f measure from precision and recall
    :param prec: precision
    :param rec: recall
    :return: f measure
    """

    if prec + rec == 0:
        f_m = 0
    else:
        f_m = float(2 * prec * rec) / (prec + rec)

    return f_m

