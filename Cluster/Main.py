'''
Created on Apr 15, 2016
@author: anup
'''
import os
from collections import defaultdict
from Document import Document
from collections import Counter
from classify import KMeans
from Validate import validate

def prune_terms(docs, min_df=3):
    """Prune Terms which do not occur on min_df number of documents"""
    #term_doc_freq = defaultdict(lambda: 0)
    term_doc_freq = Counter()
    for doc in docs:
        for term in doc.keys():
            term_doc_freq[term] += 1
            
    result = []
    for doc in docs:
        freq = Counter()
        for term in doc.keys():
            if term_doc_freq[term] >= min_df:
                freq[term] += doc[term]
        if freq:
            result.append(freq)
    
    return result

def Kvalidate():
    train_fld = os.path.join(os.getcwd(),'bbc')
    topics = set()
    
    for topic in os.listdir(train_fld):
        topics.add(topic)
    
    all_docs = list()
    for topic in topics:        
        topic_dir = os.path.join(train_fld,topic)
        for file in os.listdir(topic_dir):
            file_path = os.path.join(topic_dir,file)
            document = open(file_path,encoding='latin-1').readlines()
            all_docs.append(Document(topic,document[0],document[1:]))
    
    #80% file used for training
    validate().validate(all_docs)
    
    #60% file used for testing
    validate().validate(all_docs,k=0.4)
    
    #50% file used for testing
    validate().validate(all_docs,k=0.5)
    
    #40% file used for testing
    validate().validate(all_docs,k=0.6)
    
def main():
    Kvalidate()
    '''train_fld = os.path.join(os.getcwd(),'bbc')
    topics = set()
    
    for topic in os.listdir(train_fld):
        topics.add(topic)
    
    
    factor = 0.2
    file_path = str()
    
    test_docs = list()
    train_docs = defaultdict(lambda: list())
    for topic in topics:        
        topic_dir = os.path.join(train_fld,topic)
        
        files = os.listdir(topic_dir)
        size = int(len(files) * factor )
        for file in files[:-size]:
            file_path = os.path.join(topic_dir,file)
            document = open(file_path,encoding='latin-1').readlines()
            train_docs[topic].append(Document(topic,document[0],document[1:]).document_terms())
        for file in files[-size:]:
            file_path = os.path.join(topic_dir,file)
            document = open(file_path,encoding='latin-1').readlines()
            test_docs.append(Document(topic,document[0],document[1:]))
            
    
    for topic in topics:
        print(topic,': ',len(train_docs[topic]))
    
    print('Test Size: ',len(test_docs))
    
    for topic in topics:
        train_docs[topic] = prune_terms(train_docs[topic],2)
    
    k_means = KMeans(topics)
    k_means.train(train_docs)
    k_means.print_top_docs()
    
    correct = 0
    for test in test_docs:
        (cluster,score) = k_means.assigned_cluster(test.document_terms())
        if test.topic == cluster:
            correct += 1
    
    print('No of Correct: ', correct, ' Out of ', len(test_docs))
    print('Accuracy: ', correct * 1.0 / len(test_docs))'''
    
if __name__ == '__main__':
    main()