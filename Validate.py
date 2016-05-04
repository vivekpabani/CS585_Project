'''
Created on Apr 17, 2016
@author: anup
'''
from collections import defaultdict
from collections import Counter
from classify import KMeans
import math
import numpy

class validate(object):
    
    def prune_terms(self,docs,min_df=3):
        
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

    def validate(self,all_docs,k=0.2,prune_fact=2):
        
        documents = defaultdict(lambda: [])
        topics = set()
        
        for doc in all_docs:
            documents[doc.topic].append(doc)
            topics.add(doc.topic)
        
        start_cnt = Counter()
        step_cnt = Counter()
        end_cnt = Counter()
        
        for topic in topics:
            start_cnt[topic] = 0
            end_cnt[topic] = step_cnt[topic] = int(len(documents[topic]) * k )
        
        confusionmtrx = defaultdict(lambda: Counter())
        accuracy = []
        for i in range(int(math.ceil(1.0/k))):
            no_of_train_docs = 0
            train_docs = defaultdict(lambda: [])
            test_docs = list()
            print('****************************Performing Iteration:',i,' **********************************')
            for topic in topics:
                train = documents[topic][0:start_cnt[topic]]
                train += documents[topic][end_cnt[topic]:]
                test = documents[topic][start_cnt[topic]:end_cnt[topic]]
                for doc in train:
                    train_docs[topic].append(doc.document_terms())
                for doc in test:
                    test_docs.append(doc)
                no_of_train_docs += len(train)
                #print('Topic:',topic,'total docs:',len(documents[topic]),'train size:', len(train))
                #print('Testing offset - start:',start_cnt[topic],' end:',end_cnt[topic])
                start_cnt[topic] += step_cnt[topic]
                end_cnt[topic] += step_cnt[topic]
            print('Training Size:',no_of_train_docs)
            print('Testing Size:',len(test_docs))
            print('Pruning Terms which do not occur in at-least 2 documents')
            for topic in topics:
                train_docs[topic] = self.prune_terms(train_docs[topic],2)
            
            k_means = KMeans(topics)
            k_means.train(train_docs)
            
            correct = 0
            for test in test_docs:
                (cluster,score) = k_means.assigned_cluster(test.document_terms())
                if test.topic == cluster:
                    correct += 1
                else:
                    confusionmtrx[test.topic][cluster] += 1
            
            print('No of Correct:', correct, ' Out of ', len(test_docs))
            accuracy.append((correct * 1.0 / len(test_docs)))
            print('Accuracy:', accuracy[i])
            print('*************************************************************************************')
        
        print('*************************************************************************************')
        print('Overall Accuracy:',numpy.average(accuracy))
        print('*************************************************************************************')
        print('***********************Confusion Matrix**********************************************')
        for key in confusionmtrx.keys():
            (topic,cnt) = confusionmtrx[key].most_common(1)[0]
            print(key,'<------->',topic,' confused ',cnt,' Number of Times')
        print('*************************************************************************************')