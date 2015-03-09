# -*- coding: utf-8 -*-
import csv
import numpy as np
from random import shuffle

data = {}
with open(r'E:\Thesis\ACL\Data\dir_vs_ndir.csv', mode='r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
	  data[row['Sentence']] = row['Label']
    
print "You have", len(data), "data items"	  
data_list = data.items()
shuffle(data_list)
print data_list[0:5]
def data_into_array(text):
  data_y = [item[1] for item in text]
  data_x = [item[0].decode('utf-8',errors='ignore') for item in text]
  target_y = np.array(data_y)    
  return data_x, target_y

data, labels = data_into_array(data_list)

print "Your data is ready to use!"  
print "............................"
print "Starting the classifier"

#################################################################################################
	  
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
#from sklearn.linear_model import SGDClassifier
from sklearn import svm, cross_validation
from sklearn.pipeline import Pipeline

#count_vect = CountVectorizer()
#X_counts = count_vect.fit_transform(data_x)
#tfidf_transformer = TfidfTransformer()
#X_tfidf = tfidf_transformer.fit_transform(X_counts)

text_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', svm.SVC(kernel='linear', C=1))])
                    #SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5))])
                  
scores = cross_validation.cross_val_score(text_clf, data, labels, cv=5, scoring='f1')

print scores