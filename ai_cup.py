import os
import numpy as np
import pandas as pd
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.utils import shuffle
from lib import process_source_folder, get_intersection_vocabulary,get_union_vocabulary, plot_roc_curve,balance_data
from load_data_set import load_data

raw_testing_data,raw_traning_data = load_data()


#keep only one topic
topics = raw_traning_data['topic'].unique()
raw_traning_data = raw_traning_data[raw_traning_data['topic'] ==topics[0]]

# #balance the traning set 
raw_traning_data = balance_data(raw_traning_data,'is_rumour')
vocabularies = get_intersection_vocabulary(raw_traning_data)
print(f'len of vocabularies :{len(vocabularies)}')

vectorizer = CountVectorizer(decode_error='ignore',vocabulary=vocabularies)

x_traning = vectorizer.transform(raw_traning_data['text']).toarray()
y_traning = list(raw_traning_data['is_rumour'])

#Train model
model = RandomForestClassifier(n_estimators=100)
score = cross_validate(model,x_traning,y_traning,cv=5,scoring="accuracy")
model.fit(x_traning,y_traning)
print(f'Cross validation score: {np.mean(score["test_score"])}')

x_testing = vectorizer.transform(raw_testing_data['text']).toarray()
y_testing = list(raw_testing_data['is_rumour'])
#transform to sparse matrix to incrase speed, somehow numpy array is slower
x_testing = scipy.sparse.csr_matrix(x_testing)

#Test Model
y_testing_predict = model.predict(x_testing)
y_testing_predict_probabilities = model.predict_proba(x_testing)

print (classification_report(y_testing, y_testing_predict))
y_score = y_testing_predict_probabilities[:,1]
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_testing, y_score)

# Plotting
plot_roc_curve(false_positive_rate,true_positive_rate)

exit(0)