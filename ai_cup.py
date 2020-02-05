import pandas as pd
import random
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import glob
import json
import random
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
import numpy as np
import scipy
from lib import process_source_folder, get_intersection_vocabulary, plot_roc_curve

#Read testing data 
#store all traning data in a hash table
raw_traning_data= {}
raw_testing_data ={}



current_path = os.path.abspath('')
traning_set_path = os.path.join(current_path,"./dataset/training")
print (f'read training data from {traning_set_path}')
source_folder_pattern = f'{traning_set_path}/*'
source_folders = glob.glob(source_folder_pattern)
for source_folder in source_folders:
    print (source_folder)
    theme = os.path.basename(source_folder)
    print (theme)
    raw_traning_data[theme] = process_source_folder(source_folder)

for theme in raw_traning_data:
    print (f'{theme} data count: {len (raw_traning_data[theme])}')
    

intersection_vocabulary = get_intersection_vocabulary(raw_traning_data,raw_traning_data)
print(f'len of intersection_vocabulary :{len(intersection_vocabulary)}')

vectorizer = CountVectorizer(decode_error='ignore',vocabulary=intersection_vocabulary)

x_traning = None
y_traning = None
for theme in raw_traning_data:
    data = raw_traning_data[theme]
    x_data =  vectorizer.transform(data['text']).toarray()
    y_data = list(data['is_rumour'])
    if (x_traning is None):
        x_traning=x_data
    else:
        x_traning =np.concatenate([x_traning,x_data])
        pass
    if (y_traning is None):
        y_traning=y_data
    else:
        y_traning+=y_data
#transform to sparse matrix to incrase speed, somehow numpy array is slower
x_traning = scipy.sparse.csr_matrix(x_traning)

#balance the traning set 
x_traning_true = []
y_traning_true =  []
x_traning_false =  []
y_traning_false = []
for x,y in zip(x_traning,y_traning):
    if(y==1):
        x_traning_true.append(x.toarray())
        y_traning_true.append(y)
    else:
        x_traning_false.append(x.toarray())
        y_traning_false.append(y)
min_len = min(int(len(x_traning_true)),len(x_traning_false))
x_traning_balanced = np.concatenate(x_traning_true[:int(min_len)] +  x_traning_false[:int(min_len)])
y_traning_balanced = np.array(y_traning_true[:int(min_len)] +  y_traning_false[:int(min_len)])
#transform to sparse matrix to incrase speed, somehow numpy array is slower
x_traning_balanced = scipy.sparse.csr_matrix(x_traning_balanced)
print(x_traning_balanced.shape)
print(y_traning_balanced.shape)

#transform to sparse matrix to incrase speed, somehow numpy array is slower
x_traning_balanced = scipy.sparse.csr_matrix(x_traning_balanced)

#Train model
model = RandomForestClassifier(n_estimators=100)
score = cross_validate(model,x_traning_balanced,y_traning_balanced,cv=5,scoring="accuracy")
model.fit(x_traning_balanced,y_traning_balanced)
print(f'Cross validation score: {np.mean(score["test_score"])}')


#Read testing data
testing_set_path = os.path.join(current_path,"./dataset/testing")
print (f'read training data from {testing_set_path}')
source_folder_pattern = f'{testing_set_path}/*'
source_folders = glob.glob(source_folder_pattern)
for source_folder in source_folders:
    print (source_folder)
    theme = os.path.basename(source_folder)
    print (theme)
    raw_testing_data[theme] = process_source_folder(source_folder)

for theme in raw_testing_data:
    print (f'{theme} data count: {len (raw_testing_data[theme])}')

x_testing = None
y_testing = None
for theme in raw_testing_data:
    data = raw_testing_data[theme]
    x_data =  vectorizer.transform(data['text']).toarray()
    y_data = list(data['is_rumour'])
    if (x_testing is None):
        x_testing=x_data
    else:
        x_testing =np.concatenate([x_testing,x_data])
        pass
    if (y_testing is None):
        y_testing=y_data
    else:
        y_testing+=y_data
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