import tensorflow as tf
import numpy as np
from lib import  get_intersection_vocabulary,get_union_vocabulary, plot_roc_curve,balance_data
from load_data_set import load_data
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import roc_curve, auc, classification_report
from tf_model import get_nn_model


#Read data set
raw_traning_data,raw_testing_data = load_data()

#keep only one topic
#topics = raw_traning_data['topic'].unique()
#raw_traning_data = raw_traning_data[raw_traning_data['topic'] ==topics[0]]

# #balance the traning and testing set 
raw_traning_data = balance_data(raw_traning_data,'is_rumour')
#raw_testing_data = balance_data(raw_testing_data,'is_rumour')
vocabularies = get_intersection_vocabulary(raw_traning_data)
print(f'len of vocabularies :{len(vocabularies)}')

vectorizer = CountVectorizer(decode_error='ignore',vocabulary=vocabularies)

x_traning = vectorizer.transform(raw_traning_data['text']).toarray()
y_traning = np.asarray(raw_traning_data['is_rumour'])
x_testing = vectorizer.transform(raw_testing_data['text']).toarray()
y_testing = np.asarray(raw_testing_data['is_rumour'])

print(type(x_traning))
print(type(y_testing))
#Train model
model = get_nn_model(x_traning.shape[1])
model =get_nn_model(x_traning.shape[1])
history = model.fit(x_traning, y_traning, epochs=30, batch_size=75)
#Test Model
y = model.predict(x_testing)
y_testing_predict = list(map(lambda x:0 if x[0]<0.5 else 1,y))
y_score = list(map(lambda x:  x[0],y))
print(y_testing_predict)
y_testing_predict_probabilities = model.predict_proba(x_testing)
print (classification_report(y_testing, y_testing_predict))

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_testing, y_score)

# Plotting
plot_roc_curve(false_positive_rate,true_positive_rate)