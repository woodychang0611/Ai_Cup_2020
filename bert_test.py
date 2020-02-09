#https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#1-loading-pre-trained-bert
#https://towardsdatascience.com/fake-news-classification-with-bert-afbeee601f41

import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from load_data_set import load_data
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn import svm
from collections import Counter
import matplotlib.pyplot as plt
from lib import  get_intersection_vocabulary,get_union_vocabulary, plot_roc_curve,balance_data

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#Read data set
raw_traning_data,raw_testing_data = load_data()

#keep only one topic
#topics = raw_traning_data['topic'].unique()
raw_traning_data = raw_traning_data[raw_traning_data['topic'] ==topics[0]]

#balance the traning and testing set 
raw_traning_data = balance_data(raw_traning_data,'is_rumour')
raw_testing_data = balance_data(raw_testing_data,'is_rumour')

def bert_process_text(text):
    text = f'[CLS] {text} [SEP]'
    tokenized_text  = tokenizer.tokenize(text)
    train_tokens_ids  = tokenizer.convert_tokens_to_ids(tokenized_text)
    return train_tokens_ids

bert_ids_data_traning= list(map(lambda t:bert_process_text(t),(list(raw_traning_data['text']))))

ids_sets={}
for topic in raw_traning_data['topic'].unique():
    df = raw_traning_data[raw_traning_data['topic']==topic]
    bert_ids= map(lambda t:bert_process_text(t),(list(df['text'])))
    ids = set.union(*map(lambda d:set(d),bert_ids))
    ids_sets[topic] = ids

unique_ids = list(set.intersection(*map(lambda key:ids_sets[key],ids_sets)))

data_count = len(bert_ids_data_traning)
ids_count = len(unique_ids)

x_traning = np.zeros((data_count,ids_count))
for i in range(0,data_count):
    count_result=Counter(bert_ids_data_traning[i])
    for j in range (0,ids_count):
        if(unique_ids[j]  in count_result):
            x_traning[i,j]=count_result[unique_ids[j]]
print(x_traning)
y_traning = list(raw_traning_data['is_rumour'])

model = RandomForestClassifier(n_estimators=100)
#model = svm.SVC(gamma='auto',probability=True)
score = cross_validate(model,x_traning,y_traning,cv=5,scoring="accuracy")
model.fit(x_traning,y_traning)
print(f'Cross validation score: {np.mean(score["test_score"])}')

#test data
bert_ids_data_testing= list (map (lambda t:bert_process_text(t),(list(raw_testing_data['text']))))
data_count = len(bert_ids_data_testing)
ids_count = len(unique_ids)

x_testing = np.zeros((data_count,ids_count))
for i in range(0,data_count):
    for j in range (0,ids_count):
        if(unique_ids[j] in bert_ids_data_testing[i]):
            x_testing[i,j]=1

y_testing = list(raw_testing_data['is_rumour'])

#testing
y_testing_predict = model.predict(x_testing)
y_testing_predict_probabilities = model.predict_proba(x_testing)

print (classification_report(y_testing, y_testing_predict))
y_score = y_testing_predict_probabilities[:,1]
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_testing, y_score)

# Plotting
plot_roc_curve(false_positive_rate,true_positive_rate)