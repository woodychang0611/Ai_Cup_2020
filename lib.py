
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
import matplotlib.pyplot as plt

#process tweet folder
#e.g. .\training\charliehebdo-all-rnr-threads\rumours\552783238415265792
def process_tweet(folder):
    tweets_id = os.path.basename(folder)
    source_tweets = os.path.join(folder,f'./source-tweets/{tweets_id}.json')
    with open(source_tweets) as json_file:
        data = json.load(json_file)
        text = data['text']
        topic = os.path.dirname(os.path.realpath(os.path.join(folder,".\.."))).split('\\')[-1]
    hash = {'tweets_id':tweets_id,'text':text}
    annotation = os.path.join(folder,'annotation.json') 
    with open(annotation) as json_file:
        data = json.load(json_file)
        label_map ={'is_rumour':1,'rumour':1,'nonrumour':0,'unclear':0}
        hash['is_rumour'] = label_map[data['is_rumour']]
        hash['topic']=topic
    return hash

#process source folder 
#e.g. .\training\charliehebdo-all-rnr-threads

def process_source_folder(folder):
    tweets_source_folder_pattern = f'{folder}/*/*/*'
    tweets_folders = glob.glob(tweets_source_folder_pattern)
    return pd.DataFrame(map(process_tweet,tweets_folders[:]))

def get_vocabulary_sets(raw_data):
    vocabulary_sets = {}
    for topic in raw_data['topic'].unique():
        vectorizer = CountVectorizer(decode_error='ignore')
        vectorizer.fit(raw_data[raw_data['topic']==topic]['text'])
        vocabulary_set = set(vectorizer.vocabulary_.keys())
        vocabulary_sets[topic] = vocabulary_set
        print (f'count of vocabulary in {topic}: {len(vocabulary_sets[topic])}')
    return vocabulary_sets  

def get_intersection_vocabulary(raw_data):
    vocabulary_sets = get_vocabulary_sets(raw_data)
    #intersection
    intersection_vocabulary = set.intersection(*map(lambda key:vocabulary_sets[key],vocabulary_sets))
    return intersection_vocabulary

def get_union_vocabulary(raw_data):
    vocabulary_sets = get_vocabulary_sets(raw_data)
    #union
    union_vocabulary = set.union(*map(lambda key:vocabulary_sets[key],vocabulary_sets))
    return union_vocabulary

def plot_roc_curve(false_positive_rate,true_positive_rate):
    # Plotting
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate, c='navy', label=('AUC-'+'= %0.2f'%roc_auc))
    plt.legend(loc='lower right', prop={'size':8})
    plt.plot([0,1],[0,1], color='lightgrey', linestyle='--')
    plt.xlim([-0.05,1.0])
    plt.ylim([0.0,1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def balance_data(data,col_name):
    #balance the traning set 
    min_size = min (data.groupby([col_name]).size())
    dfs = []
    for value in data[col_name].unique():
        df= pd.DataFrame.sample(data[data[col_name]==value],n=min_size,replace=False,random_state=1)
        dfs.append(df)
    return pd.concat(dfs)

