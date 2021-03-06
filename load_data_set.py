import os
import pandas as pd
from lib import process_source_folder
from lib import  plot_roc_curve,balance_data

CURRENT_PATH = os.path.abspath('')
TRANING_SET_PATH = os.path.join(CURRENT_PATH,"./dataset/training")
TESTING_SET_PATH = os.path.join(CURRENT_PATH,"./dataset/testing")
SAVED_RAW_TRANING_DATA = os.path.join(CURRENT_PATH,"./raw_traning_data.csv")
SAVED_RAW_TESTING_DATA = os.path.join(CURRENT_PATH,"./raw_testing_data.csv")

def load_data():
    #read traning data
    if (os.path.exists(SAVED_RAW_TRANING_DATA)):
        print (f'Load traning data from saved file {SAVED_RAW_TRANING_DATA}')
        raw_traning_data = pd.read_csv(SAVED_RAW_TRANING_DATA)
    else:
        print (f'read training data from {TRANING_SET_PATH}')
        raw_traning_data = process_source_folder(TRANING_SET_PATH)
        raw_traning_data.to_csv(SAVED_RAW_TRANING_DATA)
    for topic in raw_traning_data['topic'].unique():
        count = len(raw_traning_data[raw_traning_data['topic']==topic])
        print(f'{topic}: count: {count}')

    #read testing data
    if (os.path.exists(SAVED_RAW_TESTING_DATA)):
        print (f'Load testing data from saved file {SAVED_RAW_TESTING_DATA}')
        raw_testing_data = pd.read_csv(SAVED_RAW_TESTING_DATA)
    else:
        print (f'read testing data from {TESTING_SET_PATH}')
        raw_testing_data = process_source_folder(TESTING_SET_PATH)
        raw_testing_data.to_csv(SAVED_RAW_TESTING_DATA)
    for topic in raw_testing_data['topic'].unique():
        count = len(raw_testing_data[raw_testing_data['topic']==topic])
        print(f'{topic}: count: {count}')
    return raw_traning_data,raw_testing_data