#Code from Rocheteau

from pickle import TRUE
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
import os
import argparse

def create_folder(parent_path, folder):
    if not parent_path.endswith('/'):
        parent_path += '/'
    folder_path = parent_path + folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def shuffle_hadms(hadm_ids, seed=2021):
    return shuffle(hadm_ids, random_state=seed)

def process_table(table_name, table, hadm_ids, folder_path):
    table = table[table.index.get_level_values(0).isin(hadm_ids)].copy()
    table.to_csv('{}/{}.csv'.format(folder_path, table_name))
    return

def split_train_test(path, is_test=True, seed=2021, cleanup=False, MIMIC=True):
    labels = pd.read_csv(path + 'preprocessed_labels.csv') ; labels.set_index('hadmid', inplace=True)

    if not MIMIC:
        patients = labels.patientid.unique()
    if MIMIC:
        patients = labels.uniquepid.unique()

    train, test = train_test_split(patients, test_size=0.15, random_state=seed)
    train, val = train_test_split(train, test_size=0.15/0.85, random_state=seed)

    print('==> Loading data for splitting...')
    if not MIMIC:
        if is_test:
            #add a condition here depending on the dataset to get zipped or not dataset
            timeseries = pd.read_csv(path + 'preprocessed_timeseries.csv.gz', compression='gzip', nrows=50000)
        else:
            timeseries = pd.read_csv(path + 'preprocessed_timeseries.csv.gz', compression='gzip')

    if MIMIC: 
        if is_test:
            timeseries = pd.read_csv(path + 'scaled_preprocessed_timeseries.csv', nrows=50000)
        else:
            timeseries = pd.read_csv(path + 'scaled_preprocessed_timeseries.csv')

    timeseries.set_index('hadmid', inplace=True)
    static_data = pd.read_csv(path + 'preprocessed_static.csv') ; static_data.set_index('hadmid', inplace=True)

    if is_test is False and cleanup is True:
        print('==> Removing the unsorted data...')
        os.remove(path + 'preprocessed_timeseries.csv.gz')
        os.remove(path + 'preprocessed_labels.csv')
        os.remove(path + 'preprocessed_static.csv')

    for partition_name, partition in zip(['train', 'test', 'val'], [train, test, val]):
        print('==> Preparing {} data...'.format(partition_name))
        if not MIMIC:
            hadm_ids = labels.loc[labels['patientid'].isin(partition)].index #collect admission ids corresponding to unique patients
        if MIMIC:
            hadm_ids = labels.loc[labels['uniquepid'].isin(partition)].index #collect admission ids corresponding to unique patients            
        print(f'lenght of admission ids: {len(hadm_ids)}')
        folder_path = create_folder(path, partition_name)
        with open(folder_path + '/hadm_ids.txt', 'w') as f:
            for id in hadm_ids:
                f.write("%s\n" % id)
        hadm_ids = shuffle_hadms(hadm_ids, seed=2021)
        for table_name, table in zip(['labels', 'static_data', 'timeseries'],
                                            [labels, static_data, timeseries]):
            process_table(table_name, table, hadm_ids, folder_path)
    return
    
