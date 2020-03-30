import csv
import pandas as pd
from utilities.segmentator import Segmentator
from utilities.vanilla import *

def read_file(path, file_name):
    ''' Read ecg information from a file and saves it to a dataframe  '''
    cols_of_interest = [0,1]
    ecg_data = pd.read_csv(f'{path}/{file_name}.csv', usecols=cols_of_interest)
    # drop useless header
    ecg_data = ecg_data.drop(ecg_data.index[0])
    # name columns
    ecg_data.columns = ['time', 'ECG']
    # cast some columns to float
    ecg_data['time'] = ecg_data['time'].astype(float)
    ecg_data['ECG'] = ecg_data['ECG'].astype(float)
    
    return ecg_data
    
def run(ecg_data, file_name, number_of_segments, low_cut):
    """Execute the pipeline that segmentate and extract the features of a ecg signal 
    inputted as a dataframe"""
    sample_rate = detect_sample_rate(ecg_data)
    high_cut = sample_rate/5.0
    order = 7
    data_preparation_pipeline = Pipeline([
        ('filtering', Filter(sample_rate, low_cut, high_cut, order=order)),
        ('feature_detection', Segmentator(number_of_segments, sample_rate)),
        ])

    raw_df, processed_df  = data_preparation_pipeline.fit_transform(ecg_data['ECG'])
    processed_df.reset_index(drop=True, inplace=True)
    raw_df.reset_index(drop=True, inplace=True)
    
    return raw_df, processed_df 


def load_file(path):
    files = []
    with open(path+'/header.txt') as f:
        reader = csv.reader(f)
        for row in reader:
            files.append(row[0])
        
        print("Read data for the following drivers:\n", files[:10])
        return files


def remove_outliers(original_dataset, lower_threshold, upper_threshold, column_names=[]):
    ''' Remove outliers from a dataframe
        Everything above or below these percentiles will be cut off
    '''
    # TODO: add treatment for not numerical columns
    dataset = original_dataset.copy()
    
    if column_names:
        for column in column_names:
            removed_outliers = remove(dataset[column], lower_threshold, upper_threshold)
            # save the indexes of rows that must be removed
            indexes_for_removal = dataset[column][~removed_outliers].index
            # in fact remove outliers from this column 
            #print(indexes_for_removal)
            dataset.drop(indexes_for_removal, inplace=True)
            #print(f'removed {len(indexes_for_removal)} outliers for column {column}')
            #print(f'remaining itens in dataset: {len(dataset)}')
        return dataset
            
    else:
        column_names = list(dataset.columns)
        for column in column_names:
            removed_outliers = remove(dataset[column], lower_threshold, upper_threshold)
            # save the indexes of rows that must be removed
            indexes_for_removal = dataset[column][~removed_outliers].index
            # in fact remove outliers from this column 
            dataset.drop(indexes_for_removal, inplace=True)
        return dataset
    
def remove(series, lower_threshold, upper_threshold):
    ''' Remove outliers from a single pandas Series '''
    # create a boolean mask where False values are the outliers
    removed_outliers = series.between(series.quantile(lower_threshold),
                                      series.quantile(upper_threshold))
    return removed_outliers