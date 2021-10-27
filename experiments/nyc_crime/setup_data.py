"""
NYC-CRIME data setup.

See datasets/new_york_crime for raw data installation.

"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split

import geopandas as gpd

import json 

from loguru import logger

import matplotlib.pyplot as plt
from matplotlib import animation

from matplotlib.colors import Normalize

def weeks_between_dates(start, end):
    x = pd.to_datetime(end) - pd.to_datetime(start)
    return int(x / np.timedelta64(1, 'W'))

#========================== Settings ==========================

SEED = 2
TRAIN_SPLIT=80 #80-20% train test split
NUM_FOLDS=5
FOLDS = list(range(NUM_FOLDS))
#BIN_SIZES=[int(365/2), 30, 30]
date_start='2010-01-01'
date_end='2013-01-1'
BIN_SIZES=[weeks_between_dates(date_start, date_end), 40, 40]
PD_DESC_FILTER = 'ASSAULT 3'
dataset_folder_root = '../../datasets/'


#========================== Helper functions ==========================

def get_data_file_names(config):
    """ Used by the models to find the correct pickles for a specific fold. """
    fold=config['fold']
    return f'train_data_{fold}.pickle', f'pred_data_{fold}.pickle', f'raw_data_{fold}.pickle'

def get_sub_dataframe(data_df, boundaries_gdf, date_start, date_end, bin_sizes):    
    """ Returns count data inside the new york boroughs. """
    
    df = data_df.copy()
    #get data related only to PD_DESC_FILTER
    if PD_DESC_FILTER is not None:
        df = df[df['PD_DESC'] == PD_DESC_FILTER].copy()


    #get data only within the date range
    df = df[(df['datetime'] >= date_start) & (df['datetime'] < date_end)] 
    X_raw = np.array(df[['epoch', 'Longitude', 'Latitude']])

    
    #calculate the count data using binning defined bin_sizes
    X, Y, actual_bin_sizes = utils.center_point_discretise_grid(
        X_raw, 
        bin_sizes=bin_sizes
    
    )
    print('time: ', X[X[:, 0] == X[0, 0], :].shape)
    print('unique: ', np.unique(X[:, 1:], axis=0).shape)
        
    #convert the count data to a geodataframe so it can be merged with the boundaries
    data = pd.DataFrame(
        np.hstack([X, Y]), 
        columns=['epoch', 'Longitude', 'Latitude', 'Y']
    )
    
    data_gdf = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data.Longitude, data.Latitude),
        crs='EPSG:4326'
    )
    data_gdf.to_crs('EPSG:4326')

    sub_gdf = data_gdf
    col =  sub_gdf.apply(
        lambda row: utils.lat_lon_to_polygon_buffer(row, actual_bin_sizes),
        axis=1
    )
    sub_gdf['polygon'] = col
    sub_gdf = sub_gdf.set_geometry('polygon')


    #merge and remove all data that is not in the new york boundaries
    #this causes duplicates due to one cell belonging to multiple boroughs
    sub_gdf = gpd.sjoin(
        sub_gdf,
        boundaries_gdf, 
        how='inner', 
        op='intersects'
    )

    sub_gdf.drop_duplicates(subset=['epoch', 'Longitude', 'Latitude'], inplace=True, ignore_index=True)

    print('Number of spatial bins: ', sub_gdf[sub_gdf['epoch']==sub_gdf['epoch'][0]].shape)

    
    return sub_gdf, actual_bin_sizes

def load_data():
    """ Load raw data locally. """

    def get_dataframe(root, file_name):
        raw_data = data_root + file_name
        df = pd.read_csv(raw_data, low_memory=False)
        return df

    data_root = f'{dataset_folder_root}/new_york_crime_large/'
    boundary_data_root = f'{dataset_folder_root}/new_york_crime_large/'
    df = get_dataframe(data_root, 'data/cleaned_nyc_crime.csv')
    boundaries_gdf = gpd.read_file(f'{boundary_data_root}/data/Borough_Boundaries/nyc.shp')

    return data_root, df, boundaries_gdf


if __name__ == "__main__":
    import sys
    sys.path.append('../')
    from utils import normalise_df, create_spatial_temporal_grid, datetime_to_epoch, normalise, ensure_timeseries_at_each_locations, un_normalise_df, epoch_to_datetime
    import utils

    logger.info('setting up data')

    print('BINNING: ', BIN_SIZES)

    #ensure correct data structure
    Path("data/").mkdir(exist_ok=True)
    Path("results/").mkdir(exist_ok=True)

    np.random.seed(SEED)

    data_root, df, boundaries_gdf = load_data()
    raw_df = df.copy()

    df, bin_sizes_raw = get_sub_dataframe(df, boundaries_gdf, date_start, date_end, BIN_SIZES)

    print(f'Filtered df: {df.shape} with binsizes: {bin_sizes_raw}')

    #get basic stats
    num_time_points = np.unique(np.array(df['epoch'])).shape[0]
    num_spatial_points = np.array(df['epoch']).shape[0]/num_time_points

    print('num_time_points: ', num_time_points)
    print('num_spatial_points: ', num_spatial_points)

    X_raw = np.array(df[['epoch', 'Longitude', 'Latitude']])
    Y_raw = np.array(df[['Y']])
    
    #although Y are integers we cast to float so that we can set some indices to np.nan
    Y_raw = Y_raw.astype(float)

    #raw x
    N = Y_raw.shape[0]

    for fold in range(NUM_FOLDS):
        _SEED = SEED + fold
        train_indices, test_indices = utils.train_test_split_indices(N, split=TRAIN_SPLIT/100, seed=_SEED)

        #Collect training and testing data
        X_train, Y_train = X_raw.copy(), Y_raw.copy()
        Y_train[test_indices, :] = np.nan #to keep grid structure in X we just mask the testing data in the training set

        X_test, Y_test = X_raw.copy(), Y_raw.copy()
        Y_test[train_indices, :] = np.nan

        X_all = X_raw
        Y_all = Y_raw

        #normalise all data with respect to training data
        X_train_std = np.std(X_train, axis=0)

        X_train_norm = normalise_df(X_train, wrt_to=X_train)
        X_test_norm = normalise_df(X_test, wrt_to=X_train)
        X_all_norm = normalise_df(X_all, wrt_to=X_train)

        # (x1-m)/s - (x2-m)/s = ((x-1)-(x2-m))/s = (x1-x2)/s
        bin_sizes_norm = bin_sizes_raw/X_train_std

        print('---')
        print('X_train: ', X_train_norm.shape)
        print(np.nanmean(Y_train), np.nanstd(Y_train))
        print('Y_train: ', Y_train.shape, ' Non nans: ', np.sum(np.logical_not(np.isnan(Y_train))))
        print('X_test: ', X_test_norm.shape)
        print('Y_test: ', Y_test.shape, ' Non nans: ', np.sum(np.logical_not(np.isnan(Y_test))))
        print('X_all: ', X_all.shape)
        print('Y_all: ', X_all_norm.shape)
        print('bin_sizes_norm: ', bin_sizes_raw, bin_sizes_norm)

        training_data = {
            'X': X_train_norm, 
            'Y': Y_train, 
            'bin_sizes': bin_sizes_norm
        }

        prediction_data = {
            'test': {
                'X': X_test_norm,
                'Y': Y_test
            },
            'all': {
                'X': X_all_norm,
                'Y': Y_all
            }

        }

        raw_data_dict = {
            'data': {
                'train': {
                    'X': X_train,
                    'Y': Y_train
                },
                'all': {
                    'X': X_raw,
                    'Y': Y_raw,
                    'bin_sizes': bin_sizes_raw
                }
            },
        }

        train_data_name, pred_data_name, raw_data_name = get_data_file_names({'fold': fold})

        with open('data/{train_name}'.format(train_name=train_data_name), 'wb') as file:
            pickle.dump(training_data, file)

        with open('data/{pred_name}'.format(pred_name=pred_data_name), "wb") as file:
            pickle.dump(prediction_data, file)

        with open(f'data/{raw_data_name}', "wb") as file:
            pickle.dump(raw_data_dict, file)

    logger.info('finished')


    


