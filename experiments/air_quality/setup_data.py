"""
This file creates the required folder structure and setups the experiment data

This reads the london air quality data from:
    ../../datasets/london_air_pollution

and constructs train-test splits stored in pickles inside ./data/
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

from dateutil import parser

import json 
from loguru import logger

try:
    import geopandas as gpd
    from geopandas.tools import sjoin
except ModuleNotFoundError:
    print('Geopandas not found')
    pass


#============================== Experiment Settings ==============================

NUM_FOLDS = 5
FOLDS = list(range(NUM_FOLDS))

#1 day, 2 days, 1 week
SPECIES=['pm10']
PREDICTION_SITE = 'HK6' #HK6 = Hackney - Old Street
DATE_START='2019/01/01'
DATE_END='2019/04/01'
TRAIN_SPLIT=0.8 #use 80% of data for training
datasets_folder_root = '../../datasets/'

SEED = 3

def get_data_file_names(config):
    fold = config['fold']

    return f'train_data_{fold}.pickle', f'pred_data_{fold}.pickle', f'raw_data_{fold}.pickle'

#============================== Create Experiment Data ==============================
# only create data when file run from terminal so that the experiment settings can be imported

def load_data():
    """ Locally load the raw air quality dataset. """

    def get_data(root):
        raw_data = pd.read_csv(root+'london_air_pollution/downloaded_data/aq_data.csv')
        sites_df = pd.read_csv(root+'london_air_pollution/downloaded_data/laqn_sites.csv', sep=';')
        return raw_data, sites_df

    # Get local dataset
    raw_data, sites_df = get_data(datasets_folder_root)

    return raw_data, sites_df

def get_boundary_gdf():
    """ Load the london boundary locally. """

    root = datasets_folder_root
    #try running locally
    data_root = 'london/london_shp/statistical-gis-boundaries-london/ESRI/'
    boundary_gdf = gpd.read_file(root+data_root+'LSOA_2011_London_gen_MHW.shp')

    boundary_gdf = boundary_gdf.to_crs({'init': 'epsg:4326'}) 
    return boundary_gdf


def filter_sites_not_in_london(sites_df):
    """ The LAQN datasets includes sensors that are not in London. Remove them. """

    # Approximate bounding box of London
    london_box = [
        [51.279, 51.684], #lat
        [-0.533, 0.208] #lon
    ]

    # Remove sites that do not lie within the bounding box
    sites_df = sites_df[(sites_df['Latitude'] > london_box[0][0]) & (sites_df['Latitude'] < london_box[0][1])]
    sites_df = sites_df[(sites_df['Longitude'] > london_box[1][0]) & (sites_df['Longitude'] < london_box[1][1])]

    boundary_gdf = get_boundary_gdf()

    sites_gdf = gpd.GeoDataFrame(
        sites_df, 
        geometry=gpd.points_from_xy(sites_df.Longitude, sites_df.Latitude)
    )
    sites_gdf.set_crs(epsg=4326, inplace=True)


    # Remove sites that do not lie within the boundary
    filtered_sites = sjoin(sites_gdf, boundary_gdf, how='inner',  op='within')

    return filtered_sites

def drop_locations_with_low_data(raw_data):
    """ Some sensors have been badly calibrates and have lot of missing data. We remove these with less that 40% data. """
    raw_data['is_null'] = raw_data[SPECIES[0]].isnull().astype(int)
    raw_data['one'] = 1

    _df = raw_data.groupby('site').sum()

    #one is the total number of observations
    #is null is the number that were null
    _df = _df[_df['is_null'] < _df['one'] * 0.4]


    raw_data = raw_data.merge(_df, left_on='site', right_on='site',  suffixes=[None, '_y'])

    return raw_data


def clean_data(raw_data, sites_df):
    """
    To clean the data we:
        1) Remove sites not in London
        2) Remove sensors with lots of missing values
        3) Convert datetimes to unix epoch
    """
    sites_df = filter_sites_not_in_london(sites_df)

    sites = sites_df['SiteCode']

    #merge spatial infomation to data
    raw_data = raw_data.merge(sites_df, left_on='site', right_on='SiteCode')

    raw_data = drop_locations_with_low_data(raw_data)


    #convert to datetimes
    raw_data['date'] = pd.to_datetime(raw_data['date'])
    raw_data['epoch'] = datetime_to_epoch(raw_data['date'])


    return raw_data

def get_grid(time_point=0):
    """ Create a spatial grid across London. Used for visualisations. """

    london_box = [
        [51.279, 51.684], #lat
        [-0.533, 0.332] #lon
    ]

    boundary_gdf = get_boundary_gdf()

    grid = utils.create_spatial_temporal_grid([time_point], london_box[0][0], london_box[0][1], london_box[1][0], london_box[1][1], 100, 100)

    grid_df = pd.DataFrame(grid, columns=['epoch', 'lat', 'lon'])
    grid_df = gpd.GeoDataFrame(
        grid_df, 
        geometry=gpd.points_from_xy(grid_df.lon, grid_df.lat)
    )
    grid_df.set_crs(epsg=4326, inplace=True)

    filtered_grid = sjoin(grid_df, boundary_gdf, how='inner',  op='within')

    if False:
        fig = plt.figure()
        ax = plt.gca()
        #boundary_gdf.plot(ax=ax)
        filtered_grid.plot(ax=ax)
        print(filtered_grid.shape)
        plt.show()

        exit()

    return filtered_grid



if __name__ == "__main__":
    logger.info('starting')

    # This file is also imported to extract the experiment information. To avoid import issues they are included here.
    import sys
    sys.path.append('../')
    from utils import normalise_df, create_spatial_temporal_grid, datetime_to_epoch, normalise, ensure_timeseries_at_each_locations, un_normalise_df, epoch_to_datetime, pad_with_nan_to_make_grid
    import utils 

    #ensure correct data structure
    Path("data/").mkdir(exist_ok=True)
    Path("results/").mkdir(exist_ok=True)

    #clean data and get prediction site location
    raw_data, sites_df = load_data()
    prediction_site = sites_df[sites_df['SiteCode'] == PREDICTION_SITE].iloc[0]
    data_df = clean_data(raw_data, sites_df)

    print('Number of sites: ', pd.unique(data_df['SiteCode']).shape)

    print(
        'Amount of data in jan: ', 
        data_df[(data_df['date'] >= '2019/01/01') & (data_df['date'] < '2019/02/01')].shape

    )

    #get data in daterange
    data_df = data_df[(data_df['date'] >= DATE_START) & (data_df['date'] < DATE_END)]

    #data_df may have missing oberservations, for the state space we require X, Y to be on a grid, with missings denoted by nans
    X = np.array(data_df[['epoch', 'Latitude', 'Longitude']])
    Y = np.array(data_df[SPECIES])

    #remove duplicated data
    u, unique_idx = np.unique(X, return_index=True, axis=0)
    X = X[unique_idx, :]
    Y = Y[unique_idx, :]

    # For the filtering methods to work we need a full spatio-temporal grid
    X_raw, Y_raw = pad_with_nan_to_make_grid(X.copy(), Y.copy())

    N = X.shape[0]

    print('Y: ', Y.shape, ' X_raw: ', X_raw.shape)
    print('statst: ', np.nanmean(Y_raw), np.nanmin(Y_raw), np.nanmax(Y_raw))

    #extract prediction timeseries
    prediction_idx = (X_raw[:, 1] == prediction_site['Latitude']) &  (X_raw[:, 2] == prediction_site['Longitude'])
    timeseries_x_raw = X_raw[prediction_idx, :]
    timeseries_y_raw = Y_raw[prediction_idx, :]

    slice_epoch = utils.datetime_str_to_epoch('2019/01/05 10:00:00')

    spatial_grid = get_grid(slice_epoch)
    spatial_grid_x = np.array(spatial_grid[['epoch', 'lat', 'lon']])
    spatial_grid_y = None
    print('spatial_grid: ', spatial_grid.shape)

    print('timeseries_x_raw: ', timeseries_x_raw.shape, ' timeseries_y_raw: ', timeseries_y_raw.shape)

    if False:
        print(Y_raw)
        print(timeseries_x_raw)
        plt.figure()
        plt.scatter(timeseries_x_raw[:, 0], timeseries_y_raw)
        plt.show()
        exit()

    print('number of timesteps: ', np.unique(X[:, 0]).shape)

    #construct test-train splits
    # train test splits are constructed by construction a random permutation (constrolled through seed)
    #  and then splitting this is into train-test data as specificed by TRAIN_SPLIT
    for i, fold in enumerate(FOLDS):
        train_indices, test_indices = utils.train_test_split_indices(N, split=TRAIN_SPLIT, seed=(SEED+i))

        _config = {'fold': fold}

        train_name, pred_name, raw_name = get_data_file_names(_config)

        #Collect training and testing data
        X_train, Y_train = X_raw.copy(), Y_raw.copy()
        Y_train[test_indices, :] = np.nan #to keep grid structure in X we just mask the testing data in the training set

        X_test, Y_test = X_raw.copy(), Y_raw.copy()
        Y_test[train_indices, :] = np.nan

        X_all = X_raw
        Y_all = Y_raw

        #normalise all data with respect to training data
        X_train_norm = normalise_df(X_train, wrt_to=X_train)
        X_test_norm = normalise_df(X_test, wrt_to=X_train)
        X_all_norm = normalise_df(X_all, wrt_to=X_train)
        timeseries_x_norm = normalise_df(timeseries_x_raw, wrt_to=X_train)
        spatial_grid_x_norm = normalise_df(spatial_grid_x, wrt_to=X_train)

        print('---')
        print('X_train: ', X_train_norm.shape)
        print('X_all: ', X_all.shape)
        print(np.nanmean(Y_train), np.nanstd(Y_train))
        print('Y_train: ', Y_train.shape, ' Non nans: ', np.sum(np.logical_not(np.isnan(Y_train))))
        print('X_test: ', X_test_norm.shape)
        print('Y_test: ', Y_test.shape, ' Non nans: ', np.sum(np.logical_not(np.isnan(Y_test))))

        training_data = {
            'X': X_train_norm, 
            'Y': Y_train, 
        }

        prediction_data = {
            'test': {
                'X': X_test_norm,
                'Y': Y_test

            },
            'all': {
                'X': X_all_norm,
                'Y': Y_all
            },
            'timeseries': {
                'X': timeseries_x_norm,
                'Y': timeseries_y_raw
            },
            'grid': {
                'X': spatial_grid_x_norm,
                'Y': None
            }
        }


        raw_data = {
            'all': {
                'X': X_all,
                'Y': Y_all,
            },
            'timeseries': {
                'X': timeseries_x_raw,
                'Y': timeseries_y_raw,
            },
            'grid': {
                'X': spatial_grid_x,
                'Y': None,
            }
        }

        with open(f'data/{train_name}', 'wb') as file:
            pickle.dump(training_data, file)

        with open(f'data/{pred_name}', "wb") as file:
            pickle.dump(prediction_data, file)


        with open(f'data/{raw_name}', "wb") as file:
            pickle.dump(raw_data, file)


    logger.info('seting up data finished')

