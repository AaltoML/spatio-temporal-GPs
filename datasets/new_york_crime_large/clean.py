import pandas as pd

import sys
sys.path.append('../../experiments/')
import utils

df = pd.read_csv('data/NYPD_Complaint_Data_Historic.csv')

df['datetime'] = df.apply(
    lambda row: str(row['CMPLNT_FR_DT'])+' '+str(row['CMPLNT_FR_TM']), 
axis=1)

df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
df = df[df['datetime'].isnull()==False] #drop rows with non-valid dates

df['epoch'] = utils.datetime_to_epoch(df['datetime'])

#df = df[(df['datetime'] >= '2014') & (df['datetime'] < '2015')] #only want in 2014 

df = df[df['Latitude'].notna()] #drop rows with nan lat-lons
df = df[df['Longitude'].notna()] #drop rows with nan lat-lons

df.to_csv('data/cleaned_nyc_crime.csv', index=False)

