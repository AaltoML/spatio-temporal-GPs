"""
Downloads air pollution data from http://www.londonair.org.uk adn converts to a csv file.
Will need R and Rscript to run.
"""
import urllib.request
import requests
import urllib.request
from pathlib import Path
from subprocess import call
import os
import pandas as pd
#CONFIG
#year to download data for
YEAR = 2019 
DOWNLOAD_FLAG = True
POLLUTANTS = ["nox","no2","o3","co","pm10_raw","pm10","pm25"]

#Ensure folder structure exists
Path("downloaded_data/").mkdir(exist_ok=True)
Path("downloaded_data/aq_r_data/").mkdir(exist_ok=True)
Path("downloaded_data/aq_csv/").mkdir(exist_ok=True)

def rdata_to_csv(file_rdata, file_csv_output, rdata_df):
    call(['Rscript','--vanilla','data_processing/rdata_to_csv.r',file_rdata,file_csv_output,rdata_df])

def rdata_to_csv_for_aq(file_rdata, file_csv_output, rdata_df):
    call(['Rscript','--vanilla','data_processing/rdata_to_csv_for_aq.r',file_rdata,file_csv_output,rdata_df])

#========== DOWNLOAD LAQN SENSOR SITES ============
if not os.path.isfile('downloaded_data/laqn_sites.RData'): 
    sites_url = 'http://www.londonair.org.uk/r_data/sites.RData'
    urllib.request.urlretrieve(sites_url, 'downloaded_data/laqn_sites.RData')

#Convert Rdata to csv
if not os.path.isfile('downloaded_data/laqn_sites.csv'): 
    rdata_to_csv('downloaded_data/laqn_sites.RData', 'downloaded_data/laqn_sites.csv', 'sites')

    if not os.path.isfile('downloaded_data/laqn_sites.csv'):
        raise RuntimeError('Could not generate laqn_sites.csv')

laqn_sites = pd.read_csv('downloaded_data/laqn_sites.csv', delimiter=';')

laqn_site_codes = laqn_sites['SiteCode']

#========== DOWNLOAD LAQN SENSOR SITES ============
#download Rdata
if DOWNLOAD_FLAG:
    count = 0
    for site in laqn_site_codes:
        filename = "{site}_{year}.Rdata".format(site=site,year=YEAR)

        #check if site already downloaded
        if not os.path.isfile('downloaded_data/aq_r_data/{filename}'.format(filename=filename)): 

            url = "http://www.londonair.org.uk/r_data/{filename}".format(filename=filename)
            dir_to_save = 'downloaded_data/aq_r_data/{filename}'.format(filename=filename)

            #check that file exists on server
            resp = requests.head(url)
            if resp.status_code is 200:
                #file exists
                count = count + 1
                print('Downloading {site} - {year}'.format(site=site, year=YEAR))
                while True:
                    try:
                        urllib.request.urlretrieve(url, dir_to_save)
                        break
                    except Exception as e:
                        print('Exception on site %s'%site)
                        print(e)
                        sleep(10)
                        print('trying again')
                        continue
                    except:
                        print('problem on site %s'%site)
                        sleep(10)

                print('Downloaded {site} - {year}'.format(site=site, year=YEAR))
            else:
                print('File not found: ', url)

#========== CONVERT RDATA TO CSV ============
for site in laqn_site_codes:
    filename = 'downloaded_data/aq_r_data/' + "{site}_{year}.Rdata".format(site=site,year=YEAR)
    filename_csv ='downloaded_data/aq_csv/' + '{site}_{year}.csv'.format(site=site, year=YEAR)
    if os.path.isfile(filename) and not  os.path.isfile(filename_csv):
        rdata_to_csv_for_aq(filename, filename_csv, 'x')
    else:
        print('Does not exists: ', filename)


#========== MERGE CSVs INTO ONE FILE ============
total_df = pd.DataFrame(columns=['site', 'date']+POLLUTANTS)
for site in laqn_site_codes:
    filename = "{site}_{year}.Rdata".format(site=site,year=YEAR)
    filename_csv ='downloaded_data/aq_csv/' + '{site}_{year}.csv'.format(site=site, year=YEAR)
    if os.path.isfile(filename_csv) and os.path.getsize(filename_csv) > 0:
        print(filename_csv)
        df = pd.read_csv(filename_csv, sep=';')
        print(filename_csv, ' , ',df.shape)
        result_df = df[['site', 'date']]
        for p in POLLUTANTS:
            if p in list(df.columns):
                result_df[p] = df[p]
            else:
                result_df[p] = None
        total_df = total_df.append(result_df)

total_df.to_csv('downloaded_data/aq_data.csv', index=False)
print(total_df.shape)
