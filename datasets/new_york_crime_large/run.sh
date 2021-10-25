mkdir data
cd data

curl -L https://data.cityofnewyork.us/api/views/qgea-i56i/rows.csv?accessType=DOWNLOAD  --output NYPD_Complaint_Data_Historic.csv 

cd ../

#clean data
python clean.py


