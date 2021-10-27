mkdir -p data
cd data

# Download NYPD compaint data
curl -L https://data.cityofnewyork.us/api/views/qgea-i56i/rows.csv?accessType=DOWNLOAD  --output NYPD_Complaint_Data_Historic.csv 

# Download NYC boundary
curl -L "https://data.cityofnewyork.us/api/geospatial/tqmj-j8zm?method=export&format=Shapefile" -o nyc_borough.zip

mkdir Borough_Boundaries
mv nyc_borough.zip Borough_Boundaries
cd Borough_Boundaries
unzip nyc_borough.zip

# shapefile has a unique id attached, rename for convience
mv *.dbf nyc.dbf
mv *.shp nyc.shp
mv *.shx nyc.shx
mv *.prj nyc.prj

cd ../

cd ../


#clean data
python clean.py


