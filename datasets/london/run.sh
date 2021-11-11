wd=$(dirname "$0")
curl -L https://data.london.gov.uk/download/statistical-gis-boundary-files-london/9ba8c833-6370-4b11-abdc-314aa020d5e0/statistical-gis-boundaries-london.zip --output london_shp.zip
mkdir -p london_shp
mv london_shp.zip london_shp
cd london_shp
unzip london_shp.zip
