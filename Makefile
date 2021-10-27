.PHONY: all data experiments

data:
	@echo 'Generating Data'
	@echo 'Downloading london shape file'
	#cd datasets/london && sh ./run.sh 
	@echo 'Downloading laqn air pollution data'
	#cd datasets/london_air_pollution && sh ./run.sh 
	@echo 'Downloading new york crime data'
	#cd datasets/new_york_crime_large && sh ./run.sh 
	@echo 'Generating Air Quality train-test splits'
	#cd experiments/air_quality && python setup_data.py
	@echo 'Generating NYC train-test splits'
	cd experiments/nyc_crime && python setup_data.py

experiments:
	@echo 'Running all experiments'
