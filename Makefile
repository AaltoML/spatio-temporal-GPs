.PHONY: all data experiments

data:
	@echo 'Generating Data'
	@echo 'Downloading london shape file'
	#cd datasets/london && sh ./run.sh 
	@echo 'Downloading laqn air pollution data'
	cd datasets/london_air_pollution && sh ./run.sh 
	@echo 'Downloading new york crime data'
	#cd datasets/new_york_crime_large && sh ./run.sh 

experiments:
	@echo 'Running all experiments'
