.PHONY: all data experiments

data:
	@echo 'Generating Data'
	@echo 'Downloading london shape file'
	cd datasets/london && sh ./run.sh 
	@echo 'Downloading laqn air pollution data'
	cd datasets/london_air_pollution && sh ./run.sh 
	@echo 'Downloading new york crime data'
	cd datasets/new_york_crime_large && sh ./run.sh 
	@echo 'Generating Air Quality train-test splits'
	cd experiments/air_quality && python setup_data.py
	@echo 'Generating NYC train-test splits'
	cd experiments/nyc_crime && python setup_data.py

experiments:
	@echo 'Running Air Quality experiments'
	cd experiments/air_quality && mkdir -p results
	@echo 'Running gpflow model'
	#cd experiments/air_quality/models && python m_gpflow.py 0
	#cd experiments/air_quality/models && python m_gpflow.py 1
	#cd experiments/air_quality/models && python m_gpflow.py 2
	#cd experiments/air_quality/models && python m_gpflow.py 3
	#cd experiments/air_quality/models && python m_gpflow.py 4
	@echo 'Running ski model'
	#cd experiments/air_quality/models && python m_ski.py 0
	#cd experiments/air_quality/models && python m_ski.py 1
	#cd experiments/air_quality/models && python m_ski.py 2
	#cd experiments/air_quality/models && python m_ski.py 3
	#cd experiments/air_quality/models && python m_ski.py 4
	@echo 'Running NYC experiments'
	cd experiments/nyc_crime && mkdir -p results
	#cd experiments/nyc_crime/models && python m_gpflow.py 0
	#cd experiments/nyc_crime/models && python m_gpflow.py 1
	#cd experiments/nyc_crime/models && python m_gpflow.py 2
	#cd experiments/nyc_crime/models && python m_gpflow.py 3
	#cd experiments/nyc_crime/models && python m_gpflow.py 4




