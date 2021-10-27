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
	@echo 'Running bayesnewton model'
	cd experiments/air_quality/models && python m_bayes_newt.py 0 0 0
	cd experiments/air_quality/models && python m_bayes_newt.py 1 0 0
	cd experiments/air_quality/models && python m_bayes_newt.py 2 0 0
	cd experiments/air_quality/models && python m_bayes_newt.py 3 0 0
	cd experiments/air_quality/models && python m_bayes_newt.py 4 0 0
	@echo 'Running bayesnewton mean-field model'
	cd experiments/air_quality/models && python m_bayes_newt.py 0 1 0
	cd experiments/air_quality/models && python m_bayes_newt.py 1 1 0
	cd experiments/air_quality/models && python m_bayes_newt.py 2 1 0
	cd experiments/air_quality/models && python m_bayes_newt.py 3 1 0
	cd experiments/air_quality/models && python m_bayes_newt.py 4 1 0
	@echo 'Running bayesnewton parallel model'
	cd experiments/air_quality/models && python m_bayes_newt.py 0 0 1
	cd experiments/air_quality/models && python m_bayes_newt.py 1 0 1
	cd experiments/air_quality/models && python m_bayes_newt.py 2 0 1
	cd experiments/air_quality/models && python m_bayes_newt.py 3 0 1
	cd experiments/air_quality/models && python m_bayes_newt.py 4 0 1
	@echo 'Running gpflow model'
	cd experiments/air_quality/models && python m_gpflow.py 0
	cd experiments/air_quality/models && python m_gpflow.py 1
	cd experiments/air_quality/models && python m_gpflow.py 2
	cd experiments/air_quality/models && python m_gpflow.py 3
	cd experiments/air_quality/models && python m_gpflow.py 4
	@echo 'Running ski model'
	cd experiments/air_quality/models && python m_ski.py 0
	cd experiments/air_quality/models && python m_ski.py 1
	cd experiments/air_quality/models && python m_ski.py 2
	cd experiments/air_quality/models && python m_ski.py 3
	cd experiments/air_quality/models && python m_ski.py 4
	@echo 'Running NYC experiments'
	cd experiments/nyc_crime && mkdir -p results
	@echo 'Running bayesnewton model'
	cd experiments/nyc_crime/models && python m_bayes_newt.py 0 0 0
	cd experiments/nyc_crime/models && python m_bayes_newt.py 1 0 0
	cd experiments/nyc_crime/models && python m_bayes_newt.py 2 0 0
	cd experiments/nyc_crime/models && python m_bayes_newt.py 3 0 0
	cd experiments/nyc_crime/models && python m_bayes_newt.py 4 0 0
	@echo 'Running bayesnewton mean-field model'
	cd experiments/nyc_crime/models && python m_bayes_newt.py 0 1 0
	cd experiments/nyc_crime/models && python m_bayes_newt.py 1 1 0
	cd experiments/nyc_crime/models && python m_bayes_newt.py 2 1 0
	cd experiments/nyc_crime/models && python m_bayes_newt.py 3 1 0
	cd experiments/nyc_crime/models && python m_bayes_newt.py 4 1 0
	@echo 'Running bayesnewton parallel model'
	cd experiments/nyc_crime/models && python m_bayes_newt.py 0 0 1
	cd experiments/nyc_crime/models && python m_bayes_newt.py 1 0 1
	cd experiments/nyc_crime/models && python m_bayes_newt.py 2 0 1
	cd experiments/nyc_crime/models && python m_bayes_newt.py 3 0 1
	cd experiments/nyc_crime/models && python m_bayes_newt.py 4 0 1
	@echo 'Running gpflow model'
	cd experiments/nyc_crime/models && python m_gpflow.py 0
	cd experiments/nyc_crime/models && python m_gpflow.py 1
	cd experiments/nyc_crime/models && python m_gpflow.py 2
	cd experiments/nyc_crime/models && python m_gpflow.py 3
	cd experiments/nyc_crime/models && python m_gpflow.py 4


