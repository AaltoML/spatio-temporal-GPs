# spatio-temporal-GPs

Code for the NeurIPS 2021 paper *'Spatio-Temporal Variational Gaussian Processes'* by Oliver Hamelijnck, William Wilkinson, Niki Loppi, Arno Solin and Theodoros Damoulas.

## Citing this work:
```
@inproceedings{hamelijnck2021spatio,
	title={Spatio-Temporal Variational {G}aussian Processes},
	author={Hamelijnck, Oliver and Wilkinson, William and Loppi, Niki and Solin, Arno and Damoulas, Theodoros},
	booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
	year={2021},
}
```

## Experiment Setup

This has been tested on a Macbook Pro.

### Environment Setup

We recommend using conda:

```bash
conda create -n spatio_gp python=3.7
conda activate spatio_gp
```

Then install the required python packages:

```
pip install -r requirements.txt
```

### Data Download

#### Pre-processed Data

All data, preprocessed and split into train-test splits used in the paper is provided at https://doi.org/10.5281/zenodo.4531304. Download the folder and place the corresponding datasets into `experiments/*/data` folders.

#### Manual Data Setup

We also provide scripts to generate the data. To download the raw  London air pollution and NYC crime data simply run the corresponding `./run.sh` scripts within the  `datasets` folder.

Then within each experiment folder (e.g `experiments/air_quality/`) run `python setup_data.py`.

### Running Experiments

tbd.

