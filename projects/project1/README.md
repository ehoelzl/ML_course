# EPFL Machine Learning CS-433 - Project 1 report



## Description

This repository is a fork of the ML_Course official repository. It contains the work we did for the project 1 which consisted in using ML techniques to infer the abscence of presence of the Higgs boson 

## Project structure

To be able to run the script, `test.csv` and `train.csv` must be in `data/`


## To run the project 

`git clone https://github.com/ehoelzl/ML_course`

in the repository where the project was cloned 

`cd ML_course` and then run `jupyter notebook` or execute `python run.py` (from within `scripts/`) in your terminal



## To see how we worked

Most of our logic and testing took place in the `project1.ipynb`

Our code is the project 1 directory. 

We tried splitting the different functions a maximum to increase readability. The following files are presented : 



- `costs.py` the different loss functions MAE, MSE etc

- `cross_validation.py` cross validation function for ridge, regression etc

- `data_processing.py` split, normalize, standardize, etc dataset

- `feature_exapansion.py` polynomial extension and all function to expand features

- `gradients.py` compute MAE, MSE gradients

- `implementations.py` all the required implementations

- `plots.py` Plotting

- `proj1_helpers.py` from and to csv

- `utils.py` batching, splitting

  

  

  

  





