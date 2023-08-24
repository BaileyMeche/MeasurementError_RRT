# MeasurementError_RRT

This repository contains the Python and R code for the study "The Effect of Measurement Error on Binary Randomized Response Technique (RRT) Models".

The Python code simulates data from two basic RRT models and one comprehensive model. The R code estimates the odds ratio for predictability for each model.

The abstract of the paper is provided below:

This study introduces the effect of measurement error on binary Randomized Response Technique (RRT) models. We discuss a method for estimating and accounting for measurement error in two basic models and one comprehensive model. Both theoretical and empirical results show that not accounting for measurement error leads to inaccurate estimates. We introduce estimators that account for the effect of measurement error. Furthermore, we introduce a new measure of model privacy using an odds ratio statistic which offers better interpretability than traditional methods.

To get started, clone the repository to your local machine. Then, install the necessary Python and R packages.

To run the Python code, navigate to the python directory and run the following command:

python simulate_data.py

This will simulate data from the two basic RRT models and the comprehensive model. The data will be saved in the data directory.

To run the R code, navigate to the r directory and run the following command:

Rscript estimate_odds_ratio.R

This will estimate the odds ratio for predictability for each model. The results will be saved in the results directory.

For more information, please see the paper.

