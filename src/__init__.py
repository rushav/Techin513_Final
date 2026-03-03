"""
Synthetic Sleep Environment Dataset Generator — Team 7
TECHIN 513 Final Project

Authors: Rushav Dash, Lisa Li

We generate 2,500 realistic synthetic sleep sessions that pair
bedroom environmental time-series (temperature, light, humidity,
ambient noise) with validated sleep quality labels.  The pipeline
applies rigorous signal-processing techniques—spectral synthesis,
Butterworth filtering, FFT analysis, and Poisson event modelling—
then extracts scalar features and fits a Random Forest ensemble to
produce defensible, statistically validated synthetic data.
"""
