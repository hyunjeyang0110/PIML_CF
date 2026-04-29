# PIML_CF
Physics-Informed Deep Learning for Hurricane-Induced Compound Flooding Prediction

## Overview
This repository provides deep learning model architectures and implementation code for compound flooding prediction, integrating physical constraints based on the simplified shallow water equations (SSWEs).

<p align="center">
  <img src="figures/PIML_CF overview.png" width="800">
</p>

## We investigate:
- **Where** PIML improves predictions: spatial patterns of model improvement
- **When** PIML improvements occur: temporal dynamics of predictive enhancement
- **How** physics constraints improve model behavior: mechanisms underlying physically consistent predictions

## Key Points
•	Physics-informed machine learning improves hurricane-induced compound flooding predictions in a spatiotemporally heterogeneous manner, particularly in hydrodynamically active regions and around peak flooding conditions
•	Physics constraints suppress non-physical amplification, improving physical consistency and prediction stability
•	Momentum constraints are the dominant contributor to the performance improvement, and PIML shows increasing predictive advantages under limited training data conditions

<p align="center">
  <img src="figures/Spatiotemporal Distribution of PIML Performance Improvement.png" width="800">
</p>


## Model Input / Output
**Input (19 channels):**
- Hydrodynamic states (h, u, v) at t, t-1, t-2, t-3
- Forcing: rainfall, wind, pressure, discharge
- Static: bathymetry, Manning’s n

**Output:**
- Water depth (h) at t+1
- Velocity (u, v) at t+1

## Method
- Governing equations: Simplified Shallow Water Equations (SSWEs)
- Residuals:
  - x-momentum
  - y-momentum
  - mass conservation
- Discretization: Finite Difference Method (FDM)
- Loss:
  - Data loss (MSE)
  - Physics loss (residuals)

## Deep Learning Architectures Tested
- UNet
- ConvLSTM
- SwinUNETR
- FNO


# Meta

Hyunje Yang – hyunjeyang@utexas.edu

Distributed under the Creative Commons Legal Code CC0 1.0 Universal. See ``LICENSE`` for more information.
