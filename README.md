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


## Key Features
- Physics-informed deep learning (PIML)
- ConvLSTM-based spatiotemporal modeling
- Finite-difference-based physics residuals
- Event-based training and OOD evaluation (Hurricane Ike)
- Ablation study on physics constraints

## Main Findings
- PIML improves both accuracy and physical consistency
- Improvements are **not uniform**
  - Spatial: hydrodynamically active regions
  - Temporal: peak flooding near landfall
- Momentum constraints are the dominant factor
- Robust under limited training data
- ~145× faster inference than numerical models

## Model Input / Output
**Input (19 channels):**
- Hydrodynamic states (h, u, v) at t, t-1, t-2, t-3
- Forcing: rainfall, wind, pressure, discharge
- Static: bathymetry, Manning’s n

**Output:**
- Water depth (h)
- Velocity (u, v)

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