# PIML_CF
This repository provides deep learning model architectures and implementation code for hurricane-induced compound flooding prediction, integrating physical constraints based on the simplified shallow water equations (SSWEs)

> **_Yang et al._**: Where, When, and How Do Physics-Informed Deep Learning Models Improve Hurricane-Induced Compound Flooding Predictions?

<p align="center">
  <img src="figures/PIML_CF overview.png" width="800">
</p>

## We Investigate
- **_Where_** PIML improves predictions: spatial patterns of model improvement
- **_When_** PIML improvements occur: temporal dynamics of predictive enhancement
- **_How_** physics constraints improve model behavior: mechanisms underlying physically consistent predictions

## Key Points
-	Physics-informed machine learning improves hurricane-induced compound flooding predictions in a spatiotemporally heterogeneous manner, particularly in hydrodynamically active regions and around peak flooding conditions
-	Physics constraints suppress non-physical amplification, improving physical consistency and prediction stability
-	Momentum constraints are the dominant contributor to the performance improvement, and PIML shows increasing predictive advantages under limited training data conditions

<p align="center">
  <img src="figures/Spatiotemporal Distribution of PIML Performance Improvement.png" width="700">
</p>

<p align="center">
  <img src="figures/Distribution of prediction error.png" width="700">
</p>


## Model Inputs / Outputs
**Inputs:**
- Hydrodynamic states (h, u, v) at t, t-1, t-2, t-3
- Forcing: rainfall, wind, pressure, discharge
- Static: bathymetry, Manning’s n

**Outputs:**
- Water depth (h) at t+1
- Velocity (u, v) at t+1

## Methods
- Governing equations: simplified shallow water equations (SSWEs)
- Residuals:
  - x-momentum
  - y-momentum
  - mass conservation

$$
r_u(x,y,t)=\partial_t(hu)+\nabla\cdot(hu\mathbf{u})+gh\partial_x(H)-\frac{\tau_{w,x}-\tau_{b,x}}{\rho}
$$

$$
r_v(x, y, t) = \partial_t (hv) + \nabla \cdot (hv\vec{v}) + gh\partial_y(H) - \frac{\tau_{w,y} - \tau_{b,y}}{\rho}
$$

$$
r_m(x, y, t) = \text{ReLU} \left( \partial_t h + \partial_x (Hu) + \partial_y (Hv) - (P + D) \right)
$$

- Discretization: finite difference method (FDM)

$$\left. \frac{\partial \phi}{\partial t} \right|_t = \frac{\phi^{t+1} - \phi^{t-1}}{2\Delta t}$$

$$\left. \frac{\partial \phi}{\partial \xi} \right|_{i,j} = \frac{\phi_{i,j+1} - \phi_{i,j-1}}{2\Delta \xi}$$

$$\left. \frac{\partial \phi}{\partial \eta} \right|_{i,j} = \frac{\phi_{i+1,j} - \phi_{i-1,j}}{2\Delta \eta}$$

- Total loss:
  - Data loss (MSE)
  - Physics loss (SSWEs residuals)

$$
\mathcal{L}_{\text{Total}} = \lambda_u \langle r_u^2 \rangle + \lambda_v \langle r_v^2 \rangle + \lambda_m \langle r_m^2 \rangle + \lambda_d \mathcal{L}_{\text{data}}
$$

## Deep Learning Architectures Tested
- UNet
- ConvLSTM
- SwinUNETR
- FNO


# Meta

Hyunje Yang – hyunjeyang@utexas.edu

Distributed under the Creative Commons Legal Code CC0 1.0 Universal. See ``LICENSE`` for more information
