# Financial Modelling Package

This repository contains implementations of various financial models for data analysis, forecasting, and options pricing.

---

## Table of Contents
1. [Gaussian Process Model](#gaussian-process-model)
   - [Theory](#theory)
   - [Mathematical Details](#mathematical-details)
   - [Usage](#usage)
2. [Call-Put Parity Criterion](#call-put-parity-criterion)
3. [Other Models (NSS, SABR, etc.)](#other-models)

---

## Gaussian Process Model

### Theory

The **Gaussian Process Model** is a non-parametric regression method widely used in machine learning and financial modeling. It assumes that the target variable \( y \) is a sample from a multivariate normal distribution, with a covariance structure defined by a kernel function. 

This model is particularly useful for interpolation and can capture non-linear patterns in the data.

### Mathematical Details

#### 1. **Gaussian Process Definition**
A Gaussian Process (GP) is defined as:
$$
f(x) \sim \mathcal{GP}(m(x), k(x, x')),
$$
where:
- \( m(x) \): Mean function (typically set to 0 in most practical cases).
- \( k(x, x') \): Covariance function or kernel that defines the relationship between points.

#### 2. **Kernel (RBF Kernel)**
The kernel \( k(x, x') \) used in this implementation is the **Radial Basis Function (RBF)** kernel, defined as:
$$
k(x, x') = \sigma^2 \exp\left(-\frac{(x - x')^2}{2\ell^2}\right),
$$
where:
- \( \ell \): Length scale parameter that controls the smoothness of the function.
- \( \sigma^2 \): Variance parameter that controls the amplitude of the function.

#### 3. **Model Training**
The training process involves computing the posterior distribution of the Gaussian Process. Given the training data \( \mathbf{X}_{\text{train}} \) and \( \mathbf{y}_{\text{train}} \), the kernel matrix is computed as:
$$
K(\mathbf{X}_{\text{train}}, \mathbf{X}_{\text{train}}) = k(\mathbf{X}_{\text{train}}, \mathbf{X}_{\text{train}}) + \sigma_n^2 I,
$$
where:
- \( \sigma_n^2 \): Noise variance parameter.
- \( I \): Identity matrix.

The Cholesky decomposition of \( K \) is then used to compute the precision-weighted target values:
$$
\boldsymbol{\alpha} = K^{-1} \mathbf{y}_{\text{train}}.
$$

#### 4. **Prediction**
To predict the output \( \mathbf{f}^* \) for a new set of inputs \( \mathbf{X}_{\text{test}} \), the covariance between \( \mathbf{X}_{\text{test}} \) and \( \mathbf{X}_{\text{train}} \) is computed:
$$
K_*(\mathbf{X}_{\text{test}}, \mathbf{X}_{\text{train}}) = k(\mathbf{X}_{\text{test}}, \mathbf{X}_{\text{train}}).
$$

The predictive mean is then given by:
$$
\mathbb{E}[\mathbf{f}^*] = K_*(\mathbf{X}_{\text{test}}, \mathbf{X}_{\text{train}}) \boldsymbol{\alpha}.
$$

#### 5. **Hyperparameters**
The model includes three key hyperparameters:
- Length scale (\( \ell \)): Controls the smoothness of the fitted curve.
- Variance (\( \sigma^2 \)): Determines the amplitude of variations in the target.
- Noise variance (\( \sigma_n^2 \)): Models the noise in the data.

### Usage

Here’s an example of how to use the Gaussian Process Model:

```python
# Import and initialize the model
from modelling.NSS_model import GaussianProcessModel
import numpy as np

# Define training data (maturities and forward rates)
maturities = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
forward_rates = np.array([2.0, 2.5, 3.0, 3.5, 4.0], dtype=np.float32)

# Initialize the Gaussian Process Model
gpr = GaussianProcessModel(length_scale=1.0, variance=1.0, noise_variance=0.1)

# Fit the model
gpr.fit(maturities, forward_rates)

# Predict forward rates
predicted_rates = gpr.predict(maturities)

# Print predictions
print("Predicted Forward Rates:", predicted_rates.numpy())
