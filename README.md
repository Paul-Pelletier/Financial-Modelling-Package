# Call-Put Parity Criterion: Discount Factor Justification

## Introduction
In options pricing, the call-put parity relationship provides a fundamental connection between the prices of call and put options for the same underlying asset, strike price, and expiration date. 

This document explains a modified criterion:

![Criterion](https://render.githubusercontent.com/render/math?math=%5Ctext%7BCriterion%7D%20%3D%20-%5Cfrac%7BC%20-%20P%20-%20S%7D%7BK%7D)

where:
- \( C \): Call price.
- \( P \): Put price.
- \( S \): Spot price of the underlying asset.
- \( K \): Strike price.

This criterion represents the **discount factor** applied to the strike price in the standard call-put parity equation.

---

## Mathematical Derivation

The call-put parity relationship is given by:

![Call-Put Parity](https://render.githubusercontent.com/render/math?math=C%20-%20P%20%3D%20S%20-%20K%20e%5E%7B-rT%7D)

Rearranging to isolate the discount factor \( e^{-rT} \):

![Discount Factor](https://render.githubusercontent.com/render/math?math=e%5E%7B-rT%7D%20%3D%20%5Cfrac%7BS%20-%20%28C%20-%20P%29%7D%7BK%7D)

The modified criterion becomes:

![Modified Criterion](https://render.githubusercontent.com/render/math?math=%5Ctext%7BCriterion%7D%20%3D%20-%5Cfrac%7BC%20-%20P%20-%20S%7D%7BK%7D%20%3D%20e%5E%7B-rT%7D)

---

## Interpretation
The criterion:
1. Represents the **discount factor** applied to the strike price \( K \).
2. Provides a **dimensionless measure** of consistency between call and put prices.

---

## Practical Application
This criterion allows us to:
1. Compute the **implied discount factor** directly from market-observed option prices.
2. Infer the **implied risk-free rate** \( r \):

![Implied Risk-Free Rate](https://render.githubusercontent.com/render/math?math=r%20%3D%20-%5Cfrac%7B%5Cln%28%5Ctext%7BCriterion%7D%29%7D%7BT%7D)

3. Detect pricing anomalies by comparing the computed criterion with theoretical expectations.

---

## Example
For a call price \( C = 12.50 \), put price \( P = 10.25 \), spot price \( S = 100 \), and strike price \( K = 105 \):

![Example Criterion](https://render.githubusercontent.com/render/math?math=%5Ctext%7BCriterion%7D%20%3D%20-%5Cfrac%7B12.50%20-%2010.25%20-%20100%7D%7B105%7D%20%3D%200.0143)

The discount factor is approximately:

![Discount Factor Approximation](https://render.githubusercontent.com/render/math?math=e%5E%7B-rT%7D%20%3D%200.0143)

---
