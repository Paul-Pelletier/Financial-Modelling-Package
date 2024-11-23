# Call-Put Parity Criterion: Discount Factor Justification

## Introduction
In options pricing, the call-put parity relationship provides a fundamental connection between the prices of call and put options for the same underlying asset, strike price, and expiration date. 

This document explains a modified criterion:

$$
\text{Criterion} = -\frac{C - P - S}{K}
$$

where:
- \( C \): Call price.
- \( P \): Put price.
- \( S \): Spot price of the underlying asset.
- \( K \): Strike price.

This criterion represents the **discount factor** applied to the strike price in the standard call-put parity equation.

---

## Mathematical Derivation

The call-put parity relationship is given by:

$$
C - P = S - K e^{-rT}
$$

Rearranging to isolate the discount factor \( e^{-rT} \):

$$
e^{-rT} = \frac{S - (C - P)}{K}
$$

The modified criterion becomes:

$$
\text{Criterion} = -\frac{C - P - S}{K} = e^{-rT}
$$

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

$$
r = -\frac{\ln(\text{Criterion})}{T}
$$

3. Detect pricing anomalies by comparing the computed criterion with theoretical expectations.

---

## Example
For a call price \( C = 12.50 \), put price \( P = 10.25 \), spot price \( S = 100 \), and strike price \( K = 105 \):

$$
\text{Criterion} = -\frac{12.50 - 10.25 - 100}{105} = 0.0143
$$

The discount factor is approximately:

$$
e^{-rT} = 0.0143
$$
