# Density Estimation for Financial Market Returns Using Normalizing Flows
## Overview
This project explores density estimation of financial market returns using Normalizing Flows, specifically RealNVP and Masked Autoregressive Flow (MAF). The goal is to model the probability distribution of S&P 500 daily returns and evaluate which method provides better performance.

## Dataset
Source: Yahoo Finance
Data: S&P 500 historical daily adjusted closing prices
Timeframe: January 1, 2000 – January 1, 2025
## Methods
#### Preprocessing:

Daily returns were computed from closing prices.
Data was normalized to ensure stability in training.
#### Modeling Approaches:

RealNVP (Real-valued Non-Volume Preserving Transformation)
Masked Autoregressive Flow (MAF)
#### Metrics Used for Evaluation:

Log-Likelihood: Measures how well the model fits the observed data (higher is better).
Mean Squared Error (MSE): Measures the deviation of generated samples from real returns (lower is better).
## Results
Model      Train Log-Likelihood      Test Log-Likelihood      MSE
RealNVP      2.9633                      2.7660               2.2791
MAF	        -0.7070	                    -0.9520	              3.3726

RealNVP outperformed MAF with a higher log-likelihood and lower MSE, indicating that it better captured the return distribution.
Implementation

#### Main dependencies:
tensorflow (for deep learning models) 

tensorflow-probability (for normalizing flows)

yfinance (to fetch stock data)

numpy, pandas, matplotlib (for data processing and visualization)

This script:

Fetches S&P 500 data

Preprocesses returns

Trains RealNVP and MAF models

Evaluates and compares their performance

### Visualization
Histogram of Returns vs. Generated Samples

### Future Improvements
Extend to other financial indices and asset classes

Experiment with additional normalizing flows

Optimize hyperparameters for improved density estimation
