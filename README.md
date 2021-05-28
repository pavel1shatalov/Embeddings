# Stock Price Prediction

The repository contains code for the final project for Deep Learning course at Skoltech in 2021

## Description

Time series prediction is an old field for which there are a lot of methods and approaches. We select some recent ones, implemented them and compared between each other. 

## Prerequisites

All dependencies are installed from the notebooks

## How to use

1. Clone GitHub repository
2. Run Jupyter notebooks

## Notebooks

Notebooks are split into model folders. Each folder contains all you need to run model

## Results

- We implemented 6 models which were described in recent papers. 

  1. Casual LSTM based approach
  2. [LSTM + CNN](https://arxiv.org/abs/2011.08011v2)
  3. [BiLSTM + CNN](https://ieeexplore.ieee.org/abstract/document/8666592)
  4. [3D CNN](https://www.sciencedirect.com/science/article/abs/pii/S0957417419301915)
  5. [Transformer](https://towardsdatascience.com/stock-predictions-with-state-of-the-art-transformer-and-time-embeddings-3a4485237de6)
  6. [GAN](https://www.sciencedirect.com/science/article/pii/S1877050919302789)
  7. [ST-GAN](https://arxiv.org/pdf/2102.01290.pdf)

- We tested them on Yahoo finance data.
- We compared our models between different configurations and between each other 

As a future work, we plan to test ST-GAN with some more data and with some other configurations. Moreover, we could finish our work with these methods and add them to some library related to time series forecasting.

