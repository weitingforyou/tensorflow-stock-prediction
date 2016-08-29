# Tensorflow Stock Prediction
Run CNN and DQN model with Tensorflow for stock prediction. 
Feature include daily close price, daily relative price, MA, RSI.
Stock data are collected from matplotlib.finance.

## Installation Dependencies
 - Tensorflow
 - Python 2.7
 - matplotlib
 - Numpy
 - Gym (for Deep Q Network)

## Distributed Tensorflow
Distributed version for MNIST example.
Create a cluster of Tensorflow servers, and distribute a computation graph across the cluster.
Need to consider about transmission time. Transmission time must not be longer than compute time.

## Convolutional Neural Network
Take reference from RobRomijnders's work (http://robromijnders.github.io/CNN_tsc/)

## Deep Q Network
Take reference from FloodSung's work (https://zhuanlan.zhihu.com/p/21477488?refer=intelligentunit) and yenchenlin's work

