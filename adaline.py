# Adaptive linear neuron - Adaline

import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import math
import matplotlib.pyplot as plt

class AdaptiveLinearNeuron(object):

    def __init__(self, rate = 0.01, niter = 10):
      self.rate = rate
      self.niter = niter

    def fit(self, X, y):
          
      # weights
      self.weight = np.zeros(1 + X.shape[1])

      # Number of misclassifications
      self.errors = []

      # Cost function
      self.cost = []

      for i in range(self.niter):
         output = self.net_input(X)
         errors = y - output
         print(errors)
         self.weight[1:] += self.rate * X.T.dot(errors)
         self.weight[0] += self.rate * errors.sum()
         cost = (errors**2).sum() / 2.0
         self.cost.append(cost)
      return self

    def net_input(self, X):
      """Calculate net input"""
      return np.dot(X, self.weight[1:]) + self.weight[0]

    def activation(self, X):
      """Compute linear activation"""
      return self.net_input(X)

    def predict(self, X):
      """Return class label after unit step"""
      return self.activation(X)


df = pd.read_csv('sunspot.txt', delimiter = "\t", header = None)


n = 2
dataDict = {}

for x in range(n):
    dataDict[x] = []

rowsNumber = math.floor(df[0].size / n) - n;

for x in range(rowsNumber):
    for y in range(n):
          dataDict[y].append(df[1][x + y])

data = pd.DataFrame.from_dict(dataDict)

X = data.iloc[0:rowsNumber, 0:n-1].values
y = data.iloc[0:rowsNumber, n-1].values

model = AdaptiveLinearNeuron(0.001, 20).fit(X, y)

plt.plot(range(1, len(model.cost) + 1), model.cost, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.title('Adaptive Linear Neuron')

print(y)
plt.show()