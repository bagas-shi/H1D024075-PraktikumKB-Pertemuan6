import numpy as np
import Perceptron as p

X = np.array([[1,1], [1,-1], [-1,1], [-1,-1]])
t = np.array([[1], [1], [1], [-1]])

model = p.Perceptron(alpha=0.1, epoch=10)
model.fit(X, t)