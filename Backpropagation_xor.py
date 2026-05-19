import numpy as np
import Backpropagation as b

X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
t = np.array([[-1], [1], [1], [-1]])

model = b.Backpropagation(alpha=0.3, epoch=1000, target_error=0.001)
model.fit(X, t)