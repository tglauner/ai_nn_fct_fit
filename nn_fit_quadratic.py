"""
Fit a polynomial with a neural network
Should be quite easy I would think
We'll use a multi-layer Perceptron
"""

from matplotlib import pyplot as plt 
import numpy as np 
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as mse

def to_fit(x):
	return x*x

#TODO: use random sample
#X = np.array([1,2,3,4,5,6,7,8,9,10,1.1,1.2])
X = np.random.random(10)*10
y = to_fit(X)

#with max_iter not set it does not converge
model = MLPRegressor(max_iter=100000, verbose=False)

#Single feature reshape
X = X.reshape(-1, 1)
model.fit(X, y)

X_test = np.array([2, 3, 4, 5, 6, 1])
y_test = to_fit(X_test)

X_test = X_test.reshape(-1,1)
predict = model.predict(X_test)

#print mse(predict, y_test)
print X_test
print y_test
print predict
plt.plot(X_test)
plt.plot(y_test)
plt.plot(predict)
plt.show()
