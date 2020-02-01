'''Linear Models implemented in NumPy
'''
import numpy as np
import utils

class LinearModel():

	def __init__(self, eta=0.1, n_iter=1000, reg='ridge', mu=0.9, batch_size=32):
		self.eta = eta # Learning Rate
		self.n_iter = n_iter
		self.reg = reg
		self.mu = mu # Momentum parameter - higher value discounts older terms more 
		self.batch_size = batch_size


	def _momentum(self, w_prev, G, t):
		'''calculates, in addition to the first derivative of
		the current step, the exponentially weighted moving average of the 
		first derivatives for the previous steps to get an estimate for 
		the curvature of the loss function. 

		The idea is that if we are approaching a minima, we want to slow down,
		whereas if the opposite is true we want to speed up. This helps to
		prevent zig-zagging around in the loss function by overshooting
		the minimum, so as to improve the speed of convergence.

		w_prev : previous parameter array
		G : gradient tensor
		t : current time step

		returns:
		parameter array after this time step, adjusted by momentum
		'''
		return w_prev - self.eta * utils.ewma(G, t, self.mu)


	def _bgd(self, w_prev, g_iter):
		'''Implements Batch Gradient Descent:
		Adjusts theta according to the learning rate and
		gradients calculated from some loss function

		X : array of feature vectors
		Y : array of target values
		theta : parameters array

		returns:
		parameter array after this time step
		'''
		return w_prev - self.eta * g_iter


	def _get_random_mini_batch(self, X, Y):
		'''returns a random batch of size self.batch_size
		from X and Y
		'''
		idx = np.random.randint(len(X)-self.batch_size)
		X_batch = X[idx:idx+self.batch_size]
		Y_batch = Y[idx:idx+self.batch_size]
		return X_batch, Y_batch



	def optimize(self, X, Y, loss_func):
		'''Fits Linear Regression Model to input array X by optimizing
		the loss function.

		X : array of feature vectors
		Y : array of target values

		returns:
		Optimal weight values after n_iter steps
		'''
		if X.ndim == 1:
			X = X[np.newaxis]
		elif X.ndim > 2:
			raise Exception("Your features are not linear!")

		# Add bias term to X
		X_b = utils.add_bias(X)

		# Number of samples, features
		m, n = X_b.shape

		# Initialize random weights
		w_initial = np.random.randn(n,1)

		# Initialize empty Weight and Gradient tensors
		W = np.empty((self.n_iter+1, n, 1))
		G = np.empty((self.n_iter, n, 1))

		W[0] = w_initial

		for t in range(self.n_iter):
			# Get Random Batch
			batchX, batchY = self._get_random_mini_batch(X_b, Y)

			# Get previous weights
			w_prev = W[t]

			# Calculate Gradient and weights
			g_iter = utils.gradients(loss_func, batchX, batchY, w_prev)
			G[t] = g_iter

			# Calculate new weights
			w_iter = self._momentum(w_prev, G, t)
			W[t+1] = w_iter

		# Return last calculated weights
		return W, G


if __name__ == '__main__':
	print("Testing...")
	X = 2 * np.random.rand(100, 1)
	Y = 4 + 3 * X + np.random.randn(100, 1)

	lm = LinearModel()
	W, G = lm.optimize(X,Y, utils.mse)
	print(W[-1])

