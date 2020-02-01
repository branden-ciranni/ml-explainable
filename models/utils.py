'''Utility functions for statistical functions, calculus, and others
that aren't specific to the Linear Model.
'''
import numpy as np


def ewma(tensor, t, alpha=0.90, prev_n=20):
	'''Recursively Calculate the Exponentially Weighted Moving
	Average for the given tensor at timestep t.
	
	tensor : non-empty tensor.
	alpha : Smoothing factor, between 0 and 1. Higher alpha
			discounds older observations faster.
	t : current timestep
	prev_n : number of previous observations
	
	returns:
	exponentially weighted moving average discounted by alpha at
	timestep t
	'''
	if prev_n==0 or t==0:
		return tensor[t]
	else:
		return alpha * tensor[t] + (1-alpha) * ewma(tensor, t-1, alpha, prev_n-1)


def mse(X, Y, W_t):
	'''Calculate the Mean Squared Error of the predicted
	Y_hat value with the true Y, where Y_hat = X.dot(W_t).
	This is convenient for theory, but not always useful
	in practice.

	X : array of feature vectors
	Y : array of target values
	W_t : parameters array for this time step

	returns:
	Calculated Mean Squared Error
	'''
	m = X.shape[0]
	return 2/m * X.T.dot(X.dot(W_t)-Y)


def gradients(loss_func, X, Y, W_t):
	'''Calculate the gradients for a step according to the
	loss function.

	loss_func : function which takes X, Y, and theta as
				parameters, and calculates the loss at some
				time step.
	X : array of feature vectors
	Y : array of target values
	W_t : parameters array for this time step
	'''
	m = X.shape[0]
	return loss_func(X, Y, W_t)


def add_bias(X):
	'''Adds a "1" bias term to feature array X

	ex. if X is an array:
	[[1],
	 [2],
	 [3],
	 [4],
	 [5],
	]

	add_bias transforms it to the array:
	[[1, 1],
	 [1, 2],
	 [1, 3],
	 [1, 4],
	 [1, 5],
	]

	X : array of feature vectors

	returns:
	transformed array with added bias terms
	'''
	m = X.shape[0]
	return np.c_[np.ones((m, 1)), X]


