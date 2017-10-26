#
#		Brian Santoso
#		10/25/17
#

import numpy as np
import math
import copy

np.random.seed(65536)

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def mean_squared_error(output, target):
	difference = output - target
	sum_of_squared_differences = np.dot(difference, difference)
	return 0.5 * sum_of_squared_differences / len(target)

class NeuralNet:

	def __init__(self, sizes, randomize = True):

		self.sizes = sizes
		self.num_of_layers = len(sizes)

		# array of 2d numpy arrays M x N where 
		# M is number of nodes in current layer
		# N is number of nodes in previous layer
		if randomize:
			self.weights = [2 * np.random.rand(sizes[x], sizes[x - 1]) - 1 for x in range(1, self.num_of_layers)]		
			self.biases = [2 * np.random.rand(sizes[x]) - 1 for x in range(1, self.num_of_layers)]
		else:
			self.weights = [np.ones(shape=(sizes[x], sizes[x - 1])) for x in range(1, self.num_of_layers)]
			self.biases = [np.zeros(sizes[x]) for x in range(1, self.num_of_layers)]

	def feed_forward(self, input):

		# input is a numpy array
		outputs = [input]

		for l in range(1, self.num_of_layers):
			# weights of nodes of current layer 
			# (weights array is 1 smaller than outputs since it does not include inputs, hence the l - 1)
			# weights of outputs of previous layer
			x = np.dot(self.weights[l - 1], outputs[l - 1]) + self.biases[l - 1]
			outputs.append(sigmoid(x))

		return outputs

	def back_propagation(self, outputs, target, rate):

		# output is the last vector in the outputs array
		output = outputs[self.num_of_layers - 1]
		# initialize deltas array with delta_k at the end
		deltas = [output * (1 - output) * (output - target)]

		# start at second to last layer and stop at the 2nd layer
		for l in range(self.num_of_layers - 2, 0, -1):
			# delta_j = output_j * (1 - output_j) * sum(delta_k * weight_jk)
			o_j = outputs[l] #outputs of corresponding layer
			delta = o_j * (1 - o_j) * np.dot(self.weights[l].T, deltas[0])
			deltas.insert(0, delta)

		partial_derivatives = []
		for l in range(1, self.num_of_layers):
			partial_derivatives.append(deltas[l - 1][:,None] * outputs[l - 1])

		for l in range(1, self.num_of_layers):
			self.weights[l - 1] -= rate * partial_derivatives[l - 1]
			self.biases[l - 1] -= rate * deltas[l - 1]

		return partial_derivatives

	def train(self, training_data, rate, epoch, min_error = 1e-4, printinfo = 1e99):
		d = 0
		error = 1e99
		while d < epoch and epoch > min_error:
			i = d % len(training_data)
			inp = training_data[i][0]
			target = training_data[i][1]
			outputs = self.feed_forward(inp)
			error = mean_squared_error(outputs[self.num_of_layers - 1], target)
			self.back_propagation(outputs, target, rate)
			d += 1
			if d % printinfo == 0:
				print('epoch: ', d, ' error: ', error)
		return

	def predict(self, input):
		output = self.feed_forward(input)[-1]
		prediction = np.argmax(output)
		return prediction, output

	def info(self):
		print('sizes: {0} \nnum_of_layers: {1} \nweights: {2} \nbiases: {3} \n'.format(self.sizes, self.num_of_layers, self.weights, self.biases))