#
#		Brian Santoso
#		10/18/17
#

import numpy as np
import math
import copy

# XOR Neural Net does not work with this seed?
# np.random.seed(5465456)

# XOR Neural Net works fine with this seed
# np.random.seed(2342342)

# HDR Neural Net works with this one
# np.random.seed(37891248)

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

	def info(self):
		print('sizes: {0} \nnum_of_layers: {1} \nweights: {2} \nbiases: {3} \n'.format(self.sizes, self.num_of_layers, self.weights, self.biases))


# xor_training_data = [ 	[np.array([0, 0]), np.array([0])],
# 						[np.array([1, 0]), np.array([1])],
# 						[np.array([0, 1]), np.array([1])],
# 						[np.array([1, 1]), np.array([0])]	]

# xor = NeuralNet((2, 2, 1))
# xor.train(xor_training_data, 5, 5000, 1e-4, printinfo=1000)

# # # Testing the net

# print('tests:')
# o = xor.feed_forward(np.array([0, 0]))
# print('input: (0, 0) 	output: ', o[-1:][0])
# o = xor.feed_forward(np.array([1, 0]))
# print('input: (1, 0) 	output: ', o[-1:][0])
# o = xor.feed_forward(np.array([0, 1]))
# print('input: (0, 1) 	output: ', o[-1:][0])
# o = xor.feed_forward(np.array([1, 1]))
# print('input: (1, 1) 	output: ', o[-1:][0])


# Debugging - Numerical Approximation of Gradient Descent

# EPSILON = 1e-4
# index = 1
# inp = xor_training_data[index][0]
# target = xor_training_data[index][1]
# outputs = xor.feed_forward(inp)
# gradient = xor.back_propagation(outputs, target, 0)	# partial derivatives of each weight
# print(gradient)


# print('weights ', xor.weights)

# t1 = copy.deepcopy(xor.weights)
# t2 = copy.deepcopy(xor.weights)
# t1[1][0][0] += EPSILON
# print('t1 ', t1)


# t2[1][0][0] -= EPSILON
# print('t2 ', t2)

# xor.weights = t1
# J1 = xor.feed_forward(xor_training_data[index][0])[-1:][0]
# xor.weights = t2
# J2 = xor.feed_forward(xor_training_data[index][0])[-1:][0]

# partial_derivative_approximation = (mean_squared_error(J1, target) - mean_squared_error(J2, target)) / (2 * EPSILON)
# print(partial_derivative_approximation)
# xor.info()


# Handwritten Digit Recognition

from mnist import MNIST
mndata = MNIST('samples')
images, labels = mndata.load_training()
# images, labels = mndata.load_testing()

hdr_training_data = []

for index in range(0, len(labels)):

	input = np.array(images[index]) / 255
	# input = np.array(images[index])
	output = np.zeros(10)
	output[labels[index]] = 1.0

	hdr_training_data.append([input, output])


hdr = NeuralNet((784, 32, 10))

hdr.train(hdr_training_data, 1, 50000, printinfo = 1000)








test_images, test_labels = mndata.load_testing()

test_hbr_training_data = []

for index in range(0, len(test_labels)):

	input = np.array(test_images[index]) / 255
	# input = np.array(images[index])
	output = np.zeros(10)
	output[test_labels[index]] = 1.0

	test_hbr_training_data.append([input, output])










# Predict a random image!

index = math.floor(np.random.rand() * len(test_images))
# index = 7049
print(mndata.display(test_images[index]))

print('expected: ', np.argmax(test_hbr_training_data[index][1]), ' vector: ', test_hbr_training_data[index][1])
o = hdr.feed_forward(test_hbr_training_data[index][0])

prediction = np.argmax(o[-1:][0])
print('output: ', prediction, '\nvector: ', o[-1:][0])


if prediction == test_labels[index]:
	print('CORRECT. index: ', index)
else:
	print('incorrect. index: ', index)


correct = 0
incorrect_indices = []
for index in range(len(test_labels)):

	o = hdr.feed_forward(test_hbr_training_data[index][0])
	prediction = np.argmax(o[-1:][0])


	if prediction == test_labels[index]:
		correct += 1
	else:
		incorrect_indices.append(index)
print(correct, ' correct out of ', len(test_labels))
# print(incorrect_indices)