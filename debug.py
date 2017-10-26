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