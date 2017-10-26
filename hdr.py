import neuralnet
import numpy as np
import math
from mnist import MNIST

# Handwritten Digit Recognition
# Data from MNIST
# http://yann.lecun.com/exdb/mnist/

def format_data(images, labels):
	data = []
	for index in range(0, len(labels)):
		input = np.array(images[index]) / 255
		output = np.zeros(10)
		output[labels[index]] = 1.0
		data.append([input, output])
	return data

print('Loading Data...')
mndata = MNIST('samples')
images, labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

hdr_training_data = format_data(images, labels)
test_hbr_training_data = format_data(test_images, test_labels)

hdr = neuralnet.NeuralNet((784, 77, 33, 10))
print('Training...')
hdr.train(hdr_training_data, 1, 30000, printinfo=1000)

# Predict a random image!
index = math.floor(np.random.rand() * len(test_images))
prediction, vector = hdr.predict(test_images[index])

print(mndata.display(test_images[index]))
print('expected: ', test_labels[index])
print('output: ', prediction, '\nprobability: ', max(vector), '\nvector: ', vector)

if prediction == test_labels[index]:
	print('CORRECT. index: ', index)
else:
	print('INCORRECT. index: ', index)

# Test Accuracy
# Hyperparameters optimized based on first half of testing data
# and true accuracy is calculated on second half of data
# in order to avoid manually overfitting

# First Half
# correct = 0
# incorrect_indices = []
# half = int(len(test_labels) / 2)
# for index in range(0, half):

# 	prediction = hdr.predict(test_hbr_training_data[index][0])[0]
# 	if prediction == test_labels[index]:
# 		correct += 1
# 	else:
# 		incorrect_indices.append(index)

# print(correct, ' correct out of ', half)
# print(correct / half * 100.0, '% accuracy')

# Second Half
correct = 0
incorrect_indices = []
half = int(len(test_labels) / 2)
for index in range(half, len(test_labels)):

	prediction = hdr.predict(test_hbr_training_data[index][0])[0]
	if prediction == test_labels[index]:
		correct += 1
	else:
		incorrect_indices.append(index)

print('\n', correct, ' correct out of ', half)
print(correct / half * 100.0, '% accuracy')
# print(incorrect_indices)