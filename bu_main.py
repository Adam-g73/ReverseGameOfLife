from numpy import exp, array, random, dot
import numpy as np
import sys
class NeuralNetwork():
	def __init__(self):
		# Seed the random number generator, so it generates the same numbers
		# every time the program runs.
		random.seed(1)

		# We model a single neuron, with 3 input connections and 1 output connection.
		# We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
		# and mean 0.
		weights = 2 * random.random((1, 400)) - 1
		self.synaptic_weights = weights[0]

	# The Sigmoid function, which describes an S shaped curve.
	# We pass the weighted sum of the inputs through this function to
	# normalise them between 0 and 1.
	def __sigmoid(self, x):
		for i in range(len(x)):
			x[i] = 1 / (1 + exp(-x[i]))
		return x

	# The derivative of the Sigmoid function.
	# This is the gradient of the Sigmoid curve.
	# It indicates how confident we are about the existing weight.
	def __sigmoid_derivative(self, x):
		for i in range(len(x)):
			x[i] = x[i] * (1 - x[i])
		return x

	# We train the neural network through a process of trial and error.
	# Adjusting the synaptic weights each time.
	def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
		for iteration in xrange(number_of_training_iterations):
		# Pass the training set through our neural network (a single neuron).
			for i in range(len(training_set_inputs)):
				# print('Training Set Inputs')
				# print(training_set_inputs[i])
				output = self.think(training_set_inputs[i])
				# print('Training Set Inputs')
				# print(training_set_inputs[i])
				# print('Outputs')
				# print(output)
				# print('Train set outputs')
				# print(training_set_outputs[i])
				# Calculate the error (The difference between the desired output
				# and the predicted output).
				error = training_set_outputs[i] - output
				# print('Error')
				# print(error)

				# Multiply the error by the input and again by the gradient of the Sigmoid curve.
				# This means less confident weights are adjusted more.
				# This means inputs, which are zero, do not cause changes to the weights.
				adjustment = training_set_inputs[i] * error * self.__sigmoid_derivative(output)

				# Adjust the weights.
				print('Adjustments')
				print(adjustment)
				self.synaptic_weights += adjustment
			print(iteration)


	# The neural network thinks.
	def think(self, inputs):
		# Pass inputs through our neural network (our single neuron).
		# print('Synaptic Weights')
		# print(self.synaptic_weights)
		intimeW = (inputs * self.synaptic_weights)
		# print('Inputs times synaptic weights')
		# print(intimeW)
		return	self.__sigmoid(intimeW)


if __name__ == "__main__":

	#Intialise a single neuron neural network.
	neural_network = NeuralNetwork()
	print "Random starting synaptic weights: "
	print neural_network.synaptic_weights

	# The training set. We have 4 examples, each consisting of 3 input values
	# and 1 output value.
	# file = open('train.csv', 'r')
	inputs = []
	outputs = []
	file = open('train.csv', 'r')
	for i, line in enumerate(file):
		if i == 0:
			continue
		line = map(int, line.strip().split(','))
		idx = line[:1]
		delta = line[1:2]
		if delta[0] != 1:
			continue
		start = line[2:402]
		end = line[402:]
		inputs.append(end)
		outputs.append(start)
	# print(inputs)
	# print(outputs)
	# training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
	# training_set_outputs = array([[0, 1, 1], [0, 1, 0], [1, 1, 1], [0, 0, 1]])
	training_set_inputs = array(inputs)
	training_set_outputs = array(outputs)
	# Train the neural network using a training set.
	# Do it 10,000 times and make small adjustments each time.
	print('Training')
	neural_network.train(training_set_inputs, training_set_outputs, 100)

	print "New synaptic weights after training: "
	print neural_network.synaptic_weights

	# Test the neural network with a new situation.
	print "Considering new situation [1, 0, 0] -> ?: "
	print neural_network.think(array(inputs[1]))
	print outputs[1]
	# print neural_network.think(array([1, 1, 1]))
	# print neural_network.think(array([0, 0, 1]))
	# print neural_network.think(array([0, 1, 1]))
# file = open('train.csv', 'r')
# for i, line in enumerate(file):
# 	if i == 0:
# 		continue
# 	line = map(int, line.strip().split(','))
# 	idx = line[:1]
# 	delta = line[1:2]
# 	start = line[2:402]
# 	end = line[402:]
# 	print(idx)
# 	print(delta)
# 	print(start)
# 	print(end)