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
		self.synaptic_weights = 2 * random.random((8, 1)) - 1
		

	# The Sigmoid function, which describes an S shaped curve.
	# We pass the weighted sum of the inputs through this function to
	# normalise them between 0 and 1.
	def __sigmoid(self, x):
		return 1.0 / (1.0 + np.exp(-x))

	# The derivative of the Sigmoid function.
	# This is the gradient of the Sigmoid curve.
	# It indicates how confident we are about the existing weight.
	def __sigmoid_derivative(self, x):
		return x * (1.0 - x)

	def check_bounds(self, inp, x):
		if x >= 0 and x < 400:
			return float(inp[x])
		return 0.0
	def get_neighbors(self, inp, i):
		n = [0 for _ in range(8)]
		if (i % 20 == 0):
			n[0] = 0.
			n[1] = self.check_bounds(inp, i - 20)
			n[2] = self.check_bounds(inp, i - 19)
			n[3] = 0.
			n[4] = self.check_bounds(inp, i + 1)
			n[5] = 0.
			n[6] = self.check_bounds(inp, i + 20)
			n[7] = self.check_bounds(inp, i + 21)
		elif i % 19 == 0:
			n[0] = self.check_bounds(inp, i - 21)
			n[1] = self.check_bounds(inp, i - 20)
			n[2] = 0.
			n[3] = self.check_bounds(inp, i - 1)
			n[4] = 0.
			n[5] = self.check_bounds(inp, i + 19)
			n[6] = self.check_bounds(inp, i + 20)
			n[7] = 0.
		else:
			n[0] = self.check_bounds(inp, i - 21)
			n[1] = self.check_bounds(inp, i - 20)
			n[2] = self.check_bounds(inp, i - 19)
			n[3] = self.check_bounds(inp, i - 1)
			n[4] = self.check_bounds(inp, i + 1)
			n[5] = self.check_bounds(inp, i + 19)
			n[6] = self.check_bounds(inp, i + 20)
			n[7] = self.check_bounds(inp, i + 21)
		return n
	# We train the neural network through a process of trial and error.
	# Adjusting the synaptic weights each time.
	def train(self, inputs, outputs, number_of_training_iterations):
		for iteration in xrange(number_of_training_iterations):
		# Pass the training set through our neural network (a single neuron).
			# print('Inputs')
			# print(len(inputs))
			for ix, inp in enumerate(inputs):
				neighbors = []
				for i in range(len(inp)):
					neighbors.append(self.get_neighbors(inp, i))
				n_b = array(neighbors)
				# print('Size of Outputs')
				# print(len(outputs))
				_outs = array([outputs[ix]]).T
				# print('Training Set Inputs')
				# print(training_set_inputs[i])
				output = self.think(n_b)
				# print('Training Set Inputs')
				# print(n_b)
				# print('Outputs')
				# print(output)
				# print('Output size')
				# print(len(output))
				# print('Train set outputs')
				# print(training_set_outputs[i])
				# Calculate the error (The difference between the desired output
				# and the predicted output).
				error = _outs - output
				# print('Error')
				# print(error)
				# print('Error Size')
				# print(len(error))
				# print('n_b size')
				# print(len(n_b))
				# Multiply the error by the input and again by the gradient of the Sigmoid curve.
				# This means less confident weights are adjusted more.
				# This means inputs, which are zero, do not cause changes to the weights.
				adjustment = dot(n_b.T, error * self.__sigmoid_derivative(output))
				# print('Adjustment')
				# print(adjustment)
				# print('Adjustment size')
				# print(len(adjustment))
				# Adjust the weights.
				# print('Adjustments')
				# print(adjustment)
				self.synaptic_weights += adjustment
				if ix % 100 == 0:
					sys.stdout.write("\rIteration: " + str(iteration) + "  Input: " + str(ix))


	# The neural network thinks.
	def think(self, inputs):
		# Pass inputs through our neural network (our single neuron).
		# print('Synaptic Weights')
		# print(self.synaptic_weights)
		# intimeW = (inputs * self.synaptic_weights)
		# print('Inputs times synaptic weights')
		# print(intimeW)
		return	self.__sigmoid(dot(inputs, self.synaptic_weights))


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
	# print('Outputs')
	# print(outputs)
	# training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
	# training_set_outputs = array([[0, 1, 1], [0, 1, 0], [1, 1, 1], [0, 0, 1]])
	# training_set_inputs = array(inputs)
	# training_set_outputs = array(outputs)
	# Train the neural network using a training set.
	# Do it 10,000 times and make small adjustments each time.
	print('Training')
	neural_network.train(inputs, outputs, 1)

	print "New synaptic weights after training: "
	print neural_network.synaptic_weights

	# Test the neural network with a new situation.
	print "Considering new situation [1, 0, 0] -> ?: "
	_n = []
	ip = inputs[5]
	for m in range(len(ip)):
		_n.append(neural_network.get_neighbors(ip, m))
	ll  = neural_network.think(array(_n))
	# lll = ll.astype(int)
	_l = ll.tolist()
	__l = []
	for l_ in _l:
		__l.append(int(l_[0] +  0.4999999))
	print(__l)
	cnt = 0
	for o in range(len(__l)):
		
		if o % 20 == 0:
			print("")
		if __l[o] == outputs[5][o]:
			cnt += 1
		sys.stdout.write(str(__l[o]))
	print("")
	for y, o in enumerate(outputs[5]):
		if y % 20  == 0:
			print("")
		sys.stdout.write(str(o))
	print('Count: ' + str(cnt))
	print("Percentage Correct: " + str(float(cnt) / 400.0 * 100) + "%")
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