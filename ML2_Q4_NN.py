#!/usr/local/bin/python3

from random import seed
from random import random
import numpy as np
import pandas as pd
from math import exp
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
plt.style.use('ggplot')

# Initialize a network
def initialize_network():
	network = list()
	w1_data = np.array(pd.read_csv("./mlp_param/w1.csv", sep=',',header=None))
	print("weights w1:")
	print(len(w1_data[0]))

	b1_data = np.array(pd.read_csv("./mlp_param/b1.csv", sep=',',header=None))
	print("bias b1:")
	print(b1_data)

	z1 = np.append(w1_data, b1_data, axis=1)
	print(z1)
	print(z1[0])

	w2_data = np.array(pd.read_csv("./mlp_param/w2.csv", sep=',',header=None))
	print("weights w2:")
	print(w2_data)

	b2_data = np.array(pd.read_csv("./mlp_param/b2.csv", sep=',',header=None))
	print("bias b2:")
	print(b2_data)

	z2 = np.append(w2_data, b2_data, axis=1)
	print(z2)

	w3_data = np.array(pd.read_csv("./mlp_param/w3.csv", sep=',',header=None))
	print("weights w3:")
	print(w3_data)

	b3_data = np.array(pd.read_csv("./mlp_param/b3.csv", sep=',',header=None))
	print("bias b3:")
	print(b3_data)

	z3 = np.append(w3_data, b3_data, axis=1)
	print(z3)

	### hidden layer has 4 neurons
	hidden_layer1 = [	{
						'weights':z1[i],
						'delta': np.zeros(4),
                     	'size': 4
						} for i in range(4)]
	print("hidden_layer1")
	print(hidden_layer1)
	network.append(hidden_layer1)
	## hidden layer has 4 neurons
	hidden_layer2 = [	{
						'weights':z2[i],
						'delta': np.zeros(4),
                     	'size': 4
						} for i in range(4)]
	print("hidden_layer2")
	print(hidden_layer2)
	network.append(hidden_layer2)
	###output layer has one neuron
	output_layer = [	{
						'weights':z3[i],
						'delta': np.zeros(1),
                     	'size': 1
						} for i in range(1)]
	print("outer_layer")
	print(output_layer)
	network.append(output_layer)
	return network


# Calculate neuron activation for an input
def activate(weights, inputs):
	#### last column is the bias hence weights[-1]
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation


# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs

	return inputs


# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]

		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				#print("for j in range(len(layer)):")
				#print(j)
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				#print("for j in range(len(layer)) when i is the same as len(network)-1:")
				#print(j)
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
			print("printing neuron")
			neuron['gradient'] = np.outer(errors[j], neuron['output'])
			print("printing gradient")
			print(neuron)
			print("done printing deltas for:"+ str(i))



# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	loss_arr = []
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [1]
			expected[0] = 1
			sum_error += sum([0.5*((expected[i]-outputs[i])**2) for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.5f' % (epoch, l_rate, sum_error))
		loss_arr.append(sum_error)
		#print("to find out gradient of 2nd column of weight matrix w1 in hidden layer 1")
		print(epoch)
		for layer in network:
			print(layer)
	return loss_arr


def main():
	print("Hello World!")
	seed(1)

	input_nt = np.array([[-1, 0, 1]])

	input_txp = input_nt.transpose()
	print(input_txp)
	n_inputs = len(input_nt[0])
	print("n inputs")
	print(n_inputs)

	output_arr = np.array([[float(1.0)]])
	print (output_arr)

	n_outputs = len(set([row[-1] for row in output_arr]))
	print("n outputs")
	print(n_outputs)

	network = initialize_network()

	print("printing network:")
	for layer in network:
		print(layer)

	print("printed network")
	row = [-1, 0, 1, None]
	output = forward_propagate(network, row)
	print(output)

	# identifying loss after one round of forward propagation:
	loss = ((np.sum(output[0] - float(1.0))**2)) * .5
	print("loss after first round:")
	print(loss)

	print("network after one round of forward propagation")
	for layer in network:
		print(layer)

	##running one round of backpropagation
	expected = [1]
	backward_propagate_error(network, expected)

	print("network after one round of back propagation!")

	for layer in network:
		print(layer)

		for dicts in layer:
			print(dicts)
			print(dicts['delta'])

	loss_arr = train_network(network, input_nt, 1.0, 10, 1)
	print(loss_arr)
	print("final Layer in network--")
	for layer in network:
		print(layer)

	epochs_arr = [i+1 for i in range(10)]
	print(epochs_arr)

	plt.plot(epochs_arr, loss_arr)
	plt.scatter(epochs_arr, loss_arr)
	plt.legend()
	plt.show()





if __name__ == "__main__":
	main()
