""""Creates dummy dataset for sorting numbers."""
import numpy as np
import itertools
import random
import sys
sys.dont_write_bytecode = True

def write_file(filename, data):
	file = open(filename, 'w')
	for entry in data:
		file.write(str(entry) +'\n')
	file.close()


def convertToOneHot(vector, num_classes=None):
	"""
	Converts an input 1-D vector of integers into an output
	2-D array of one-hot vectors, where an i'th input value
	of j will set a '1' in the i'th row, j'th column of the
	output array.

	Example:
	    v = np.array((1, 0, 4))
	    one_hot_v = convertToOneHot(v)
	    print one_hot_v

	    [[0 1 0 0 0]
	     [1 0 0 0 0]
	     [0 0 0 0 1]]
	"""

	assert isinstance(vector, np.ndarray)
	assert len(vector) > 0

	if num_classes is None:
		num_classes = np.max(vector)+1
	else:
		assert num_classes > 0
		assert num_classes >= np.max(vector)

	result = np.zeros(shape=(len(vector), num_classes))
	result[np.arange(len(vector)), vector] = 1
	return np.asarray(result.astype(int))


def get_one_hot_repr(inputs, dataset_config):
	one_hot_repr = []
	for i in inputs:
		one_hot_repr.append(convertToOneHot(i, dataset_config.upper_limit+1))
	return np.asarray(one_hot_repr)


def get_training_and_test_dataset(dataset_config):
	np.random.seed(dataset_config.seed)
	int_range = range(1,dataset_config.upper_limit+1)

	if dataset_config.number_repetition:
		encoder_inputs_train = np.random.randint(low = 1,
						 				   high = dataset_config.upper_limit,
						 				   size = (dataset_config.seq_lenght, dataset_config.training_example_count))
	else:
		encoder_inputs_train = []
		for i in range(dataset_config.training_example_count):
			encoder_inputs_train.append(random.sample(int_range, dataset_config.seq_lenght))
		encoder_inputs_train = np.transpose(np.asarray(encoder_inputs_train))	

	decoder_inputs_train = [sorted(inputs) for inputs in np.transpose(encoder_inputs_train)]
	decoder_inputs_train = np.transpose(decoder_inputs_train)


	if dataset_config.number_repetition:
		encoder_inputs_test = np.random.randint(low = 1,
						 				   high = dataset_config.upper_limit,
						 				   size = (dataset_config.seq_lenght, dataset_config.test_example_count))
	else:
		encoder_inputs_test = []
		for i in range(dataset_config.test_example_count):
			encoder_inputs_test.append(random.sample(int_range, dataset_config.seq_lenght))
		encoder_inputs_test = np.transpose(np.asarray(encoder_inputs_test))


	decoder_inputs_test = [sorted(inputs) for inputs in np.transpose(encoder_inputs_test)]
	decoder_inputs_test = np.transpose(decoder_inputs_test)
	
	write_file("dataset/encoder_train_data.txt", encoder_inputs_train)
	write_file("dataset/decoder_train_data.txt", decoder_inputs_train)
	write_file("dataset/encoder_test_data.txt", encoder_inputs_test)
	write_file("dataset/decoder_test_data.txt", decoder_inputs_test)
	
	return (encoder_inputs_train, decoder_inputs_train), (encoder_inputs_test, decoder_inputs_test)


class temp(object):
	seed = 10
	upper_limit = 10
	lenght =3
	training_example_count = 2
	test_example_count = 3
	number_repetition = False
	one_hot_repr = True


# print get_training_and_test_dataset(d)