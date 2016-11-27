"""Seq2seq model example for sorting numbers."""
import sys
sys.dont_write_bytecode = True
from model import SimpleSeq2SeqModel
import tensorflow as tf
import numpy as np
import data_utils


class Config(object):
	cell_size = 512
	num_layers = 1
	batch_size = 32
	input_size = 128
	learning_rate = 0.1
	cell_type = "gru"
	encoder_end_padding = False
	decoder_go_padding = True #This has to be always true.
	seed = 200
	output_projection = None
	number_epochs = 100
	lr_decay = 0.5
	decay_epoch = 20
	keep_prob = 0.5
	intializations = 0.1
	#Dataset-parameters
	#Assuming that the lower limit is always '1'. 
	#<go> & <end> padding have index '0'.
	upper_limit = 10
	source_vocab_size = upper_limit + 2#Change to upper_limit+1 if end_padding = True.
	target_vocab_size = upper_limit + 2
	seq_lenght = 5
	training_example_count = 5
	test_example_count = 2
	encoder_time_steps = seq_lenght
	decoder_time_steps = seq_lenght
	number_repetition = True
	one_hot_repr = True
	attention_mechanism = False
	training = True


class TestConfig(object):
	"""Extremely small config for debugging."""
	cell_size = 10
	num_layers = 1
	batch_size = 2
	input_size = 12
	learning_rate = 0.1
	cell_type = "gru"
	encoder_end_padding = False
	decoder_go_padding = True #This has to be always true.
	seed = 200
	output_projection = None
	number_epochs = 100
	lr_decay = 0.5
	decay_epoch = 3
	keep_prob = 0.8
	intializations = 0.1
	#Dataset-parameters
	#Assuming that the lower limit is always '1'. 
	#<go> & <end> padding have index '0'.
	upper_limit = 10
	source_vocab_size = upper_limit + 2#Change to upper_limit+1 if end_padding = True.
	target_vocab_size = upper_limit + 2
	seq_lenght = 5
	training_example_count = 5
	test_example_count = 2
	encoder_time_steps = seq_lenght
	decoder_time_steps = seq_lenght
	number_repetition = True
	one_hot_repr = True
	attention_mechanism = False
	training = True


def get_input_feed_directory(encoder_inputs, decoder_inputs, model):
	input_feed = {}
	for l in xrange(len(encoder_inputs)):
		input_feed[model.encoder_inputs[l].name] = encoder_inputs[l]
	for l in xrange(len(decoder_inputs)):
		input_feed[model.decoder_inputs[l].name] = decoder_inputs[l]

	return input_feed


def run_epoch(sess, model, config, encoder_inputs, decoder_inputs):
	total_batches = (encoder_inputs.shape[0])/config.batch_size
	for batch_number in range(total_batches):
		batch_encoder_inputs = encoder_inputs[:,batch_number*config.batch_size:(batch_number+1)*config.batch_size]
		batch_decoder_inputs = decoder_inputs[:,batch_number*config.batch_size:(batch_number+1)*config.batch_size]
		input_feed = get_input_feed_directory(batch_encoder_inputs, batch_decoder_inputs, model)

		sess.run([model.optimizer], input_feed)

	#last batch if len(train_x)/config.batch_size leaves reminder.
	if len(encoder_inputs[0])%config.batch_size != 0:
		last_batch_count = len(encoder_inputs)%config.batch_size
		batch_encoder_inputs = encoder_inputs[:,-last_batch_count:]
		batch_decoder_inputs = decoder_inputs[:,-last_batch_count:]
		input_feed = get_input_feed_directory(batch_encoder_inputs, batch_decoder_inputs, model)
		sess.run([model.optimizer], input_feed)


def pad_inputs(encoder_inputs, decoder_inputs, config):
	if config.decoder_go_padding:
		zeros = np.zeros(shape=(1, decoder_inputs.shape[1]), dtype = int)
		decoder_inputs = np.concatenate((zeros, decoder_inputs), axis=0)

	if config.encoder_end_padding:
		padding = np.full(shape=[1,encoder_inputs.shape[1]], fill_value=config.upper_limit+1, dtype=int)
		encoder_inputs = np.concatenate((encoder_inputs, padding), axis=0)

	return encoder_inputs, decoder_inputs


def adjust_timesteps(config):
	if config.decoder_go_padding:
		config.decoder_time_steps += 1
	if config.encoder_end_padding:
		config.encoder_time_steps += 1


def train(config):
	train, test = data_utils.get_training_and_test_dataset(config)
	encoder_inputs_train, decoder_inputs_train = pad_inputs(train[0], train[1], config)
	encoder_inputs_test, decoder_inputs_test = pad_inputs(test[0], test[1], config)

	adjust_timesteps(config)

	if config.one_hot_repr:
		encoder_inputs_train_one_hot = data_utils.get_one_hot_repr(encoder_inputs_train, config)
		decoder_inputs_train_one_hot = data_utils.get_one_hot_repr(decoder_inputs_train, config)
		encoder_inputs_test_one_hot = data_utils.get_one_hot_repr(encoder_inputs_test, config)
		decoder_inputs_test_one_hot = data_utils.get_one_hot_repr(decoder_inputs_test, config)

	initializer = tf.random_uniform_initializer(-config.intializations, config.intializations, seed = config.seed)

	with tf.Graph().as_default():
		with tf.variable_scope("block", initializer=initializer):
			model = SimpleSeq2SeqModel(config)
			sess = tf.Session()
			sess.run([tf.initialize_all_variables()])
			
			# Input feed: encoder inputs, decoder inputs, as provided.
			if config.one_hot_repr:
				train_feed = get_input_feed_directory(encoder_inputs_train_one_hot, decoder_inputs_train_one_hot, model)
				test_feed = get_input_feed_directory(encoder_inputs_test_one_hot, decoder_inputs_test_one_hot, model)
				encoder_inputs, decoder_inputs = encoder_inputs_train_one_hot, decoder_inputs_train_one_hot

			else:
				train_feed = get_input_feed_directory(encoder_inputs_train, decoder_inputs_train, model)
				test_feed = get_input_feed_directory(encoder_inputs_test, decoder_inputs_test, model)
				encoder_inputs, decoder_inputs = encoder_inputs_train, decoder_inputs_train

			for epoch in xrange(1, config.number_epochs):
				run_epoch(sess, model, config, encoder_inputs, decoder_inputs)
				print "Epoch number %s, Loss : %s"%(epoch, sess.run([model.loss], train_feed)[0]), "test loss : %s"%sess.run([model.loss], test_feed)[0]
				if epoch%config.decay_epoch==0:
					lr_value = sess.run([model.learning_rate])[0]*config.lr_decay
					print "New learning rate %s"%lr_value
					model.assign_lr(sess, lr_value)
					config.decay_epoch = config.decay_epoch*2

				model.training = False
				model.keep_prob = 1.0
				if config.one_hot_repr:
					encoder_single_example, decoder_single_example = encoder_inputs_test_one_hot[:,0,:].reshape([-1, 1, config.upper_limit+1]), decoder_inputs_test_one_hot[:,0,:].reshape([-1, 1, config.upper_limit+1])
					single_example_feed = get_input_feed_directory(encoder_single_example, decoder_single_example, model)
					print encoder_inputs_test[:,0], decoder_inputs_test[:,0], sess.run([model.predictions], single_example_feed)[0][1].reshape([-1])
				else:
					encoder_single_example, decoder_single_example = encoder_inputs_test[:,0].reshape([-1,1]), decoder_inputs_test[:,0].reshape([-1,1])
					single_example_feed = get_input_feed_directory(encoder_single_example, decoder_single_example, model)
					print encoder_single_example.reshape([-1]), decoder_single_example.reshape([-1]),sess.run([model.predictions], single_example_feed)[0][1].reshape([-1])
				model.training = True
				model.keep_prob = config.keep_prob


# train(TestConfig)