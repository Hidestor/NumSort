"""Seq2Seq model."""
import tensorflow as tf
import sys
sys.dont_write_bytecode = True

class SimpleSeq2SeqModel(object):

	def __init__(self, config):
		self.source_vocab_size = config.source_vocab_size
		self.target_vocab_size = config.target_vocab_size
		self.cell_size = config.cell_size
		self.num_layers = config.num_layers
		self.input_size = config.input_size
		self.learning_rate = tf.Variable(float(config.learning_rate), trainable=False, dtype=tf.float32)
		self.cell_type = config.cell_type
		self.output_projection = config.output_projection
		self.saver = tf.train.Saver()
		self.encoder_time_steps = config.encoder_time_steps
		self.decoder_time_steps = config.decoder_time_steps
		self.keep_prob = config.keep_prob
		self.training = config.training

		if self.cell_type == "lstm":
			self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size)
		elif self.cell_type == "gru":
			self.cell = tf.nn.rnn_cell.GRUCell(self.cell_size)

		if self.num_layers > 1:
			self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * num_layers)
		
		self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, output_keep_prob=self.keep_prob)

		#Feed for inputs.
		self.encoder_inputs = []
		self.decoder_inputs = []

		if config.one_hot_repr:
			self.cell = tf.nn.rnn_cell.OutputProjectionWrapper(self.cell, config.upper_limit+1)
			for i in range(self.encoder_time_steps):
				self.encoder_inputs.append(tf.placeholder(tf.float32,
														  shape=[None, config.upper_limit+1],
														  name="encoder{0}".format(i)))

			for i in xrange(self.decoder_time_steps):
				self.decoder_inputs.append(tf.placeholder(tf.float32,
														  shape=[None, config.upper_limit+1],
														  name="decoder{0}".format(i)))

			self.targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]

			#Append the end token as the last target output.
			self.targets.append(self.decoder_inputs[0])

		else:
			for i in range(self.encoder_time_steps):
				self.encoder_inputs.append(tf.placeholder(tf.int32,
														  shape=[None],
														  name="encoder{0}".format(i)))

			for i in xrange(self.decoder_time_steps):
				self.decoder_inputs.append(tf.placeholder(tf.int32,
														  shape=[None],
														  name="decoder{0}".format(i)))

			self.targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]

			#Append the end token as the last target output.
			self.targets.append(self.decoder_inputs[0])



		if config.one_hot_repr:
			self.logits, self.state = tf.nn.seq2seq.basic_rnn_seq2seq(encoder_inputs = self.encoder_inputs,
							  decoder_inputs = self.decoder_inputs,
							  cell = self.cell,
							  dtype=tf.float32)

		else:
			if config.attention_mechanism:
				self.logits, self.state = tf.nn.seq2seq.embedding_attention_seq2seq(self.encoder_inputs,
																					self.decoder_inputs,
																					self.cell,
																					num_encoder_symbols = self.source_vocab_size,
																					num_decoder_symbols = self.target_vocab_size,
																					embedding_size = self.input_size,
																					feed_previous = self.training)
			else:
				self.logits, self.state = tf.nn.seq2seq.embedding_rnn_seq2seq(self.encoder_inputs,
																			   self.decoder_inputs,
																			   self.cell,
																			   num_encoder_symbols=self.source_vocab_size,
																			   num_decoder_symbols=self.target_vocab_size,
																			   embedding_size=self.input_size,
																			   output_projection=self.output_projection,
																			   feed_previous=self.training)


		if config.one_hot_repr:
			self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = self.logits,
																targets = self.targets)
			self.loss = tf.reduce_sum(self.loss, [0,1,2])

		else:
			self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.logits,
																	   labels = self.targets)
			self.loss = tf.reduce_sum(self.loss, [0,1])


		self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
		
		self.predictions  = tf.nn.top_k(self.logits, 1)

	def assign_lr(self, sess, lr_value):
		sess.run(tf.assign(self.learning_rate, lr_value))

		# self.correct_predictions = tf.reduce_sum(tf.cast(self.predictions, tf.int32))