#encoding="utf-8"
#!/usr/bin/env python

import numpy as np
import tensorflow as tf

class TextRNNCNN(object):
	def __init__(self, embedding_mat, non_static, hidden_unit, num_layers, sequence_length,
		num_classes, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

		self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
		self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
		self.batch_size = tf.placeholder(tf.int32)

		self.pad = tf.placeholder(tf.float32, [None, 1, embedding_size, 1], name='pad')
		self.real_len = tf.placeholder(tf.int32, [None], name='real_len')

		l2_loss = tf.constant(0.0)

		with tf.device('/cpu:0'), tf.name_scope('embedding'):
			if not non_static:
				W = tf.constant(embedding_mat, name='W')
			else:
				W = tf.Variable(embedding_mat, name='W')
			inputs = tf.nn.embedding_lookup(W, self.input_x)
			# self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
			# emb = tf.expand_dims(self.embedded_chars, -1)
		######################################################
		# begin to use the input the rnn model
		lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_unit)

		# lstm_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_unit)
		lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)

		cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers)
		self._initial_state = cell.zero_state(self.batch_size, tf.float32)

		# inputs = [tf.squeeze(input_, [1])
		# for input_ in tf.split(1, num_steps, inputs)]
		# 	outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)

		outputs = []
		state = self._initial_state
		with tf.variable_scope("RNN"):
			for time_step in range(sequence_length):
				if time_step > 0:
					tf.get_variable_scope().reuse_variables()
				(cell_output, state) = cell(inputs[:, time_step, :], state)
				outputs.append(cell_output)

		output = tf.reshape(tf.concat(1, outputs), [-1, sequence_length, embedding_size])

		#begin to use the output from rnn and input to cnn , cnn model
		emb = tf.expand_dims(output, -1)
		#######################################################################
		pooled_outputs = []
		for i, filter_size in enumerate(filter_sizes):
			with tf.name_scope("conv-maxpool-%s" % filter_size):
				 # Convolution Layer
				filter_shape = [filter_size, embedding_size, 1, num_filters]
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
				b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
				conv = tf.nn.conv2d(
                    emb,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
				# Apply nonlinearity
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
				# Maxpooling over the outputs
				pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
				pooled_outputs.append(pooled)

		# Combine all the pooled features
		num_filters_total = num_filters * len(filter_sizes)
		self.h_pool = tf.concat(3, pooled_outputs)
		self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

		# Add dropout
		with tf.name_scope("dropout"):
			self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

		with tf.name_scope('output'):
			self.W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name='W')
			b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
			l2_loss += tf.nn.l2_loss(W)
			l2_loss += tf.nn.l2_loss(b)
			self.scores = tf.nn.xw_plus_b(self.h_drop, self.W, b, name='scores')
			self.predictions = tf.argmax(self.scores, 1, name='predictions')

		with tf.name_scope('loss'):
			losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
			self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

		with tf.name_scope('accuracy'):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name='accuracy')

		with tf.name_scope('num_correct'):
			correct = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.num_correct = tf.reduce_sum(tf.cast(correct, 'float'))
