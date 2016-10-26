__author__ = 'Dimitris'


import tensorflow as tf
import numpy as np

n_hidden = 10

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 100 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

def RNN(_X, _istate, _weights, _biases):
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) # (n_steps*batch_size, n_input)
    # Linear activation
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']
    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden)
    # Get lstm cell output
    outputs, states = tf.nn.rnn(lstm_cell, _X, initial_state=_istate)
    # Linear activation
    # Get inner loop last output
    return tf.matmul(outputs[-1], _weights['out']) + _biases['out']


# lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(100, state_is_tuple=True, forget_bias=1.0)
#
# istate = tf.placeholder("float", [None, 2*n_hidden]) #state & cell => 2x n_hidden
# x = tf.placeholder("float", [None, 4, 5])
#
# outputs, states = tf.nn.rnn(lstm_cell, x, initial_state=istate)
#
#
# sess = tf.InteractiveSession()
# tf.initialize_all_variables().run()








