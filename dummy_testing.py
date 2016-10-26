__author__ = 'Dimitris'


import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

vocab_size = 20
embedding_size = 100

input_x = np.array([[1,2,3,4]])

W = tf.Variable( tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
embedded_chars = tf.nn.embedding_lookup(W, input_x)
embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
tf.initialize_all_variables().run()

embedded_chars_expanded.eval()[0].shape
embedded_chars.eval()[0].shape

W.eval().shape




































