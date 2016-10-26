__author__ = 'Dimitris'



import numpy as np
import tensorflow as tf

n_classes = 20

data_y = [
    [6],
    [7],
    [6],
    [8],
    [9],
    [10],
    [10],
    [12],
    [9],
    [6]
]

data_y = np.array(data_y)

y = tf.placeholder(tf.int32, [None, 1])

oh = tf.reshape(tf.one_hot(y,n_classes), shape=[-1, n_classes])

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for i in range(10):
        onehots, valuee = sess.run([oh], feed_dict={
            y: data_y
        })
        print(onehots)
        print(valuee)
























