
import numpy as np
import tensorflow as tf

n_classes = 20
n_input = 1
vocabulary_size = 50
embedding_size = 100
b_size = 10

n_input = 5 # no of feats : edw 5 dinw 5 ades apo indices

# data_x = np.random.randint(1,10,[b_size,n_input])
# # data_x = np.random.randint(1,10,[10,n_input])
# data_y = np.random.rand(b_size,n_classes)

data_x = [
    [1,2,3,4,5],
    [2,3,4,5,6],
    [1,2,3,4,5],
    [3,4,5,6,7],
    [4,5,6,7,8],
    [5,6,7,8,9],
    [5,6,7,8,9],
    [7,8,9,10,11],
    [4,5,6,7,8],
    [1,2,3,4,5],
]

data_x = np.array(data_x)

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

# x = tf.placeholder(tf.int32, [10, n_input])
x = tf.placeholder(tf.int32, [None, n_input])
# print(x.get_shape().as_list())
# y = tf.placeholder(tf.float32, [10, n_classes])
# y = tf.placeholder(tf.float32, [None, n_classes])
y = tf.placeholder(tf.int32, [None, 1])
# print(x.get_shape().as_list())

weights = {
    'embeddings': tf.Variable( tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)),
    'out': tf.Variable(tf.random_normal([embedding_size, n_classes]))
}

biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Parameters
learning_rate = 0.001
# training_iters = 200000
# batch_size = 128
# display_step = 10
dropout = 0.75 # Dropout, probability to keep units

# tf.nn.embedding_lookup(params, ids, partition_strategy='mod', name=None, validate_indices=True)
# ids must be x
# embeddings = tf.Variable( tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
# h'
# nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size)))
# nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# tf.nn.max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)
# value: A 4-D Tensor with shape [batch, height, width, channels] and type tf.float32.
# ksize: A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
# strides: A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor.
# padding: A string, either 'VALID' or 'SAME'.
# data_format: A string. 'NHWC' and 'NCHW' are supported.
# name: Optional name for the operation.


#tf.one_hot(indices, depth, on_value=None, off_value=None, axis=None, dtype=None, name=None)
#
#

# Create model
def create_model(x, weights, biases, dropout):
    print(x.get_shape().as_list())
    #
    embed = tf.nn.embedding_lookup(weights['embeddings'], x)
    print(embed.get_shape().as_list())
    #
    embed = tf.reshape(embed, shape=[-1, n_input, embedding_size, 1])
    print(embed.get_shape().as_list())
    #
    max = tf.nn.max_pool(embed, ksize=[1, n_input, 1, 1], strides=[1, 1, 1, 1],padding='VALID')
    print(max.get_shape().as_list())
    max = tf.reshape(max, shape=[-1, embedding_size])
    print(max.get_shape().as_list())
    # print(weights['out'].get_shape().as_list())
    # print(biases['out'].get_shape().as_list())
    mul = tf.matmul(max, weights['out'])
    print(mul.get_shape().as_list())
    out = tf.add(mul, biases['out'])
    print(out.get_shape().as_list())
    out = tf.nn.relu(out)
    print(out.get_shape().as_list())
    return out

pred = create_model(x, weights, biases, keep_prob)

oh = tf.reshape(tf.one_hot(y,n_classes), shape=[-1, n_classes])

# Define loss and optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,tf.one_hot(tf.to_int32(y),n_classes)))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, oh))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
# correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
preds = tf.cast(tf.argmax(pred, 1), tf.int32)
correct_pred = tf.equal(preds, y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for i in range(10):
        sess.run(optimizer, feed_dict={
            x: data_x,
            y: data_y,
            keep_prob: dropout
        })
        loss, acc, predictions = sess.run([cost, accuracy, preds], feed_dict={
            x: data_x,
            y: data_y,
            keep_prob: 1.
        })
        print(predictions)
        print(loss)
        print(acc)
























