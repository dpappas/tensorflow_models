
import numpy as np
import tensorflow as tf

n_classes = 20
n_input = 1
vocabulary_size = 50
total_speakers = 100
speaker_embedding_size = 100
text_embedding_size = 100
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

text_x =  tf.placeholder(tf.int32, [None, n_input])
speaker_x = tf.placeholder(tf.int32, [None, n_input])
y = tf.placeholder(tf.int32, [None, 1])

# x = tf.placeholder(tf.int32, [10, n_input])
# print(x.get_shape().as_list())
# y = tf.placeholder(tf.float32, [10, n_classes])
# y = tf.placeholder(tf.float32, [None, n_classes])
# print(x.get_shape().as_list())

weights = {
    'text_embeddings': tf.Variable( tf.random_uniform([vocabulary_size, text_embedding_size], -1.0, 1.0)),
    'speaker_embeddings': tf.Variable( tf.random_uniform([total_speakers, speaker_embedding_size], -1.0, 1.0)),
    'out': tf.Variable(tf.random_normal([text_embedding_size+speaker_embedding_size, n_classes]))
}

biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Parameters
learning_rate = 0.1
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


# class tf.nn.rnn_cell.BasicLSTMCell
# num_units: int, The number of units in the LSTM cell.
# forget_bias: float, The bias added to forget gates (see above).
# input_size: Deprecated and unused.
# state_is_tuple: If True, accepted and returned states are 2-tuples of the c_state and m_state. If False, they are concatenated along the column axis. The latter behavior will soon be deprecated.
# activation: Activation function of the inner states.

#tf.one_hot(indices, depth, on_value=None, off_value=None, axis=None, dtype=None, name=None)
#
#

# Create model
def create_model(text_x, speaker_x, weights, biases, dropout):
    print(text_x.get_shape().as_list())
    # handle text
    # lookup the embedding
    text_embed = tf.nn.embedding_lookup(weights['text_embeddings'], text_x)
    print('text_embed : ', text_embed.get_shape().as_list())
    # reshape embedding to pass from max_pool
    text_embed = tf.reshape(text_embed, shape=[-1, n_input, text_embedding_size, 1])
    print('text_embed : ', text_embed.get_shape().as_list())
    # max_pool of 5 embeddings
    max_text = tf.nn.max_pool(text_embed, ksize=[1, n_input, 1, 1], strides=[1, 1, 1, 1],padding='VALID')
    print('max_text : ', max_text.get_shape().as_list())
    # reshape embedding to pass from matmul output
    max_text = tf.reshape(max_text, shape=[-1, text_embedding_size])
    print('max_text : ', max_text.get_shape().as_list())
    #  handle speakers
    # lookup the embedding
    speaker_embed = tf.nn.embedding_lookup(weights['speaker_embeddings'], speaker_x )
    print('speaker_embed : ', speaker_embed.get_shape().as_list())
    # reshape embedding to pass from max_pool
    speaker_embed = tf.reshape(speaker_embed, shape=[-1, n_input, speaker_embedding_size, 1])
    print('speaker_embed : ', speaker_embed.get_shape().as_list())
    # max_pool of 5 embeddings
    max_speaker = tf.nn.max_pool(text_embed, ksize=[1, n_input, 1, 1], strides=[1, 1, 1, 1],padding='VALID')
    print('max_speaker : ', max_speaker.get_shape().as_list())
    # reshape embedding to pass from matmul output
    max_speaker = tf.reshape(max_speaker, shape=[-1, text_embedding_size])
    print('max_speaker : ', max_speaker.get_shape().as_list())
    # concatenate the 2 embeddings
    conc = tf.concat(1, [max_text, max_speaker])
    print('conc : ',conc.get_shape().as_list())
    # print(weights['out'].get_shape().as_list())
    # print(biases['out'].get_shape().as_list())
    mul = tf.matmul(conc, weights['out'])
    print('mul : ',mul.get_shape().as_list())
    out = tf.add(mul, biases['out'])
    print('out : ',out.get_shape().as_list())
    out = tf.nn.relu(out)
    print('out : ',out.get_shape().as_list())
    return out

pred = create_model(text_x, speaker_x, weights, biases, keep_prob)

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
    for i in range(1000):
        sess.run(optimizer, feed_dict={
            text_x : data_x,
            speaker_x : data_x,
            y: data_y,
            keep_prob: dropout
        })
        # loss, acc, predictions, prediction_probas = sess.run([cost, accuracy, preds, pred], feed_dict={
        loss, acc, predictions = sess.run([cost, accuracy, preds], feed_dict={
            text_x : data_x,
            speaker_x : data_x,
            y: data_y,
            keep_prob: 1.
        })
        print(predictions)
        print(loss)
        print(acc)
























