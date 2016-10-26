


# import tensorflow as tf
# import numpy as np
# from tensorflow.models.rnn import rnn
#
# with tf.device('/cpu:0'):
#     #
#     vocab_size = 20
#     no_of_speakers = 1
#     text_embedding_size = 100
#     speaker_embedding_size = 100
#     #
#     text_x = np.array(
#         [
#             [1,2,3,4],
#             [1,2,3,4],
#             [1,2,3,4]
#         ]
#     )
#     speaker_x = np.array(
#         [
#             [0,0,0,0],
#             [0,0,0,0],
#             [0,0,0,0]
#         ]
#     )
#     #
#     W = tf.Variable( tf.random_uniform([vocab_size, text_embedding_size], -1.0, 1.0), name="W")
#     embedded_text = tf.nn.embedding_lookup(W, text_x)
#     # embedded_text_expanded = tf.expand_dims(embedded_text, -1)
#     #
#     W2 = tf.Variable( tf.random_uniform([ no_of_speakers , speaker_embedding_size ], -1.0, 1.0), name="W")
#     embedded_speaker = tf.nn.embedding_lookup(W2,speaker_x )
#     # embedded_speaker_expanded = tf.expand_dims(embedded_speaker, -1)
#     #
#     joined = tf.concat(2,[embedded_text, embedded_speaker])
#     # joined_expanded = tf.concat(2,[embedded_text_expanded, embedded_speaker_expanded])
#
# sess = tf.InteractiveSession()
# tf.initialize_all_variables().run()


import tensorflow as tf
import numpy as np

vocab_size = 20
text_embedding_size = 100
b_size = 3
timesteps = 4
no_of_speakers = 1
speaker_embedding_size = 100
lstm_size = 25
dense_size = 100

graph = tf.Graph()
with graph.as_default():
    text_x = tf.placeholder(tf.int32, shape=[b_size,timesteps])
    W = tf.Variable( tf.random_uniform([vocab_size, text_embedding_size], -1.0, 1.0), name="W")
    embedded_text = tf.nn.embedding_lookup(W, text_x)
    speaker_x = tf.placeholder(tf.int32, shape=[b_size,timesteps])
    W2 = tf.Variable( tf.random_uniform([ no_of_speakers , speaker_embedding_size ], -1.0, 1.0), name="W")
    embedded_speaker = tf.nn.embedding_lookup(W2,speaker_x )
    joined = tf.concat(2,[embedded_text, embedded_speaker])
    #Parameters:
    # Input gate: input, previous output, and bias.
    joined_emb_size = text_embedding_size+speaker_embedding_size
    ix = tf.Variable(tf.truncated_normal([joined_emb_size, lstm_size], -0.1, 0.1))
    im = tf.Variable(tf.truncated_normal([lstm_size, lstm_size], -0.1, 0.1))
    ib = tf.Variable(tf.zeros([1, lstm_size]))
    # Forget gate: input, previous output, and bias.
    fx = tf.Variable(tf.truncated_normal([joined_emb_size, lstm_size], -0.1, 0.1))
    fm = tf.Variable(tf.truncated_normal([lstm_size, lstm_size], -0.1, 0.1))
    fb = tf.Variable(tf.zeros([1, lstm_size]))
    # Memory cell: input, state and bias.
    cx = tf.Variable(tf.truncated_normal([joined_emb_size, lstm_size], -0.1, 0.1))
    cm = tf.Variable(tf.truncated_normal([lstm_size, lstm_size], -0.1, 0.1))
    cb = tf.Variable(tf.zeros([1, lstm_size]))
    # Output gate: input, previous output, and bias.
    ox = tf.Variable(tf.truncated_normal([joined_emb_size, lstm_size], -0.1, 0.1))
    om = tf.Variable(tf.truncated_normal([lstm_size, lstm_size], -0.1, 0.1))
    ob = tf.Variable(tf.zeros([1, lstm_size]))
    # Variables saving state across unrollings.
    # saved_output = tf.Variable(tf.zeros([batch_size, lstm_size]), trainable=False)
    # saved_state = tf.Variable(tf.zeros([batch_size, lstm_size]), trainable=False)
    # Classifier weights and biases.
    w = tf.Variable(tf.truncated_normal([lstm_size, joined_emb_size], -0.1, 0.1))
    b = tf.Variable(tf.zeros([joined_emb_size]))
    # Definition of the cell computation.
    def lstm_cell(i, o, state):
        """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
        Note that in this formulation, we omit the various connections between the
        previous state and the gates."""
        input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
        forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
        update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
        state = forget_gate * state + input_gate * tf.tanh(update)
        output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
        return output_gate * tf.tanh(state), state

with tf.Session(graph=graph) as session:
    with tf.device("/cpu:0"):
        tf.initialize_all_variables().run()
        feed_dict = {
            text_x : np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4]]),
            speaker_x : np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]])
        }
        joined = session.run( [joined], feed_dict=feed_dict)



















