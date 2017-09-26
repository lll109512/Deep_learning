import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

# Hyper Parameters
BATCH_SIZE = 200
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
LR = 0.01               # learning rates

mnist = input_data.read_data_sets('./mnist', one_hot=True)      # they has been normalized to range (0,1)
test_x = mnist.test.images[:5000]
test_y = mnist.test.labels[:5000]

print(test_x.shape)
print(test_y.shape)
tf_x = tf.placeholder(tf.float32,[None,TIME_STEP * INPUT_SIZE])
x_reshape = tf.reshape(tf_x, [-1,TIME_STEP,INPUT_SIZE])
tf_y = tf.placeholder(tf.int32, shape=[None,10])

rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units = 64)
out_puts, (h_c,h_n) = tf.nn.dynamic_rnn(rnn_cell, x_reshape, dtype=tf.float32, time_major=False)

output = tf.layers.dense(out_puts[:, -1, :], 10)

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)

train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1] # return (acc, update_op), and create 2 local variables

sess = tf.Session()

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)

for step in range(1000):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    if step % 50 == 0:      # testing
        accuracy_ = sess.run(accuracy, {tf_x: test_x, tf_y: test_y})
        print('train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)
