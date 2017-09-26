import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance

mean = tf.Variable(tf.random_normal([2, ], 13., 1.), dtype=tf.float32)
cov = tf.Variable(5. * tf.eye(DNA_SIZE), dtype=tf.float32)
mvn = MultivariateNormalFullCovariance(loc=mean,covariance_matrix=cov)
make_kid = mvn.sample(N_POP)
