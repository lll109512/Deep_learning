import numpy as np
import tensorflow as tf
import load
from sklearn.metrics import confusion_matrix


Train_data_X, Train_data_y = load.Train_data_X, load.Train_data_y
Test_data_X, Test_data_y = load.Test_data_X, load.Test_data_y
print('Train:', Train_data_X.shape, Train_data_y.shape)
print('Test :', Test_data_X.shape, Test_data_y.shape)

image_size = load.image_size
num_lables = load.num_lables
num_channels = load.num_channels


def get_chunk(samples, labels, chunkSize):
	'''
	Iterator/Generator: get a batch of data
	这个函数是一个迭代器/生成器，用于每一次只得到 chunkSize 这么多的数据
	用于 for loop， just like range() function
	'''
	if len(samples) != len(labels):
	    raise Exception('Length of samples and labels must equal')
	stepStart = 0  # initial step
	i = 0
	while stepStart < len(samples):
	    stepEnd = stepStart + chunkSize
	    if stepEnd < len(samples):
	        yield i, samples[stepStart:stepEnd], labels[stepStart:stepEnd]
	        i += 1
	    stepStart = stepEnd


class Network():

	def __init__(self, num_hidden, batch_size):
	    self.num_hidden = num_hidden
	    # Hyper parameters
	    self.batch_size = batch_size
	    self.test_batch_size = 500

	    # graph related
	    self.graph = tf.Graph()
	    self.tf_train_samples = None
	    self.tf_train_lables = None
	    self.tf_test_samples = None
	    self.tf_test_lables = None
	    self.tf_test_prediction = None

	    # initialize
	    self.merged = None
	    self.define_graph()
	    self.session = tf.Session(graph=self.graph)
	    writer = tf.train.SummaryWriter('./board', graph=self.graph)

	def define_graph(self):
		with self.graph.as_default():
			with tf.name_scope('input'):
				self.tf_train_samples = tf.placeholder(tf.float32, shape=(
					self.batch_size, image_size, image_size, num_channels))
				self.tf_train_lables = tf.placeholder(
					tf.float32, shape=(self.batch_size, num_lables))
				self.tf_test_samples = tf.placeholder(tf.float32, shape=(
					self.test_batch_size, image_size, image_size, num_channels))
			with tf.name_scope('fc1'):
				fc1_weights = tf.Variable(tf.truncated_normal(
					[image_size * image_size, self.num_hidden], stddev=0.1))
				fc1_biases = tf.Variable(
					tf.constant(0.1, shape=[self.num_hidden]))
			with tf.name_scope('fc2'):
				fc2_weights = tf.Variable(tf.truncated_normal(
					[self.num_hidden, num_lables], stddev=0.1))
				fc2_biases = tf.Variable(tf.constant(0.1, shape=[num_lables]))

            def model(data):
                shape = data.get_shape().as_list()
                print(data.get_shape(), shape)
                reshape = tf.reshape(
                    data, [shape[0], shape[1] * shape[2] * shape[3]])
                print(reshape.get_shape(), fc1_weights.get_shape(),
                      fc1_biases.get_shape())
                hidden = tf.nn.relu(
                    tf.matmul(reshape, fc1_weights) + fc1_biases)

                return tf.matmul(hidden, fc2_weights) + fc2_biases

            logist = model(self.tf_train_samples)
			with tf.name_scope('loss')
            	self.loss = tf.reduce_mean(
	                tf.nn.softmax_cross_entropy_with_logits(logist, self.tf_train_lables))
				tf.scalar_summary('Loss',self.loss)

            self.optimizer = tf.train.GradientDescentOptimizer(
                0.0001).minimize(self.loss)

            self.train_prediction = tf.nn.softmax(logist)
            self.test_prediction = tf.nn.softmax(model(self.tf_test_samples))

			self.merged = tf.merge_all_summaries()

    def run(self):
        '''
        用到Session
        '''
        # private function
        def print_confusion_matrix(confusionMatrix):
            print('Confusion    Matrix:')
            for i, line in enumerate(confusionMatrix):
                print(line, line[i] / np.sum(line))
            a = 0
            for i, column in enumerate(np.transpose(confusionMatrix, (1, 0))):
                a += (column[i] / np.sum(column)) * (np.sum(column) / 26000)
                print(column[i] / np.sum(column))
            print('\n', np.sum(confusionMatrix), a)

        with self.session as session:
            tf.initialize_all_variables().run()

            # 训练
            print('Start Training')
            # batch 1000
            for i, samples, labels in get_chunk(Train_data_X, Train_data_y, chunkSize=self.batch_size):
                _, l, predictions, summary = session.run(
                    [self.optimizer, self.loss, self.train_prediction,self.merged],
                    feed_dict={self.tf_train_samples: samples,
                               self.tf_train_lables: labels}
                )

				self.writer.add_summary(summary,i)
                # labels is True Labels
                accuracy, _ = self.accuracy(predictions, labels)
                if i % 50 == 0:
                    print('Minibatch loss at step %d: %f' % (i, l))
                    print('Minibatch accuracy: %.1f%%' % accuracy)
            ###

            # 测试
            accuracies = []
            confusionMatrices = []
            for i, samples, labels in get_chunk(Test_data_X, Test_data_y, chunkSize=self.test_batch_size):
                result = self.test_prediction.eval(
                    feed_dict={self.tf_test_samples: samples})
                accuracy, cm = self.accuracy(
                    result, labels, need_confusion_matrix=True)
                accuracies.append(accuracy)
                confusionMatrices.append(cm)
                print('Test Accuracy: %.1f%%' % accuracy)
            print(' Average  Accuracy:', np.average(accuracies))
            print('Standard Deviation:', np.std(accuracies))
            print_confusion_matrix(np.add.reduce(confusionMatrices))
            ###

    def accuracy(self, predictions, labels, need_confusion_matrix=False):
        '''
                计算预测的正确率与召回率
                @return: accuracy and confusionMatrix as a tuple
        '''
        _predictions = np.argmax(predictions, 1)
        _labels = np.argmax(labels, 1)
        cm = confusion_matrix(
            _labels, _predictions) if need_confusion_matrix else None
        # == is overloaded for numpy array
        accuracy = (100.0 * np.sum(_predictions ==
                                   _labels) / predictions.shape[0])
        return accuracy, cm

if __name__ == '__main__':
    net = Network(num_hidden=128, batch_size=100)
    net.run()
