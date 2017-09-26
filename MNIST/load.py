import gzip
import os
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from sklearn.metrics import confusion_matrix
import pandas as pd

mnist = read_data_sets("../Datas/MNIST_data", one_hot=True)

Image_size = 28
Num_of_labels = 10
Num_of_channel = 1

data=pd.read_csv('test.csv').values


print("Train:",mnist.train.images.shape,mnist.train.labels.shape)
print("Test:",mnist.test.images.shape,mnist.test.labels.shape)


class CNN():
    def __init__(self, batch_size,num_of_hidden,conv_depth,pooling_scale,patch_size):
        self.test_batch_size = 500
        self.batch_size = batch_size

        # Hyper Parameters
        self.pooling_scale = pooling_scale
        self.num_of_hidden = num_of_hidden
        self.patch_size = patch_size
        self.conv1_depth = conv_depth
        self.conv2_depth = conv_depth*2
        self.last_conv_depth = self.conv2_depth
        self.pooling_stride = self.pooling_scale

        #Graph
        self.graph = tf.Graph()
        self.tf_train_data = None
        self.tf_train_labels = None
        self.tf_test_data = None
        self.tf_test_labels = None
        self.tf_test_prediction = None


        # 统计
        self.merged = None
        self.train_summaries = []
        self.test_summaries = []

        #initailize
        self.define_graph()
        self.session = tf.Session(graph=self.graph)
        self.writer = tf.train.SummaryWriter('./board', self.graph)
        self.saver = None
        self.save_path = 'model/default.ckpt'


    def define_graph(self):
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

        with self.graph.as_default():
            with tf.name_scope('inputs'):
                self.tf_train_data = tf.placeholder(
                tf.float32, shape=(self.batch_size,Image_size*Image_size*Num_of_channel), name='tf_train_data'
                )
                self.tf_train_labels = tf.placeholder(
                tf.float32, shape=(self.batch_size,Num_of_labels), name='tf_train_labels'
                )
                self.tf_test_data = tf.placeholder(
                tf.float32, shape=(self.test_batch_size,Image_size*Image_size*Num_of_channel), name='tf_test_data'
                )
            with tf.name_scope('Conv1'):
                conv1_weights = weight_variable([self.patch_size, self.patch_size, Num_of_channel, self.conv1_depth])
                conv1_biases = bias_variable([self.conv1_depth])
            with tf.name_scope('Conv2'):
                conv2_weights = weight_variable([self.patch_size, self.patch_size, self.conv1_depth, self.conv2_depth])
                conv2_biases = bias_variable([self.conv2_depth])

            with tf.name_scope('Fc_1'):
                down_scale = self.pooling_scale * self.pooling_scale
                Fc1_weights = weight_variable([Image_size//down_scale * Image_size//down_scale * self.last_conv_depth, self.num_of_hidden])
                fc1_biases = bias_variable([self.num_of_hidden])
                self.train_summaries.append(tf.histogram_summary('fc1_weights', Fc1_weights))
                self.train_summaries.append(tf.histogram_summary('fc1_biases', fc1_biases))

            with tf.name_scope('Fc_2'):
                Fc2_weights = weight_variable([self.num_of_hidden,Num_of_labels])
                fc2_biases = bias_variable([Num_of_labels])
                self.train_summaries.append(tf.histogram_summary('fc2_weights', Fc2_weights))
                self.train_summaries.append(tf.histogram_summary('fc2_biases', fc2_biases))

            def model(data,train = True):
                with tf.name_scope('Conv1_model'):
                    data_image = tf.reshape(data, [-1,28,28,1])
                    h_conv1 = tf.nn.relu(conv2d(data_image, conv1_weights) + conv1_biases)
                    if not train:
                        self.visualize_filter_map(h_conv1,how_many=self.conv1_depth,display_size=28,name='Conv1')
                    h_pool1 = max_pool_2x2(h_conv1)
                    if not train:
                        self.visualize_filter_map(h_pool1,how_many=self.conv1_depth,display_size=14,name='pool1')

                with tf.name_scope('Conv2_model'):
                    h_conv2 = tf.nn.relu(conv2d(h_pool1, conv2_weights) + conv2_biases)
                    if not train:
                        self.visualize_filter_map(h_conv2,how_many=self.conv2_depth,display_size=14,name='Conv2')
                    hidden = max_pool_2x2(h_conv2)
                    if not train:
                        self.visualize_filter_map(hidden,how_many=self.conv2_depth,display_size=7,name='pool2')


                with tf.name_scope('Fc1_model'):
                    shape = hidden.get_shape().as_list()
                    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
                    h_fc1 = tf.nn.relu(tf.matmul(reshape, Fc1_weights) + fc1_biases)

                with tf.name_scope('Fc2_model'):
                    #dropout
                    h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)
                    return tf.nn.relu(tf.matmul(h_fc1_drop, Fc2_weights) + fc2_biases)


            logits = model(self.tf_train_data)
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.tf_train_labels))
                self.train_summaries.append(tf.scalar_summary('Loss', self.loss))

            #with tf.name_scope('learning_rate'):
            #    global_step = tf.Variable(0)
            #    lr = 0.0001
            #    dr = 0.99
            #    learning_rate = tf.train.exponential_decay(lr, global_step*self.batch_size, 1000, dr, staircase=True)

            with tf.name_scope('optimizer'):
                self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

            with tf.name_scope('train_prediction'):
                self.train_prediction = tf.nn.softmax(logits, name='train_prediction')
            with tf.name_scope('train_prediction'):
                self.test_prediction = tf.nn.softmax(model(self.tf_test_data,train =False), name='test_prediction')


            self.merged_train_summary = tf.merge_summary(self.train_summaries)
            self.merged_test_summary = tf.merge_summary(self.test_summaries)

            self.saver = tf.train.Saver(tf.all_variables())

            pass

    def run(self,Continue = False):
        with self.session as session:
            if Continue:
                self.saver = tf.train.Saver(tf.all_variables())
                self.saver.restore(session,self.save_path)
                print('Start Continue Training')
            else:
                tf.initialize_all_variables().run()
                print('Start Frist Training')


            for i in range(1):
                batch = mnist.train.next_batch(self.batch_size)
                _, l, predictions,summary = session.run([self.optimizer, self.loss, self.train_prediction,self.merged_train_summary],
                    feed_dict={self.tf_train_data: batch[0], self.tf_train_labels: batch[1]})
                self.writer.add_summary(summary, i)

                if i % 50 == 0:
                    accuracy,_ = self.accuracy(predictions,batch[1])
                    print('Minibatch loss at step %d: %f' % (i, l))
                    print('Minibatch accuracy: %.1f%%' % accuracy)


            #test
            accuracies = []
            confusionMatrices = []
            for i in range(1):
                batch = mnist.test.next_batch(self.test_batch_size)
                result, summary = session.run(
                    [self.test_prediction, self.merged_test_summary],
                    feed_dict={self.tf_test_data: batch[0]}
                    )
                self.writer.add_summary(summary, i)
                accuracy,cm = self.accuracy(result,batch[1],need_confusion_matrix=True)
                accuracies.append(accuracy)
                confusionMatrices.append(cm)
                print('Test Accuracy: %.1f%%' % accuracy)
            print(' Average  Accuracy:', np.average(accuracies))
            print('Standard Deviation:', np.std(accuracies))
            self.saver = tf.train.Saver(tf.all_variables())
            self.saver.save(session,self.save_path)
        pass


    def test(self,data):
        print('Before session')
        if self.saver is None:
            self.define_graph()
        with tf.Session(graph=self.graph) as session:
            self.saver.restore(session,self.save_path)
            result =[]
            for i,batch in test_data_iterator(data,500):
                result.append(session.run([self.test_prediction],feed_dict={self.tf_test_data: batch})[0])


        return result


    def accuracy(self,predictions,labels,need_confusion_matrix = False):
        _predictions = np.argmax(predictions, 1)
        _labels = np.argmax(labels, 1)
        cm = confusion_matrix(_labels, _predictions) if need_confusion_matrix else None
        accuracy = (100.0 * np.sum(_predictions == _labels) / predictions.shape[0])
        return accuracy,cm
        pass

    def visualize_filter_map(self, tensor, *, how_many, display_size, name):
        #print(tensor.get_shape)
        filter_map = tensor[-1]
        #print(filter_map.get_shape())
        filter_map = tf.transpose(filter_map, perm=[2, 0, 1])
        #print(filter_map.get_shape())
        filter_map = tf.reshape(filter_map, (how_many, display_size, display_size, 1))
        #print(how_many)
        self.test_summaries.append(tf.image_summary(name, tensor=filter_map, max_images=how_many))
pass

def test_data_iterator(samples,chunkSize):
    Length = len(samples)
    stepStart = 0  # initial step
    i = 0
    while stepStart + chunkSize <= Length:
        yield i, samples[stepStart:stepStart + chunkSize]
        stepStart = stepStart + chunkSize
        i += 1


if __name__ == '__main__':
    net = CNN(num_of_hidden=1024, batch_size=50, patch_size=5, conv_depth=32, pooling_scale=2)
    net.run(Continue =True)
    #net.define_graph()
    # result = net.test(data)
    # print(len(result))
    # new_list = []
    # print(len(result[0]))
    # for i,line in enumerate(result):
    #     new_list.extend(result[i])
    # new_list = np.array(new_list)
    # new_list = np.argmax(new_list,axis=1)
    # new_list = pd.Series(new_list)
    # new_list.to_csv('result.csv')
