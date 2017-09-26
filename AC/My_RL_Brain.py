import tensorflow as tf
import numpy as np


class Actor(object):
    def __init__(self,sess,n_feature,n_actions,lr = 0.001):
        self.sess = sess
        self.n_feature = n_feature
        self.n_actions = n_actions
        self.lr = lr

        self._build()

    def _build(self):
        self.s = tf.placeholder(tf.float32,[1,self.n_feature],'state')
        self.a = tf.placeholder(tf.int32,None,'act')
        self.TD_error = tf.placeholder(tf.float32,None,'TD_error')
        with tf.name_scope('Actor'):
            layer = tf.layers.dense(self.s, 20, activation=tf.nn.relu,
                kernel_initializer=tf.random_uniform_initializer(0.,.3), bias_initializer=tf.constant_initializer(0.1),
                name = 'layera')

            self.act_prob = tf.layers.dense(layer,self.n_actions, activation= tf.nn.softmax,
                kernel_initializer=tf.random_uniform_initializer(0.,.3), bias_initializer=tf.constant_initializer(0.1),
                name = 'act_prob')

        with tf.name_scope('exp_v'):
            log_prob = tf.log(self.act_prob[0,self.a])
            self.exp_v = log_prob * self.TD_error

        with tf.name_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(-self.exp_v)


    def choose_action(self,s):
        s = s[np.newaxis,:]
        prob = self.sess.run(self.act_prob,{self.s: s})
        return np.random.choice(self.n_actions,p=prob.ravel())

    def learn(self,s,a,TD_error):
        s = s[np.newaxis,:]
        _,exp_v = self.sess.run([self._train_op,self.exp_v],{self.s:s,self.a:a,self.TD_error:TD_error})
        return exp_v

class Critic(object):
    def __init__(self,sess,n_feature,reward_decay = 0.9,lr = 0.002):
        self.sess = sess
        self.n_feature = n_feature
        self.lr = lr
        self.GAMMA = reward_decay

        self._build()

    def _build(self):
        self.r = tf.placeholder(tf.float32,None,'reward')
        self.s = tf.placeholder(tf.float32,[1,self.n_feature], 'state')
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")

        with tf.name_scope('Critic'):
            layer = tf.layers.dense(self.s,20,activation = tf.nn.relu,
                kernel_initializer=tf.random_uniform_initializer(0.,.3), bias_initializer=tf.constant_initializer(0.1),
                name = 'layerc')
            self.v = tf.layers.dense(layer,1,activation = None,
                kernel_initializer=tf.random_uniform_initializer(0.,.3), bias_initializer=tf.constant_initializer(0.1),
                name = 'V')

        with tf.name_scope('loss'):
            self.TD_error = self.r + self.GAMMA*self.v_ - self.v
            self.loss = tf.square(self.TD_error)

        with tf.name_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


    def learn(self,s,r,s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.sess.run(self.v,{self.s: s_})
        _,TD_error = self.sess.run([self._train_op,self.TD_error],{self.s: s,self.r:r,self.v_ : v_})

        return TD_error
