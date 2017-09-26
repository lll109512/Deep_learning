import tensorflow as tf
import numpy as np

class PolicyGradient(object):
    def __init__(self,
                n_actions,
                n_features,
                learning_rate=0.01,
                reward_decay=0.9,
                output_graph=False,):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self._buildnet()
        self.sess = tf.Session()
        if output_graph:    # 是否输出 tensorboard 文件
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def _buildnet(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, shape=[None,self.n_features], name='observation')
            self.tf_acts = tf.placeholder(tf.int32,shape=[None,],name = 'actions_num')
            self.tf_vt = tf.placeholder(tf.float32,shape=[None,],name = 'actions_value')

        layer = tf.layers.dense(self.tf_obs, units=10, activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),name='fc1')

        all_act = tf.layers.dense(layer,units = self.n_actions,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),name='fc2')

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')


        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts) # 所选 action 的概率 -log 值
            #neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)



    def store_transition(self,s,a,r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def choose_action(self,s):
        prob_weigths = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs:s[np.newaxis,:]})
        action = np.random.choice(self.n_actions,p = prob_weigths.ravel())
        return action

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs


    def learn(self):
        vt = self._discount_and_norm_rewards()
        self.sess.run(self.train_op, feed_dict={self.tf_obs:np.vstack(self.ep_obs),
                                                self.tf_acts:self.ep_as,
                                                self.tf_vt:vt})

        self.ep_as,self.ep_obs,self.ep_rs =[],[],[]
        return vt
