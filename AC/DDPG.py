import tensorflow as tf
import gym
import numpy as np


MAX_EPISODES = 200
MAX_EP_STEPS = 400
LR_A = 0.01  # learning rate for actor
LR_C = 0.01  # learning rate for critic
GAMMA = 0.9  # reward discount
REPLACE_ITER_A = 500
REPLACE_ITER_C = 300
MEMORY_CAPACITY = 7000
BATCH_SIZE = 32

RENDER = False
ENV_NAME = 'Pendulum-v0'


class DDPG(object):
    def __init__(self,n_actions,n_features,ac_lr,cr_lr,bound,reward_decay=0.9,
                repalce_iter_a = 200,repalce_iter_c = 200, MemorySize = 10000,Batch_size = 32):
        self.n_actions = n_actions
        self.n_features = n_features
        self.ac_lr = ac_lr
        self.cr_lr = cr_lr
        self.GAMMA = reward_decay
        self.a_bound = bound
        self.MemorySize = MemorySize
        self.pointer = 0
        self.repalce_iter_a = repalce_iter_a
        self.repalce_iter_c = repalce_iter_c
        self.a_replace_counter = 0
        self.c_replace_counter = 0
        self.Batch_size = Batch_size
        self.Memory = np.zeros([MemorySize,2*self.n_features + 1 + n_actions])
        self.sess = tf.Session()
        self._build()
        self.sess.run(tf.global_variables_initializer())


    def _build(self):
        self.S = tf.placeholder(tf.float32,[None,self.n_features],'S')
        self.R = tf.placeholder(tf.float32,[None,1],'R')
        self.S_ = tf.placeholder(tf.float32,[None,self.n_features],'S_')

        with tf.variable_scope('Actor'):
            self.A = self._build_a(self.S,'eval',True)
            A_  = self._build_a(self.S_, 'target', False)

        with tf.variable_scope('Critic'):
            self.V = self._build_c(self.S, self.A, 'eval', True)
            V_ = self._build_c(self.S_, A_, 'target', False)

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # q_target = self.R + self.GAMMA * V_
        # td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.V)
        TD_error = tf.reduce_mean(tf.square(self.R + self.GAMMA * V_ - self.V))
        self.ctrain = tf.train.AdamOptimizer(self.cr_lr).minimize(TD_error,var_list=self.ce_params)

        a_loss = - tf.reduce_mean(self.V)    # maximize the V
        self.atrain = tf.train.AdamOptimizer(self.ac_lr).minimize(a_loss, var_list=self.ae_params)
        pass

    def choose_action(self,s):
        return self.sess.run(self.A,{self.S:s[np.newaxis,:]})[0]

    def store_transition(self,s,a,r,s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.MemorySize  # replace the old memory with new memory
        self.Memory[index, :] = transition
        self.pointer += 1

    def learn(self):
        if self.a_replace_counter % self.repalce_iter_a == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.at_params, self.ae_params)])
        if self.c_replace_counter % self.repalce_iter_a == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.ct_params, self.ce_params)])
        self.a_replace_counter += 1;
        self.c_replace_counter += 1
        indices = np.random.choice(self.MemorySize, size=self.Batch_size)
        bt = self.Memory[indices,:]
        bs = bt[:,:self.n_features]
        ba = bt[:,self.n_features:self.n_features+self.n_actions]
        br = bt[:, -self.n_features - 1: -self.n_features]
        bs_ = bt[:,-self.n_features:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.A: ba, self.R: br, self.S_: bs_})
        pass


    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.n_actions, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.n_features, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.n_actions, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high
    ddpg = DDPG(a_dim, s_dim, LR_A, LR_C, a_bound,repalce_iter_a = REPLACE_ITER_A,
        repalce_iter_c = REPLACE_ITER_C, MemorySize = MEMORY_CAPACITY, Batch_size = BATCH_SIZE)
    var = 3  # control exploration

    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # Add exploration noise
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
            s_, r, done, info = env.step(a)

            ddpg.store_transition(s, a, r / 10, s_)

            if ddpg.pointer > MEMORY_CAPACITY:
                var *= .9995    # decay the action randomness
                ddpg.learn()

            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS-1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                if ep_reward > -1000:RENDER = True
                break
