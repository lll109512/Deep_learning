import numpy as np
import tensorflow as tf


class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            n_hidden_layer_nodes = 10,
            output_graph=False,
            Restore_path=None,
            Double_DQN=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.n_hidden_layer_nodes = n_hidden_layer_nodes
        self.memory_counter = 0
        self.Double_DQN = Double_DQN

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        self.sess = tf.Session()
        if Restore_path == None:
            self.sess.run(tf.global_variables_initializer())
        else:
            save_path = Restore_path
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, save_path)
            print("Model restored.")

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.loss = []

    def _build_net(self):
        #Firstly, build net work construction
        #Evaluate_net
        # Input
        self.s = tf.placeholder(tf.float32, shape=[None,self.n_features], name='s')
        self.q_target = tf.placeholder(tf.float32, shape=[None,self.n_actions], name='q_t')

        with tf.variable_scope('eval_net'):
            #construction
            c_name = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            w_initializer = tf.random_normal_initializer(0.,0.3)
            b_initializer = tf.constant_initializer(0.1)
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', shape=[self.n_features,self.n_hidden_layer_nodes],initializer=w_initializer,collections=c_name)
                b1 = tf.get_variable('b1', shape=[1,self.n_hidden_layer_nodes],initializer=b_initializer,collections=c_name)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', shape=[self.n_hidden_layer_nodes,self.n_actions],initializer=w_initializer,collections=c_name)
                b2 = tf.get_variable('b2', shape=[1,self.n_actions],initializer=b_initializer,collections=c_name)

            #Logist
            l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
            l2 = tf.matmul(l1,w2) + b2
            self.q_eval = l2

            #Train eval_net
            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target , self.q_eval))

            with tf.variable_scope('train'):
                self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)


        #Freezing net work
        #Input
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')

        #construction
        c_names = ['freeze_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        with tf.variable_scope('Freezing'):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', shape=[self.n_features,self.n_hidden_layer_nodes],initializer=w_initializer,collections=c_name)
                b1 = tf.get_variable('b1', shape=[1,self.n_hidden_layer_nodes],initializer=b_initializer,collections=c_name)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', shape=[self.n_hidden_layer_nodes,self.n_actions],initializer=w_initializer,collections=c_name)
                b2 = tf.get_variable('b2', shape=[1,self.n_actions],initializer=b_initializer,collections=c_name)

            #Logist
            l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)
            l2 = tf.matmul(l1,w2) + b2
            self.q_next = l2

    def store_transition(self,s,a,r,s_):
        transition = np.hstack((s,[a,r],s_))
        index = self.memory_counter % self.memory_size
        self.memory[index,:] = transition
        self.memory_counter += 1

    def choose_action(self,observation):
        observation = observation[np.newaxis,:]

        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s:observation})
            action = np.argmax(actions_value)
            # print(actions_value)
        else:
            action = np.random.randint(0,self.n_actions)
        return action

    def _replace_freeze_params(self):
        f_params = tf.get_collection('freeze_net_params')
        # print(f_params)
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(f, e) for f, e in zip(f_params, e_params)])
    #
    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_freeze_params()
            print('Replace freeze network')
            print()

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size,size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter,size = self.batch_size)

        samples = self.memory[sample_index,:]
        #分别从两个NN中得到对下一个action(s_)的Q_value
        #如果是Double_DQN,则先从最新的NN返回的next中选择可以获得最大Q的Action，然后使用这个Action再去freezed的NN中得到真实值
        #最终使用这个真实值的Q值去更新
        q_next,q_eval4next = self.sess.run([self.q_next,self.q_eval],
                                            feed_dict={self.s_:samples[:,-self.n_features:],
                                                        self.s:samples[:,-self.n_features:]})
        q_eval = self.sess.run(self.q_eval, feed_dict={self.s: samples[:, :self.n_features]})

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = samples[:, self.n_features].astype(int)
        reward = samples[:, self.n_features + 1]

        if self.Double_DQN:
            max_act4next = np.argmax(q_eval4next,axis=1)
            selected_q_next = q_next[batch_index,max_act4next]
        else:
            selected_q_next = np.max(q_next, axis=1)

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: samples[:, :self.n_features],
                                                self.q_target: q_target})

        if self.epsilon < self.epsilon_max:
            self.epsilon += self.epsilon_increment
        self.learn_step_counter +=1




    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def save(self,path):
        self.saver = tf.train.Saver()
        save_path = self.saver.save(self.sess, path)
        print('save success')
