import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(2)

class SumTree(object):
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story the data with it priority in tree and data frameworks.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity    # for all priority values
        self.tree = np.zeros(2*capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)    # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add_new_priority(self, p, data):
        leaf_idx = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data # update data_frame
        self.update(leaf_idx, p)    # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]

        self.tree[tree_idx] = p
        self._propagate_change(tree_idx, change)

    def _propagate_change(self, tree_idx, change):
        """change the sum of priority value in all parent nodes"""
        parent_idx = (tree_idx - 1) // 2
        self.tree[parent_idx] += change
        if parent_idx != 0:
            self._propagate_change(parent_idx, change)

    def get_leaf(self, lower_bound):
        leaf_idx = self._retrieve(lower_bound)  # search the max leaf priority based on the lower_bound
        data_idx = leaf_idx - self.capacity + 1
        return [leaf_idx, self.tree[leaf_idx], self.data[data_idx]]

    def _retrieve(self, lower_bound, parent_idx=0):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        left_child_idx = 2 * parent_idx + 1
        right_child_idx = left_child_idx + 1

        if left_child_idx >= len(self.tree):    # end search when no more child
            return parent_idx

        if self.tree[left_child_idx] == self.tree[right_child_idx]:
            return self._retrieve(lower_bound, np.random.choice([left_child_idx, right_child_idx]))
        if lower_bound <= self.tree[left_child_idx]:  # downward search, always search for a higher priority node
            return self._retrieve(lower_bound, left_child_idx)
        else:
            return self._retrieve(lower_bound-self.tree[left_child_idx], right_child_idx)

    @property
    def root_priority(self):
        return self.tree[0]     # the root

class Memory(object):   # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6     # [0~1] convert the importance of TD error to priority
    beta = 0.4      # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add_new_priority(max_p, transition)   # set the max p for new p

    def sample(self, n):
        batch_idx, batch_memory, ISWeights = [], [], []
        segment = self.tree.root_priority / n
        self.beta = np.min([1, self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.root_priority
        #Maxiwi 是用于normalize的 ，公式为 1/maxi wi
        #wi = (1/N * 1/P(i))^beta
        #为了让wi最大化，P(i)也就是概率需要最小化。
        #所以在上面选择的是min_prob
        #P(i) = P(i)^alpha / SUMi(P(i))
        maxiwi = np.power(self.tree.capacity * min_prob, -self.beta)  # for later normalizing ISWeights
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            lower_bound = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(lower_bound)
            prob = p / self.tree.root_priority
            ISWeights.append(self.tree.capacity * prob)
            batch_idx.append(idx)
            batch_memory.append(data)

        ISWeights = np.vstack(ISWeights)
        ISWeights = np.power(ISWeights, -self.beta) / maxiwi  # normalize
        return batch_idx, np.vstack(batch_memory), ISWeights

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def _get_priority(self, error):
        error += self.epsilon  # avoid 0
        #去掉之后发现收敛速度更快……
        #估计不去掉会导致有些明显的error不能被及时训练到
        # clipped_error = np.clip(error, 0, self.abs_err_upper)
        return np.power(error, self.alpha)


class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.005,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=500,
            memory_size=10000,
            batch_size=32,
            e_greedy_increment=None,
            n_hidden_layer_nodes = 10,
            output_graph=False,
            Restore_path=None,
            Double_DQN=False,
            #Priority DQN 要与 Double DQN 一起使用
            Priority_DQN=False,
            Dueling_DQN=False,
            sess=None,
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
        self.Priority_DQN = Priority_DQN
        self.Dueling_DQN = Dueling_DQN

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        if self.Priority_DQN:
            self.memory = Memory(capacity = self.memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        if sess is None:
            self.sess = tf.Session()
            if Restore_path == None:
                self.sess.run(tf.global_variables_initializer())
            else:
                save_path = Restore_path
                self.saver = tf.train.Saver()
                self.saver.restore(self.sess, save_path)
                print("Model restored.")
        else:
            self.sess = sess

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
        if self.Priority_DQN:
            self.ISWeights = tf.placeholder(tf.float32,shape=[None,1],name='ISWeights')

        with tf.variable_scope('eval_net'):
            #construction
            c_name = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            w_initializer = tf.random_normal_initializer(0.,0.3)
            b_initializer = tf.constant_initializer(0.1)
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', shape=[self.n_features,self.n_hidden_layer_nodes],initializer=w_initializer,collections=c_name)
                b1 = tf.get_variable('b1', shape=[1,self.n_hidden_layer_nodes],initializer=b_initializer,collections=c_name)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            if self.Dueling_DQN:
                with tf.variable_scope('value'):
                    w2 = tf.get_variable('w2', shape=[self.n_hidden_layer_nodes,1],initializer=w_initializer,collections=c_name)
                    b2 = tf.get_variable('b2', shape=[1,1],initializer=b_initializer,collections=c_name)
                    self.V = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Adventage'):
                    w2 = tf.get_variable('w2', [self.n_hidden_layer_nodes, self.n_actions], initializer=w_initializer, collections=c_name)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_name)
                    self.A = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Q'):
                    self.q_eval = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)

            else:
                with tf.variable_scope('l2'):
                    w2 = tf.get_variable('w2', shape=[self.n_hidden_layer_nodes,self.n_actions],initializer=w_initializer,collections=c_name)
                    b2 = tf.get_variable('b2', shape=[1,self.n_actions],initializer=b_initializer,collections=c_name)
                    self.q_eval = tf.matmul(l1, w2) + b2


            #Train eval_net
            with tf.variable_scope('loss'):
                if self.Priority_DQN:
                    self.abs_error = tf.reduce_sum(tf.abs(self.q_target - self.q_eval),axis=1)
                    self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target , self.q_eval))
                else:
                    self.loss = tf.reduce_mean(tf.squared_difference(self.q_target , self.q_eval))

            with tf.variable_scope('train'):
                self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)


        #Freezing net work
        #Input
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')

        #construction
        c_name = ['freeze_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        with tf.variable_scope('Freezing'):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', shape=[self.n_features,self.n_hidden_layer_nodes],initializer=w_initializer,collections=c_name)
                b1 = tf.get_variable('b1', shape=[1,self.n_hidden_layer_nodes],initializer=b_initializer,collections=c_name)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            if self.Dueling_DQN:
                with tf.variable_scope('value'):
                    w2 = tf.get_variable('w2', shape=[self.n_hidden_layer_nodes,1],initializer=w_initializer,collections=c_name)
                    b2 = tf.get_variable('b2', shape=[1,1],initializer=b_initializer,collections=c_name)
                    self.V = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Adventage'):
                    w2 = tf.get_variable('w2', [self.n_hidden_layer_nodes, self.n_actions], initializer=w_initializer, collections=c_name)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_name)
                    self.A = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Q'):
                    self.q_next = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)
            else:
                with tf.variable_scope('l2'):
                    w2 = tf.get_variable('w2', shape=[self.n_hidden_layer_nodes,self.n_actions],initializer=w_initializer,collections=c_name)
                    b2 = tf.get_variable('b2', shape=[1,self.n_actions],initializer=b_initializer,collections=c_name)
                    self.q_next = tf.matmul(l1,w2) + b2

    def store_transition(self,s,a,r,s_):
        transition = np.hstack((s,[a,r],s_))
        if self.Priority_DQN:
            self.memory.store(transition)
        else:
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

        if self.Priority_DQN:
            tree_idx, samples, ISWeights = self.memory.sample(self.batch_size)
        else:
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

        if self.Priority_DQN:
            _,abs_errors,self.cost =self.sess.run([self._train_op,self.abs_error,self.loss],
                                                feed_dict={self.ISWeights:ISWeights,
                                                            self.s: samples[:,:self.n_features],
                                                            self.q_target:q_target})
            for i,idx in enumerate(tree_idx):
                self.memory.update(idx, abs_errors[i])

        else:
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.s: samples[:, :self.n_features],
                                                    self.q_target: q_target})

        if self.epsilon < self.epsilon_max:
            self.epsilon += self.epsilon_increment
        self.learn_step_counter +=1


    def save(self,path):
        self.saver = tf.train.Saver()
        save_path = self.saver.save(self.sess, path)
        print('save success')
