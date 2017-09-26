import gym
from gym import wrappers
from My_RL_brain_with_priority import DeepQNetwork
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def run(episode_number,RL):
    env = gym.make('MountainCar-v0')   # 定义使用 gym 库中的那一个环境
    env = env.unwrapped # 不做这个会有很多限制
    # print(env.action_space) # 查看这个环境中可用的 action 有多少个
    # print(env.observation_space)    # 查看这个环境中可用的 state 的 observation 有多少个
    # print(env.observation_space.high)   # 查看 observation 最高取值
    # print(env.observation_space.low)    # 查看 observation 最低取值

    total_steps = 0
    ac_r = 0
    steps = []
    episodes = []
    # env = wrappers.Monitor(env, '/tmp/MountainCar-v0')
    # env.seed(1)
    for i_episode in range(episode_number):
        observation = env.reset()
        while True:
            if i_episode>20:
                env.render()

            action = RL.choose_action(observation)  # 选行为

            observation_, reward, done, info = env.step(action) # 获取下一个 state
            # position, velocity = observation_   # 细分开, 为了修改原配的 reward
            if done:
                reward = 10
            # else:
            #     reward = max(((position)/8), abs(velocity))
            # print(reward)

            RL.store_transition(observation, action, reward, observation_)
            #建议大小和memory size 一样
            if total_steps > 10000:
                RL.learn()  # 学习

            ac_r += reward

            if done:
                print('episode: ', i_episode,
                      'ac_r: ', round(ac_r, 2),
                      'epsilon: ', round(RL.epsilon, 2))
                steps.append(total_steps)
                episodes.append(i_episode)
                break

            observation = observation_
            total_steps += 1

    # RL.save("Double_DQN.ckpt")
    return np.vstack((episodes, steps))



    # RL.plot_cost()
if __name__ == '__main__':
    sess = tf.Session()
    with tf.variable_scope('N0'):
        RL0 = DeepQNetwork(n_actions=3,
                          n_features=2,
                          e_greedy=0.9,e_greedy_increment=0.00005,
                          memory_size=10000,sess=sess,
                          n_hidden_layer_nodes = 15,Double_DQN=False, Priority_DQN=False,
                          output_graph=False)

    with tf.variable_scope('N1'):
        RL1 = DeepQNetwork(n_actions=3,
                          n_features=2,
                          e_greedy=0.9,e_greedy_increment=0.00005,
                          memory_size=10000,sess=sess,
                          n_hidden_layer_nodes = 15,Double_DQN=True, Priority_DQN=False,
                          output_graph=False)
    with tf.variable_scope('N2'):
        RL2 = DeepQNetwork(n_actions=3,
                          n_features=2,
                          e_greedy=0.9,e_greedy_increment=0.00005,
                          memory_size=10000,sess=sess,
                          n_hidden_layer_nodes = 15,Double_DQN=True, Priority_DQN=True,
                          output_graph=False)

    with tf.variable_scope('N3'):
        RL3 = DeepQNetwork(n_actions=3,
                          n_features=2,
                          e_greedy=0.9,e_greedy_increment=0.000005,
                          memory_size=10000,sess=sess,
                          n_hidden_layer_nodes = 15,Double_DQN=True, Priority_DQN=True,Dueling_DQN=True,
                          output_graph=False)

    sess.run(tf.global_variables_initializer())
    # hist_hatural = run(20,RL0)
    hist_Duel = run(40, RL3)
    # his_D = run(20,RL1)
    # his_prio_D = run(20,RL2)
    # plt.plot(hist_hatural[0, :], hist_hatural[1, :] - hist_hatural[1, 0], c='b', label='Natural DQN')
    # plt.plot(his_D[0, :], his_D[1, :] - his_D[1, 0], c='g', label='Double DQN only')
    plt.plot(hist_Duel[0, :], hist_Duel[1, :] - hist_Duel[1, 0], c='y', label='+Duel DQN')
    # plt.plot(his_prio_D[0, :], his_prio_D[1, :] - his_prio_D[1, 0], c='r', label='DQN with prioritized replay and D')
    plt.legend(loc='best')
    plt.ylabel('total training time')
    plt.xlabel('episode')
    plt.grid()
    plt.show()
