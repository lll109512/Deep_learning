import gym
from gym import wrappers
from My_RL_brain import DeepQNetwork

def run(episode_number):
    env = gym.make('MountainCar-v0')   # 定义使用 gym 库中的那一个环境
    env = env.unwrapped # 不做这个会有很多限制
    print(env.action_space) # 查看这个环境中可用的 action 有多少个
    print(env.observation_space)    # 查看这个环境中可用的 state 的 observation 有多少个
    print(env.observation_space.high)   # 查看 observation 最高取值
    print(env.observation_space.low)    # 查看 observation 最低取值

    RL = DeepQNetwork(n_actions=env.action_space.n,
                      n_features=env.observation_space.shape[0],
                      learning_rate=0.01, e_greedy=0.9,
                      replace_target_iter=100, memory_size=5000,
                      n_hidden_layer_nodes = 15,Double_DQN=True,Restore_path="Double_DQN.ckpt",
                      output_graph=True)
    total_steps = 0
    ac_r = 0
    # env = wrappers.Monitor(env, '/tmp/MountainCar-v0')

    for i_episode in range(episode_number):
        observation = env.reset()
        while True:
            env.render()

            action = RL.choose_action(observation)  # 选行为
            
            observation_, reward, done, info = env.step(action) # 获取下一个 state
            # position, velocity = observation_   # 细分开, 为了修改原配的 reward
            if done:
                reward = 1
            # else:
            #     reward = max(((position)/8), abs(velocity))
            # print(reward)
            else:
                reward /= 10

            RL.store_transition(observation, action, reward, observation_)
            if total_steps > 1000:
                RL.learn()  # 学习

            ac_r += reward

            if done:
                print('episode: ', i_episode,
                      'ac_r: ', round(ac_r, 2),
                      'epsilon: ', round(RL.epsilon, 2))
                break

            observation = observation_
            total_steps += 1

    RL.save("Double_DQN.ckpt")


    # RL.plot_cost()
if __name__ == '__main__':
    run(400)
