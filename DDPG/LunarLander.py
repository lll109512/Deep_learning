import gym
import numpy
from RL_Brain import *


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    print('Action number = {}'.format(env.action_space.n)) # 查看这个环境中可用的 action 有多少个
    print('Observation space = {}'.format(env.observation_space.shape[0]))    # 查看这个环境中可用的 state 的 observation 有多少个
    Episode = 2000
    Memory_size  = 50000
    RENDER = False
    n_actions = env.action_space.n
    n_features = env.observation_space.shape[0]
    print(env.observation_space.high)   # 查看 observation 最高取值
    print(env.observation_space.low)

    env.seed(1)

    step = 0
    # RL = DDPG(n_actions, n_features, ac_lr = 0.00005, cr_lr = 0.00009, bound =0.99, reward_decay=0.99, repalce_iter_a = 500, repalce_iter_c = 400, MemorySize = Memory_size, Batch_size = 100)
    RL = DuelingDQNPrioritizedReplay(
        n_actions=n_actions, n_features=n_features, learning_rate=0.00005, e_greedy=0.95, reward_decay=0.99,
        batch_size=64, replace_target_iter=2000,
        memory_size=Memory_size, e_greedy_increment=0.00001)
    exp = 0
    for x in range(Episode):
        ac_r = 0
        s = env.reset()
        while True:
            if RL.epsilon > 0.7: env.render()
            a = RL.choose_action(s)
            s_, r, done, info = env.step(a)
            if r == -100:
                r = -10
            r /= 100
            RL.store_transition(s,a,r,s_)
            ac_r += r
            if step > Memory_size:
                RL.learn()
            if done:
                exp = ac_r*0.01 + exp*0.99
                land = '| Landed' if r == 1 else '| ------'
                print('Epi: ', x,
                      land,
                      '| Epi_R: ', round(ac_r, 2),
                      '| Running_R: ', round(exp, 2),
                      '| Epsilon: ', round(RL.epsilon, 3))
                break
            s = s_
            step+=1


    RL.save("Lander.ckpt")
