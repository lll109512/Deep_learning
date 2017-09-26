from My_RL_Brain import *
import numpy as np
import gym
import tensorflow as tf

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.seed(1)  # reproducible
    env = env.unwrapped
    N_F = env.observation_space.shape[0]
    N_A = env.action_space.n
    Episode_number = 3000
    RENDER = False
    Threshold = 200
    OUTPUT_GRAPH = False
    Sess = tf.Session()
    actor = Actor(Sess, N_F, N_A, lr = 0.001)
    critic = Critic(Sess,N_F,lr = 0.01)

    Sess.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs/", sess.graph)

    for i in range(Episode_number):
        s = env.reset()
        t = 0
        track_r = []
        while True:
            if RENDER: env.render()

            a = actor.choose_action(s)
            s_, r, done, info = env.step(a)
            if done : r = -20
            track_r.append(r)
            td_error = critic.learn(s,r,s_)
            actor.learn(s, a, td_error)

            s = s_
            t += 1

            if done:
                ep_rs_sum = sum(track_r)
                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                if running_reward > Threshold: RENDER = True  # rendering
                print("episode:", i, "  reward:", int(running_reward))
                break
