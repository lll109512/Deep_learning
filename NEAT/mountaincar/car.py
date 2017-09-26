import gym
import neat
import numpy as np
import visualize


class NM:
    def __init__(self, GENERATION_EP, EP_STEP, CONFIG):
        self.GENERATION_EP = GENERATION_EP
        self.EP_STEP = EP_STEP
        self.CONFIG = CONFIG
        self.env = gym.make('MountainCar-v0')
        self.env = self.env.unwrapped
        pass

    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            net = neat.nn.RecurrentNetwork.create(genome, config)
            ep_r = []
            for _ in range(self.GENERATION_EP):
                accumulative_r = 0.
                observation = self.env.reset()
                for t in range(self.EP_STEP):
                    action_values = net.activate(observation)
                    action = np.argmax(action_values)
                    observation_, reward, done, _ = self.env.step(action)
                    if reward == -100: reward = -30
                    reward /= 100
                    if done:
                        break
                    accumulative_r += reward
                    observation = observation_
                ep_r.append(accumulative_r)
            genome.fitness = np.min(ep_r) / np.float64(self.EP_STEP)
        pass

    def run(self):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation, self.CONFIG)
        pop = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        stats = neat.StatisticsReporter()
        pop.add_reporter(neat.StdOutReporter(True))
        pop.add_reporter(stats)
        pop.add_reporter(neat.Checkpointer(5))

        pop.run(self.eval_genomes, 20)

        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)
        pass

    def evaluation(self, checkpoint):
        p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-%i' % checkpoint)
        winner = p.run(self.eval_genomes, 1)
        net = neat.nn.RecurrentNetwork.create(winner, p.config)
        node_names = {-1: 'In0', -2: 'In1', -3: 'In3', -4: 'In4', 0: 'act1', 1: 'act2', 2 : 'act3'}
        visualize.draw_net(p.config, winner, False, node_names=node_names)
        while True:
            s = self.env.reset()
            while True:
                self.env.render()
                a = np.argmax(net.activate(s))
                s, r, done, _ = self.env.step(a)
                if done:
                    break


if __name__ == '__main__':
    e = NM(10, 300, "config")
    e.run()
    e.evaluation(19)
