import gym
import neat
import numpy as np
import visualize


class NM:
    def __init__(self, GENERATION_EP, EP_STEP, CONFIG):
        self.GENERATION_EP = GENERATION_EP
        self.EP_STEP = EP_STEP
        self.CONFIG = CONFIG
        pass

    def eval_genomes(self, genomes, config):
        env = gym.make("LunarLander-v2")
        env = env.unwrapped
        for genome_id, genome in genomes:
            net = neat.nn.RecurrentNetwork.create(genome, config)
            ep_r = []
            # success_sum = 0
            for _ in range(self.GENERATION_EP):
                accumulative_r = 0.
                observation = env.reset()
                c = 0
                for t in range(self.EP_STEP):
                    action_values = net.activate(observation)
                    action = np.argmax(action_values)
                    observation_, reward, done, _ = env.step(action)
                    if reward == -100: reward = -30
                    accumulative_r += reward
                    if done:
                        if reward == 100:
                            print("success")
                            # success_sum += 1
                            c = t
                        break
                    observation = observation_
                ep_r.append(accumulative_r/(t+1))
            genome.fitness = np.average(ep_r)
            # genome.fitness = (success_sum / self.GENERATION_EP)
        pass

    def run(self,restore = None):
        if restore == None:
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation, self.CONFIG)
            pop = neat.Population(config)
        else:
            pop = neat.Checkpointer.restore_checkpoint('neat-checkpoint-%i' % restore)

        # Add a stdout reporter to show progress in the terminal.
        stats = neat.StatisticsReporter()
        pop.add_reporter(neat.StdOutReporter(True))
        pop.add_reporter(stats)
        pop.add_reporter(neat.Checkpointer(5))
        pe = neat.ParallelEvaluator(4, self.eval_genomes)
        pop.run(pe.eval_function, 30)

        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)
        pass



    def evaluation(self, checkpoint):
        p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-%i' % checkpoint)
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        winner = p.run(self.eval_genomes, 1)
        genomes = stats.best_unique_genomes(5)
        best_net = []
        for g in genomes:
            best_net.append(neat.nn.RecurrentNetwork.create(g, p.config))

        # net = neat.nn.RecurrentNetwork.create(winner, p.config)
        node_names = {-1: 'In0', -2: 'In1', -3: 'In3', -4: 'In4', -5: 'In5', -6: 'In6', -7: 'In7',-8: 'In8', 0: 'act1', 1: 'act2', 2 : 'act3', 3 : 'act4'}
        visualize.draw_net(p.config, winner, False, node_names=node_names)

        env = gym.make("LunarLander-v2")
        env = env.unwrapped
        while True:
            s = env.reset()
            for i in range(450):
                env.render()
                votes = np.zeros((4,))
                for net in best_net:
                    votes[np.argmax(net.activate(s))] += 1
                # best_action = np.argmax(net.activate(s))
                best_action = np.argmax(votes)
                s, r, done, _ = env.step(best_action)
                if done:
                    if r == 100:
                        print('|SUCS|')
                    else:
                        print('|    |')
                    break



if __name__ == '__main__':
    e = NM(10, 450, "config")
    # e.run(83)
    e.evaluation(102)
