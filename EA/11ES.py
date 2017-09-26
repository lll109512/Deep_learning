import numpy as np
import matplotlib.pyplot as plt


class ES11:
    def __init__(self, BOUND, DNA_SIZE):
        self.BOUND = BOUND
        self.DNA_SIZE = DNA_SIZE
        self.P = np.random.rand(1, self.DNA_SIZE)
        self.MUTATION_ST = np.random.rand(1, self.DNA_SIZE)
        self.K = np.random.rand(1, self.DNA_SIZE)

    def get_fitness(self, pred):
        return pred.flatten()

    def F(self, x):
        return np.sin(10 * x) * x + np.cos(2 * x) * x

    def make_kid(self):
        self.K = self.P + self.MUTATION_ST * np.random.randn(self.DNA_SIZE)
        self.K = np.clip(self.K,*self.BOUND)

    def kill_bad(self):
        fp = self.get_fitness(self.F(self.P))[0]
        fk = self.get_fitness(self.F(self.K))[0]
        p_target = 1 / 5
        if fp > fk:
            ps = 0
        else:
            self.P = self.K
            ps = 1
        self.MUTATION_ST *= np.exp((1 / np.sqrt(self.DNA_SIZE + 1)) * (ps - p_target) / (1 - p_target))
        return fp

    def Evo(self, n_generation):
        plt.ion()
        # something about plotting
        x = np.linspace(*self.BOUND, 200)
        for _ in range(n_generation):
            self.make_kid()
            self.kill_bad()
            plt.cla()
            plt.scatter(self.P, self.F(self.P), s=200, lw=0, c='red', alpha=0.5)
            plt.scatter(self.K, self.F(self.K), s=200, lw=0, c='blue', alpha=0.5)
            plt.plot(x, self.F(x))
            plt.text(0, -7, 'Mutation strength=%.2f' % self.MUTATION_ST)
            plt.pause(0.05)
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    es = ES11([0, 5], 1)
    es.Evo(200)
