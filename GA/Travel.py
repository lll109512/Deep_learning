import numpy as np
import matplotlib.pyplot as plt
N_CITIES = 30
CROSS_RATE = 0.1
MUTATION_RATE = 0.02
POP_SIZE = 500
N_GENERATIONS = 500


class TravelSalesPerson(object):
    def __init__(self, n_cities):
        self.city_position = np.random.rand(n_cities, 2)
        plt.ion()

    def plotting(self, lx, ly, total_d):
        plt.cla()
        plt.scatter(self.city_position[:, 0].T, self.city_position[:, 1].T, s=100, c='k')
        plt.plot(lx.T, ly.T, 'r-')
        plt.text(-0.05, -0.05, "Total distance=%.2f" % total_d, fontdict={'size': 20, 'color': 'red'})
        plt.xlim((-0.1, 1.1))
        plt.ylim((-0.1, 1.1))
        plt.pause(0.001)

class GA:
    def __init__(self,DNA_size,cross_rate,mutation_rate,pop_size):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.pop_size = pop_size

        self.pop = np.vstack([np.random.permutation(DNA_size) for _ in range(pop_size)])

    def translateDNA(self, DNA, city_position):     # get cities' coord in order
        line_x = np.empty_like(DNA, dtype=np.float64)
        line_y = np.empty_like(DNA, dtype=np.float64)
        for i, d in enumerate(DNA):
            city_coord = city_position[d]
            line_x[i, :] = city_coord[:, 0]
            line_y[i, :] = city_coord[:, 1]
        return line_x, line_y

    def get_fitness(self, line_x, line_y):
        total_distance = np.empty((line_x.shape[0],), dtype=np.float64)
        for i, (xs, ys) in enumerate(zip(line_x, line_y)):
            total_distance[i] = np.sum(np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys))))
        fitness = np.exp(self.DNA_size * 2 / total_distance)
        return fitness, total_distance

    def select(self, fitness):
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        return self.pop[idx]

    def crossover(self,parent,pop):
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)
            cross_points = np.random.randint(0,2,self.DNA_size,dtype = np.bool)
            keep_city = parent[~cross_points]
            swap_city = np.setdiff1d(pop[i_], keep_city)
            parent[:] = np.concatenate((keep_city,swap_city))
        return parent

    def mutate(self,child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutation_rate:
                swap_point = np.random.randint(self.DNA_size)
                swapA, swapB = child[point], child[swap_point]
                child[point],child[swap_point] = swapB, swapA
        return child

    def evolve(self,fitness):
        pop = self.select(fitness)
        pop_copy = pop.copy()
        for parent in pop:
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop
        pass


if __name__ == '__main__':
    ga = GA(N_CITIES, CROSS_RATE, MUTATION_RATE, POP_SIZE)
    env = TravelSalesPerson(N_CITIES)

    for generation in range(N_GENERATIONS):
        lx, ly = ga.translateDNA(ga.pop, env.city_position)
        fitness, distance = ga.get_fitness(lx,ly)
        best_idx = np.argmax(fitness)
        best_DNA = ga.pop[best_idx]
        print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx],)
        env.plotting(lx[best_idx], ly[best_idx], distance[best_idx])
        ga.evolve(fitness)
    plt.ioff()
    plt.show()
