import numpy as np

TARGETS = 'Hellow world!'
POP_SIZE = 300
CROSS_RATE = 0.4
MUTATION_RATE = 0.001
N_GENERATIONS = 3000

DNA_SIZE = len(TARGETS)
TARGET_ASCII = np.fromstring(TARGETS, dtype=np.uint8)
ASCII_BOUND = [32, 126]

class GA:
    def __init__(self,DNA_size,DNA_bound,cross_rate,mutation_rate,pop_size):
        self.DNA_size = DNA_size
        DNA_bound[1] += 1   #这里是为了在range(*DNA_bound)的时候避免漏掉最后一位126号字符
        self.DNA_bound = DNA_bound
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.pop_size = pop_size

        self.pop = np.random.randint(*self.DNA_bound,size=(self.pop_size,self.DNA_size),dtype = (np.uint8))

    def translateDNA(self,DNA):
        return DNA.tostring().decode('ascii')

    def get_fitness(self):
        match_count = (self.pop == TARGET_ASCII).sum(axis=1)
        return match_count

    def select(self):
        fitness = self.get_fitness() + 1e-4 #避免 0 fitness
        idx = np.random.choice(np.arange(self.pop_size),size = self.pop_size,replace=True, p=fitness/fitness.sum())
        return self.pop[idx]

    def crossover(self,parent,pop):
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)
            cross_points = np.random.randint(0,2,self.DNA_size,dtype = np.bool)
            parent[cross_points] = pop[i_,cross_points]
        return parent

    def mutate(self,child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutation_rate:
                child[point] = np.random.randint(*self.DNA_bound)
        return child

    def evolve(self):
        pop = self.select()
        pop_copy = pop.copy()
        for parent in pop:
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop
        pass

if __name__ == '__main__':
    ga = GA(DNA_SIZE, ASCII_BOUND, CROSS_RATE, MUTATION_RATE, POP_SIZE)

    for generation in range(N_GENERATIONS):
        fitness = ga.get_fitness()
        best_DNA = ga.pop[np.argmax(fitness)]
        best_phrase = ga.translateDNA(best_DNA)
        print('Gen', generation, ': ', best_phrase)
        if best_phrase == TARGETS:
            break
        ga.evolve()
