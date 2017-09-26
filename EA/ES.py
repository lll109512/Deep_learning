import numpy as np
import matplotlib.pyplot as plt
class ES:
    def __init__(self,DNA_SIZE,POP_SIZE,BOUND):
        self.BOUND = BOUND
        self.POP_SIZE = POP_SIZE
        self.DNA_SIZE = DNA_SIZE
        self.pop = dict(DNA=5 * np.random.rand(1, DNA_SIZE).repeat(POP_SIZE, axis=0),
           MUTATION_ST=np.random.rand(POP_SIZE, DNA_SIZE))

    def make_kid(self,N_kid):
        kids = {'DNA': np.empty((N_kid, self.DNA_SIZE))}
        kids['MUTATION_ST'] = np.empty_like(kids['DNA'])
        for kd ,ks in zip(kids['DNA'],kids['MUTATION_ST']):
            p1,p2 = np.random.choice(self.POP_SIZE,size = 2,replace = False)
            cp = np.random.randint(0,2,self.DNA_SIZE,dtype = np.bool)
            kd[cp] = self.pop['DNA'][p1,cp]
            kd[~cp] = self.pop['DNA'][p2,~cp]
            ks[cp] = self.pop['MUTATION_ST'][p1,cp]
            ks[~cp] = self.pop['MUTATION_ST'][p2,~cp]

            ks[:] = np.maximum(ks + (np.random.rand(*ks.shape) - 0.5),0.)
            kd += ks* np.random.randn(*kd.shape)
            kd[:] = np.clip(kd, *self.BOUND)
        return kids

    def kill_bad(self,kids):
        for key in self.pop.keys():
            self.pop[key] = np.vstack((self.pop[key],kids[key]))
        fitness = self.get_fitness(self.F(self.pop['DNA']))
        idx = np.arange(self.pop['DNA'].shape[0])
        good_idx = idx[fitness.argsort()][-self.POP_SIZE:]
        for key in self.pop.keys():
            self.pop[key] = self.pop[key][good_idx]
        return self.pop

    def get_fitness(self,pred):
        return pred.flatten()

    def F(self,x): return np.sin(10*x)*x + np.cos(2*x)*x

    def Evo(self,N_generation):
        plt.ion()
        # something about plotting
        x = np.linspace(*self.BOUND, 200)
        plt.plot(x, self.F(x))
        for _ in range(N_generation):
            if 'sca' in globals():
                sca.remove()
            sca = plt.scatter(self.pop['DNA'], self.F(self.pop['DNA']), s=200, lw=0, c='red', alpha=0.5)
            plt.pause(0.05)
            kids = self.make_kid(self.POP_SIZE)
            self.kill_bad(kids)
        plt.ioff()
        plt.show()





if __name__ == '__main__':
    es = ES(1, 200, [0,5])
    es.Evo(200)
