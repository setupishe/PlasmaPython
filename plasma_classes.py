import numpy as np
import math
from abc import ABC, abstractmethod
import pickle

class Particles():
    '''
    Set of macroparticles

    '''
    marker = 'P'
    
    def __init__(self, n_macro: int, concentration: float, q: float, m: float):
        '''
        n_macro: macroparticles number
        concentration: concentration in one macroparticle
        q_over_m: charge over mass ratio
        q: charge of one MICROparticle 
        '''
        self.n_macro = n_macro
        self.concentration = concentration
        self.x = np.zeros((n_macro,), np.double)
        self.v = np.zeros((n_macro,), np.double)
        self.m = m
        self.q = q

        self.normalised = False

    def normalise(self, h: float, tau: float):
        if not self.normalised:
            self.x = self.x/h
            self.v = self.v*tau/h
            self.normalised = True
        else:
            print("Particles are already normalised")

    def denormalise(self, h: float, tau: float):
        if self.normalised:
            self.x = self.x*h
            self.v = self.v*h/tau
            self.normalised = False
        else:
            print("Particles are already denormalized")
    def __add__(self, other):
        if self.concentration != other.concentration:
            raise ValueError("Cannot add two sets of particles with different concentrations")
        if self.q != other.q:
            raise ValueError("Cannot add two sets of particles with different charge values")
        if self.m != other.m:
            raise ValueError("Cannot add two sets of particles with different mass values")

        new_particles = Particles(self.n_macro + other.n_macro, self.concentration, self.q, self.m)
        new_particles.normalised = True
        new_particles.x = np.concatenate((self.x, other.x))
        new_particles.v = np.concatenate((self.v, other.v))
        return new_particles
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            new_particles = Particles(1, self.concentration, self.q, self.m)
            new_particles.normalised = self.normalised
            new_particles.x = self.x[idx:idx+1]
            new_particles.v = self.v[idx:idx+1]
            def update_parent():
                self.x[idx] = new_particles.x
                self.v[idx] = new_particles.v

            new_particles.update_parent = update_parent
            return new_particles
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(self.n_macro)
            new_particles = Particles(stop - start, self.concentration, self.q, self.m)
            new_particles.normalised = self.normalised
            new_particles.x = self.x[start:stop:step]
            new_particles.v = self.v[start:stop:step]
            def update_parent():
                self.x[idx] = new_particles.x
                self.v[idx] = new_particles.v

            new_particles.update_parent = update_parent
            return new_particles
        else:
            raise TypeError("Invalid argument type")
        
    def __setitem__(self, idx, values):
        start, stop, step = idx.indices(len(self))
        if isinstance(values, Particles):
            if len(values) != stop - start:
                raise ValueError("Cannot assign a slice of particles with a different length")
            self.x[idx] = values.x
            self.v[idx] = values.v
        else:
            self.x[idx] = values[0]
            self.v[idx] = values[1]
        self.update_parent(start, stop)

        
        
        

class Nodes():
    '''
    Spatial grid of nodes
    '''
    marker = 'N'
    
    def __init__(self, n: int):
        self.length = n+1

        self.rho = np.zeros((n+1,), np.double)
        self.E = np.zeros((n+1,), np.double)
        self.phi = np.zeros((n+1,), np.double)
        self.electricEnergy = np.zeros((n+1,), np.double)
        


#Todo: add tests
if __name__ == '__main__':
    nodes = Nodes(10)
    print(nodes.system_matrix)


class Distribution(ABC):

    @abstractmethod
    def distr(x: float) -> float:
        """
        calculates distr func(x)
        x: velocity (v in the integral)
        """


class Maxwell(Distribution):
    """ 
    Maxwell distribution
    temp: temperature
    k_b: boltzmann's constant
    m: mass
    """

    def __init__(self, temp: float, k_b: float, m: float):
        self.alpha = -m/(2*k_b*temp)

    def distr(self, x: float):
        return math.exp(self.alpha*x**2)
    
class Wall:
    """
    System's wall, can absorb particles and it's charges
    """

    marker = 'W'
    
    def __init__(self, left: float, right: float, number: int, h: float, 
                 side: str):
        self.left = left/h
        self.right = right/h
        self.h = h
        self.number = number
        self.particles_lst = []
        self.side = side
    