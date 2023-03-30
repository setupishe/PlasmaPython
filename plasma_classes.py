import numpy as np
import math
from abc import ABC, abstractmethod

class Particles():
    '''
    Set of macroparticles

    '''
    
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
        

class Nodes():
    '''
    Spatial grid of nodes
    '''

    
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