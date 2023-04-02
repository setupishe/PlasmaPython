import numpy as np
from plasma_classes import *
import math
from typing import Callable
import random

def is_diagonally_dominant(x: np.array) -> bool:
    """
    Checks if input matrix is diagonally dominant.
    args:
    x: 2d np.array
    """
    b = np.abs(np.diag(x))
    di = np.diag_indices(x.shape[0])
    stripped = np.abs(x.copy())
    stripped[di] = 0

    summ = np.sum(stripped, axis=1)
    first_condition = np.greater_equal(b, summ)
    second_condition = np.greater(b, summ)
    
    return (np.all(first_condition) and np.any(second_condition))


def thomas_algorithm(A: np.array, d: np.array) -> np.array:
    '''
    Returns roots of system of linear equations with tridiagonal matrix.
    args:
    A: tridiagonal or 3-column matrix
    d: constant terms
    '''
    if A.shape[0] == A.shape[1]:
        if not is_diagonally_dominant(A):
            print("WARNING: system matrix is not diagonally dominant")

        b = np.diag(A)
        c = np.diag(A, k=1)
        a = np.diag(A, k=-1)
        a = np.insert(a, 0, 0)
        c = np.insert(c, c.shape[0], 0)

    else:
        a, b, c = A[:, 0], A[:, 1], A[:, 2] 
        
    n = A.shape[0]

    p = np.zeros(n-1)
    q = np.zeros(n-1)
    x = np.zeros(n)

    #forward pass:
    p[0] = -c[0]/b[0]
    q[0] = d[0]/b[0]

    for i in range(1, n-1):
        p[i] = -c[i]/(a[i]*p[i-1]+b[i])
        q[i] = (d[i]-a[i]*q[i-1])/(a[i]*p[i-1]+b[i])
    
    #backward pass:
    x[n-1] = (d[n-1]-a[n-1]*q[n-2])/(a[n-1]*p[n-2]+b[n-1])
    for i in range(n-2, -1, -1):
        x[i] = x[i+1]*p[i]+q[i]
    return x


def calc_fields(nodes: Nodes, h: float, eps: float, periodic = False) -> None:
    """
    Calculates potential and field based on charge density using Thomas algorithm
    args:
    nodes: spatial grid of nodes
    h: cell size
    eps: epsilon, permittivity
    periodic: defines boundary conditions
    """
    if periodic:
        M = nodes.length
        system_matrix = np.zeros((M-2, 3))

        #граничные условия
        system_matrix[0] = [0, -2, 1]
        system_matrix[M-3] = [1, -2, 0]
        system_matrix[1:M-3] = [1, -2, 1]

        nodes.phi[0] = nodes.phi[-1] = 0
        nodes.phi[1:-1] = thomas_algorithm(system_matrix, -nodes.rho[1:-1]*h**2/eps)
    else:
        M = nodes.length
        system_matrix = np.zeros((M, 3))

        #граничные условия
        system_matrix[0] = system_matrix[M-1] = [0, 1, 0]
        system_matrix[1:M-1] = [1, -2, 1]
        nodes.phi = thomas_algorithm(system_matrix, -nodes.rho*h**2/eps)
    
    nodes.E[0] = -(nodes.phi[1]-nodes.phi[0])/h
    nodes.E[-1] = -(nodes.phi[-1]-nodes.phi[-2])/h
    for i in range(1, nodes.length-1):
        nodes.E[i] = -(nodes.phi[i+1]-nodes.phi[i-1])/(2*h)

def accel(particles: Particles, nodes: Nodes, L: float, h: float, tau: float, 
zerostep=False)-> None:
    """ 
    Calculates velocities using weighting electric field and leapfrog method
    All calculations are performed using normalized values
    args:
    particles: set of macroparticles
    nodes: spatial grid of nodes
    L: system's length
    h: h: cell size
    tau: time step
    zerostep: if True, calculates velocity on -tau/2 step
    """
    for i in range(particles.n_macro):
        E = weight_field_value(particles.x[i], nodes.E)
        E = (particles.q*E*(tau**2))/(particles.m*h)
        if zerostep:
            particles.v[i] = particles.v[i] - E
        else:
            particles.v[i] = particles.v[i] + E

def weight_field_value(x: float, value_field: np.array):
    """
    Gets field value for particle's position using first-order weighting
    x: particle's position
    value_field: nodes array of field values: rho, phi etc.
    """
    x_j = math.floor(x)
    x_jplus1 = x_j + 1
    res = (x_jplus1 - x)*value_field[x_j] + (x - x_j)*value_field[x_jplus1]
    return res

def move(particles: Particles, nodes: Nodes, mode = "default", consistency = False):
    '''
    Moves particles using velocities obtained by leapfrog method in accel()
    All calculations are performed using normalized values
    args:
    particles: set of physical particles
    nodes: spatial grid of nodes
    h: h: cell size
    periodic: defines particle's behavoiur when exceeding the system's boundaries:
    if True, particle appears on the on the other side, else disappears (!!Todo!!)
    consistency: if True, raises exception if Courant condition is violated
    '''
    l = nodes.length - 1
    for i in range(particles.n_macro):
        if consistency and (particles.v[i] > 1):
            raise Exception(f'Too fast! Particle number {i} has flown for more than one cell.')
        particles.x[i] = particles.x[i] + particles.v[i]
        if mode == "periodic":
            if particles.x[i] > l:
                particles.x[i] = abs(particles.x[i]) % l
            elif particles.x[i] < 0:
                particles.x[i] = l-(-particles.x[i]%l)
        elif mode == "mirror":
            if particles.x[i] > l:
                particles.x[i] = l - abs(particles.x[i]) % l
            elif particles.x[i] < 0:
                particles.x[i] = abs(particles.x[i]%l)
        #todo: написать уничтожение частиц


def getrho(nodes: Nodes, *particles_tpl, periodic = False):
    """
    Obtains rho value in the nodes using 1-order wieghting
    params: 
    nodes: spatial grid of nodes
    particles_tpl: set or tuple of sets of physical macroparticles
    periodic: defines boundary conditions
    """
    nodes.rho *= 0 
    for particles in particles_tpl:
        for i in range(particles.n_macro):
            x = particles.x[i]
            x_j = math.floor(x)
            x_jplus1 = x_j + 1
            left = particles.concentration*particles.q*(x_jplus1 - x)
            right = particles.concentration*particles.q*(x - x_j)
            #print(left, right)
            nodes.rho[x_j] += left
            nodes.rho[x_jplus1] += right
            if periodic:
                if x_j == 0:
                    nodes.rho[nodes.length-1] += left
                if x_jplus1 == nodes.length-1:
                    nodes.rho[0] += right


def set_homogeneous(particles: Particles, L: float):
    """ 
    Sets homogenous spatial distribution of particles
    params: 
    particles: set of physical macroparticles
    L: system's length
    N_p: total number of macroparticles
    """
    N_p = particles.n_macro
    particles.x = np.linspace(0, L, N_p, endpoint=False) + L/N_p/2

def set_distr(particles: Particles, distribution: Distribution, min: float, max: float, n: int):
    """
    sets macroparticles' velocities accortind to distribution function
    args:
    particles : sets of macroparticles
    distribution: distribution function class (e.g. maxwell)
    min, max: limits for v, should be >> V_t
    n: number of integration fractions
    """
    dx = (max-min)/n
    norm = 0
    x = min
    while x < max:
        norm += distribution.distr(x)*dx
        x += dx

    for i in range(particles.n_macro):
        res = 0
        x = min
        r = random.random()
        while res < r:
            left = distribution.distr(x)
            right = distribution.distr(x+dx)
            res += (left+right)*dx/2/norm
            x += dx
        x -= dx
        #sign = random.choice((-1, 1))
        particles.v[i] = x#*sign
        #print(i)



def get_distr(particles: Particles, xmin: float, xmax: float):
    """ 
    obtain velocity distribution in chosen range
    args:
    particles: set of macroparticles
    xmin, xmax: spatial range for probing
    """
    res = []
    for i in range(particles.n_macro):
        if particles.x[i] >= xmin and particles.x[i] <= xmax:
            res.append(particles.v[i])
    return res