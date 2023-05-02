import numpy as np
from plasma_classes import *
import math
from typing import Callable
import random
import bisect
import pickle

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
    left = (x_jplus1 - x)*value_field[x_j]
    right = (x - x_j)*value_field[x_jplus1]
    res = left + right
    return res

def move(particles: Particles, nodes: Nodes, mode = "default", consistency = False):
    '''
    Moves particles using velocities obtained by leapfrog method in accel()
    All calculations are performed using normalized values
    args:
    particles: set of physical particles
    nodes: spatial grid of nodes
    mode: defines particle's behavoiur when exceeding the system's boundaries:
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


def get_rho(nodes: Nodes, particles, periodic = False):
    """
    Obtains rho value in the nodes using 1-order weighting
    params: 
    nodes: spatial grid of nodes
    particles_tpl: set or tuple of sets of physical macroparticles
    periodic: defines boundary conditions
    """
    for i in range(particles.n_macro):
        x = particles.x[i]
        x_j = math.floor(x)
        if x_j == 200:
            print(particles.x[particles.x > 200])
            print(particles.q)
        x_jplus1 = x_j + 1
        left = particles.concentration*particles.q*(x_jplus1 - x)
        right = particles.concentration*particles.q*(x - x_j)
        nodes.rho[x_j] += left
        nodes.rho[x_jplus1] += right
        if periodic:
            if x_j == 0:
                nodes.rho[nodes.length-1] += left
            if x_jplus1 == nodes.length-1:
                nodes.rho[0] += right


def set_homogeneous(particles: Particles, left: float, right: float):
    """ 
    Sets homogenous spatial distribution of particles
    params: 
    particles: set of physical macroparticles
    L: system's length
    N_p: total number of macroparticles
    """
    N_p = particles.n_macro
    particles.x = np.linspace(left, right, N_p, endpoint=False) + (right-left)/N_p/2

def get_integral(distribution: Distribution, min: float, max: float, n: int, max_p=0.99999999):
    """
    calculates dict{probability: velocity}
    distribution: distribution function class (e.g. maxwell)
    min, max: limits for v, should be >> V_t
    n: number of integration fractions
    max_p: maximum calculated probability
    """
    
    dx = (max-min)/n
    norm = 0
    x = min
    while x < max:
        norm += distribution.distr(x)*dx
        x += dx

    integral_dict = {}
    prob = 0
    x = min

    while prob < max_p:
        integral_dict[prob] = x
        left = distribution.distr(x)
        right = distribution.distr(x+dx)
        prob += (left+right)*dx/2/norm
        x += dx

    return integral_dict

def set_distr(particles: Particles, integral_dict, h, tau, n_range = None):
    """
    sets macroparticles' velocities accortind to distribution function
    args:
    particles : sets of macroparticles
    distribution: distribution function class (e.g. maxwell)
    min, max: limits for v, should be >> V_t
    n: number of integration fractions
    neutral_range: determine if particles should be modified within the range
    """
    
    mask = np.ones(particles.n_macro)

    if n_range is not None:
        mask = (particles.x > n_range[0]) & (particles.x < n_range[1])

    particles.denormalise(h, tau)
    
    probs_keys = list(integral_dict.keys())
    for i in range(particles.n_macro):
        if mask[i]:
            r = random.random()
            ind = bisect.bisect_left(probs_keys, r)
            if ind == len(probs_keys):
                ind = -1
            key = probs_keys[ind]
            particles.v[i] = integral_dict[key]

    particles.normalise(h, tau)



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

def calc_kinetic_energy(particles: Particles, h: float, tau: float):
    """
    calculates kinetic energy of all particles
    args:
    particles : sets of macroparticles
    h: h: cell size
    tau: time step
    """
    if particles.normalised:
        particles.denormalise(h, tau)
    res = np.sum(particles.v**2)*particles.m*particles.concentration/2
    particles.normalise(h, tau)
    return res

def calc_electric_energy(particles: Particles, nodes: Nodes):
    """
    calculates electric energy of all particles
    args:
    particles : sets of macroparticles
    nodes: spatial grid of nodes
    """
    res = 0
    rho = particles.q*particles.concentration
    for i in range(particles.n_macro):
        x = particles.x[i]
        phi = weight_field_value(x, nodes.phi)
        res += rho*phi
    return res


def account_walls(particles_lst: Particles, walls: list[Wall], SEE=None, Energy=None, injection=None):
    for particles in particles_lst:
        params = (particles.concentration, particles.q, particles.m)
        for wall in walls:
            # Identifying the absorbed particles
            absorbed_mask = (particles.x <= wall.right) & (particles.x >= wall.left)
            if np.sum(absorbed_mask) == 0:
                continue
            freeze_coordinate = wall.right - wall.h/10 if wall.side == "left" else wall.left + wall.h/10
            if Energy is not None:
                electric = 0
                kinetic = 0
                summ = 0
            SEE_success = False

            if SEE is not None and particles.q < 0:
                # Step 1: Discern particles capable of generating secondary electrons
                particles.denormalise(SEE["h"], SEE["tau"])
                energy = 0.5 * particles.m * particles.v ** 2
                particles.normalise(SEE["h"], SEE["tau"])
                emit_mask = energy > SEE["E1"]
                absorbed_emit_mask = absorbed_mask & emit_mask
                if np.sum(absorbed_emit_mask) > 0:
                    SEE_success = True
                    # Step 2: Calculate the secondary electron emission yield (σ)
                    sigma = ((energy[absorbed_emit_mask]) / SEE["E1"]) ** SEE["alpha"]
                    secondary_counts = np.floor(sigma).astype(int)
                    # Step 3: Adding generated electrons to the system and ions to the wall
                    probabilities = np.random.rand(len(sigma))
                    secondary_counts += (probabilities < (np.floor(sigma)-sigma+1)).astype(int)
                    total_secondary = np.sum(secondary_counts)

                    new_electrons = Particles(total_secondary, *params)
                    new_coordinate = wall.right + 1 if wall.side == "left" else wall.left - 1
                    new_electrons.x = np.full(new_electrons.n_macro, new_coordinate)
                    new_electrons.normalised = True
                    set_distr(new_electrons, SEE["see_integral"], SEE["h"], SEE["tau"])
                    
                    #print("emittion")
                    # print(new_electrons.x)
                    # print(new_electrons.v)

                    positrons = Particles(total_secondary, *params)
                    positrons.q *= -1
        
                    positrons.x = np.full(positrons.n_macro, freeze_coordinate)
                    positrons.v = np.zeros(positrons.n_macro)

                    wall.particles_lst.append(positrons)

                    if Energy is not None:
                        electric -= calc_electric_energy(new_electrons, Energy["nodes"])
                        kinetic -= calc_kinetic_energy(new_electrons, Energy["h"], Energy["tau"])
                        summ -= electric + kinetic

            absorbed_particles = particles[absorbed_mask].deepcopy()
            if Energy is not None:
                        electric += calc_electric_energy(absorbed_particles, Energy["nodes"])
                        kinetic += calc_kinetic_energy(absorbed_particles, Energy["h"], Energy["tau"])
                        summ += electric + kinetic
                        Energy["electric"].append(electric)
                        Energy["kinetic"].append(kinetic)
                        Energy["summ"].append(summ)

            if particles.q < 0:
                # Excluding absorbed particles from the original set
                particles.delete(absorbed_mask)
            else:
                #setting random coordinate inside source region
                center = (injection["nodes"].length - 1)/2
                base = (injection["n_range"][1] - injection["n_range"][0])/2
                shift = base*(2*np.random.rand(np.sum(absorbed_mask)) - 1)
                #ions
                particles.x[absorbed_mask] = shift + center
                set_distr(particles, injection["i_integral"], injection["h"], injection["tau"], injection["n_range"])

                #electrons
                paired_electrons = particles[absorbed_mask].deepcopy()
                paired_electrons.q *= -1
                paired_electrons.m = injection["electrons"].m
                set_distr(paired_electrons, injection["e_integral"], injection["h"], injection["tau"], injection["n_range"])

                particles_lst[0] += paired_electrons
                

            
            absorbed_particles.x = np.full(absorbed_particles.n_macro, freeze_coordinate)
            absorbed_particles.v = np.zeros(absorbed_particles.n_macro)
            wall.particles_lst.append(absorbed_particles)

            if SEE_success:
                particles += new_electrons

def central_difference(arr, dt):
    first_deriv = np.zeros_like(arr)
    first_deriv[0] = (arr[1] - arr[0]) / dt
    first_deriv[1:-1] = (arr[2:] - arr[:-2]) / (2 * dt)
    first_deriv[-1] = (arr[-1] - arr[-2]) / dt

    return first_deriv

def history2flux(history: np.array, tau):
    res = []
    for i in range(history.shape[0]):
        diff = np.mean(central_difference(history[i], tau))
        res.append(diff)
    return res

def save_to_file(obj, filename):
    with open(filename, 'wb') as f:
        marker = obj.marker
        dict_data = obj.__dict__
        pickle.dump((marker, dict_data), f)

def load_from_file(filename):
    with open(filename, 'rb') as f:
        marker, dict_data = pickle.load(f)
        
    classes = {
        'P': Particles,
        'N': Nodes,
        'W': Wall
    }
    
    if marker not in classes:
        raise ValueError(f"File {filename} has unknown marker string: {marker}")
    
    cls = classes[marker]
    obj = cls.__new__(cls)
    obj.__dict__.update(dict_data)
    return obj

