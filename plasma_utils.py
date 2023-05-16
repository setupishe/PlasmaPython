import numpy as np
from plasma_classes import *
import math
import random
import bisect
import pickle
import os

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


def calc_fields(nodes: Nodes, h: float, eps: float, periodic = False, walls=None) -> None:
    """
    Calculates potential and field based on charge density using Thomas algorithm
    args:
    nodes: spatial grid of nodes
    h: cell size
    eps: epsilon, permittivity
    periodic: defines boundary conditions
    """
    if walls is not None:
        left = right = 0
        for wall in walls:
            if wall.side == "left":
                left = wall.right
                left_charge = wall.charge
            else:
                right = wall.left
                right_charge = wall.charge
        
        M = nodes.length - left - (nodes.length -1 - right)
        system_matrix = np.zeros((M, 3))
        system_matrix[0] = system_matrix[M-1] = [0, 1, 0]
        system_matrix[1:M-1] = [1, -2, 1]

        rho = nodes.rho[left:right+1]
        rho[0] = left_charge
        rho[-1] = right_charge
        rho *= -h**2/eps
        res_phi = thomas_algorithm(system_matrix, rho)
        nodes.phi *= 0
        nodes.phi[left:right+1] = res_phi
        

    elif periodic:
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

def weight_field_value(x: np.array, value_field: np.array):
    """
    Gets field value for particle's position using first-order weighting
    x: particle's position
    value_field: nodes array of field values: rho, phi etc.
    """
    x_j = np.floor(x).astype(int)
    x_jplus1 = x_j + 1
    left = (x_jplus1 - x)*value_field[x_j]
    right = (x - x_j)*value_field[x_jplus1]
    res = left + right
    return res

def accel(particles: Particles, nodes: Nodes, A: float, zerostep=False)-> None:
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
    E = weight_field_value(particles.x, nodes.E)
    E *= A
    if zerostep:
        particles.v -= E
    else:
        particles.v += E



def move(particles: Particles, nodes: Nodes, mode="default", consistency=False):
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
    
    # update particle positions
    particles.x += particles.v
    
    # handle periodic/mirror boundary conditions
    if mode == "periodic":
        particles.x = np.where(particles.x > l, np.abs(particles.x) % l, particles.x)
        particles.x = np.where(particles.x < 0, l - (-particles.x % l), particles.x)
    elif mode == "mirror":
        particles.x = np.where(particles.x > l, l - np.abs(particles.x) % l, particles.x)
        particles.x = np.where(particles.x < 0, np.abs(particles.x % l), particles.x)
    
    # check for Courant condition violation
    if consistency and np.any(particles.v > 1):
        idx = np.argmax(particles.v > 1)
        raise Exception(f'Too fast! Particle number {idx} has flown for more than one cell.')


def get_rho(nodes: Nodes, particles, periodic=False):
    """
    Obtains rho value in the nodes using 1-order weighting
    params:
    nodes: spatial grid of nodes
    particles_tpl: set or tuple of sets of physical macroparticles
    periodic: defines boundary conditions
    """
    conc = np.zeros(nodes.length, dtype=np.double)

    x_j = np.floor(particles.x).astype(int)
    x_jplus1 = x_j + 1

    left = particles.concentration * (x_jplus1 - particles.x)
    right = particles.concentration * (particles.x - x_j)
    np.add.at(conc, x_j, left)
    np.add.at(conc, x_jplus1, right)
    #TODO: fix 
    if periodic:
        if np.any(x_j == 0):
            conc[0] += np.sum(left[x_j == 0])
        if np.any(x_jplus1 == nodes.length - 1):
            conc[nodes.length - 1] += np.sum(right[x_jplus1 == nodes.length - 1])
    if particles.q > 0:
        nodes.conc_i += conc
    else:
        nodes.conc_e += conc
    rho = conc * particles.q
    np.copyto(nodes.rho, nodes.rho + rho, where=rho != 0)


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

def range_mask(particles, n_range):
    """
    generates boolean mask for particles in range
    args:
    particles : sets of macroparticles
    n_range = neutral range
    nodes: spatial grid of nodes
    """

    mask = (particles.x >= n_range[0]) & (particles.x <= n_range[1])
    return mask

def range_coordinates(n_range, mask):
    """
    generates random coordinates range
    args:
    particles : sets of macroparticles
    n_range = neutral range
    """
    center = (n_range[1] + n_range[0])/2
    base = (n_range[1] - n_range[0])/2
    shift = base*(2*np.random.rand(int(np.sum(mask))) - 1)
    coordinates = center + shift
    return coordinates

def set_distr(particles: Particles, integral_dict, h, tau, n_range = None):
    """
    sets macroparticles' velocities accortind to distribution function
    args:
    particles: sets of macroparticles
    inregral_dict: precalculated set of velocities
    h: spatial grid step
    tau: time step
    n_range: determine if particles should be modified within the range
    """
    
    mask = np.ones(particles.n_macro)

    if n_range is not None:
        mask = range_mask(particles, n_range)

    particles.denormalise(h, tau)
    
    probs_keys = list(integral_dict.keys())
    for i in range(particles.n_macro):
        if mask[i]:
            r = random.random()
            ind = bisect.bisect_left(probs_keys, r)
            if ind == len(probs_keys):
                ind = -1
            key = probs_keys[ind]
            sign = 1 if particles.v[i] >= 0 else -1
            #particles.v[i] = abs(integral_dict[key])*sign
            particles.v[i] = integral_dict[key]

    particles.normalise(h, tau)

def pump_particles(particles_lst, constant_n, n_range, windows=1):
    """
    particles: sets of macroparticles
    inregral_dict: precalculated set of velocities
    h: spatial grid step
    tau: time step
    n_range: determine if particles should be modified within the range
    """
    len_range = n_range[1] - n_range[0]
    
    if len_range % windows != 0:
        raise ValueError("The length of the neutral area must be divisible by \
                         the number of windows without a remainder")
    window = len_range/windows
    left = right = 0
    for i in range(windows):
        left = n_range[0] + window*i
        right = left + window
        window_range = (left, right)
        N = 0
        for particles in particles_lst:
            mask = range_mask(particles, window_range)
            slc = particles[mask]
            N += slc.n_macro
        delta = constant_n - N
        n = delta//2
        if n > 0:
            print("FLUX!!!", n)
            for particles in particles_lst:
                mask = range_mask(particles, window_range)
                slc = particles[mask]

                
                new_particles = Particles(n, particles.concentration,
                                        particles.q, particles.m)
                new_particles.normalised = particles.normalised
                new_particles.x = range_coordinates(window_range, mask)[:new_particles.n_macro]
                new_particles.v = np.random.choice(slc.v, new_particles.n_macro)
                particles.add(new_particles)



def get_distr(particles: Particles, n_range):
    """ 
    obtain velocity distribution in chosen range
    args:
    particles: set of macroparticles
    xmin, xmax: spatial range for probing
    """
    mask = range_mask(particles, n_range)
    return particles[mask].v.copy()

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

                    # positrons = Particles(total_secondary, *params)
                    # positrons.q *= -1
        
                    # positrons.x = range_coordinates((wall.left, wall.right), np.ones(total_secondary))
                    # positrons.v = np.zeros(positrons.n_macro)

                    # wall.particles_lst.append(positrons)
                    wall.charge += abs(particles.q*total_secondary)*particles.concentration

                    if Energy is not None:
                        electric -= calc_electric_energy(new_electrons, Energy["nodes"])
                        kinetic -= calc_kinetic_energy(new_electrons, Energy["h"], Energy["tau"])
                        summ -= electric + kinetic

            absorbed_particles = particles[absorbed_mask].deepcopy()
            if Energy is not None:
                        kinetic += calc_kinetic_energy(absorbed_particles, Energy["h"], Energy["tau"])
                        summ += electric + kinetic
                        Energy["electric"].append(electric)

            if injection is not None and particles.q > 0:
            
                particles.x[absorbed_mask] = range_coordinates(injection["n_range"], absorbed_mask)
                set_distr(particles, injection["i_integral"], injection["h"], injection["tau"], injection["n_range"])

                #electrons
                paired_electrons = particles[absorbed_mask].deepcopy()
                paired_electrons.q *= -1
                paired_electrons.m = injection["electrons"].m
                set_distr(paired_electrons, injection["e_integral"], injection["h"], injection["tau"], injection["n_range"])

                particles_lst[0].add(paired_electrons)

            else:
                # Excluding absorbed particles from the original set
                particles.delete(absorbed_mask)

                
            # absorbed_particles.x = range_coordinates((wall.left, wall.right), absorbed_mask)
            # absorbed_particles.v = np.zeros(absorbed_particles.n_macro)
            # wall.particles_lst.append(absorbed_particles)
            wall.charge += particles.q*absorbed_particles.n_macro*particles.concentration

            if SEE_success:
                particles.add(new_electrons)


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

def save_system_state(iteration: int, nodes: Nodes, particles: Particles, walls: list[Wall], file_path: str):
    """
    Serialize the system's state and append it to a binary file.
    
    iteration: int - the current iteration of the simulation.
    nodes: Nodes - the spatial grid of nodes in the system.
    particles: Particles - the set of particles in the system.
    walls: List[Wall] - the walls in the system.
    file_path: str - the path to the binary file to write to.
    """
    # Serialize the system's state.
    serialized_data = {
        "iteration": iteration,
        "nodes": nodes,
        "particles": particles,
        "walls": walls
    }
    serialized_bytes = pickle.dumps(serialized_data)

    # Open the file in append and read mode.
    with open(file_path, "ab+") as f:
        # Move the file pointer to the end of the file.
        f.seek(0, os.SEEK_END)

        # Write a header containing the size of the serialized data.
        size_bytes = len(serialized_bytes).to_bytes(4, byteorder="big")
        f.write(size_bytes)

        # Write the serialized data.
        f.write(serialized_bytes)

def load_system_state(file_path: str, iteration: int):
    """
    Load the system's state from a binary file at the specified iteration.

    file_path: str - the path to the binary file to read from.
    iteration: int - the iteration to load from the file.
    Returns:
        - The nodes object at the specified iteration.
        - The particles object at the specified iteration.
        - The walls object at the specified iteration.
    """
    # Open the file in read-only mode.
    with open(file_path, "rb") as f:
        # Compute the position in the file where the serialized data for the specified iteration begins.
        header_size = 4  # The size of the header containing the size of the serialized data.
        data_offset = 0
        current_iteration = -1
        print("----")
        while current_iteration < iteration:
            # Move the file pointer to the beginning of the serialized data.
            f.seek(data_offset, os.SEEK_SET)

            # Read the size of the serialized data.
            size_bytes = f.read(header_size)
            if not size_bytes:
                raise ValueError(f"No data for iteration {iteration} found in file {file_path}")
            size = int.from_bytes(size_bytes, byteorder="big")

            # Compute the position in the file where the next serialized data begins.
            data_offset = f.tell() + size
            if data_offset == header_size:
                raise ValueError(f"No data for iteration {iteration} found in file {file_path}")

            # Update the current iteration.
            current_iteration += 1

        # Move the file pointer back to the beginning of the serialized data for the specified iteration.
        f.seek(data_offset - size - header_size, os.SEEK_SET)

        # Read the size of the serialized data.
        size_bytes = f.read(header_size)
        size = int.from_bytes(size_bytes, byteorder="big")

        # Read the serialized data.
        serialized_bytes = f.read(size)
        serialized_data = pickle.loads(serialized_bytes)

    # Return the nodes, particles, and walls from the deserialized data.
    return serialized_data["nodes"], serialized_data["particles"], serialized_data["walls"]


def loop_over_states(file_path: str):
    """
    A generator that yields the system's state at each iteration from a binary file.

    file_path: str - the path to the binary file to read from.
    Yields:
        - A tuple containing the nodes object, the particles object, and the walls object.
    """
    # Open the file in read-only mode.
    with open(file_path, "rb") as f:
        # Compute the position in the file where the serialized data begins.
        header_size = 4  # The size of the header containing the size of the serialized data.
        data_offset = 0

        while True:
            # Move the file pointer to the beginning of the serialized data.
            f.seek(data_offset, os.SEEK_SET)

            # Read the size of the serialized data.
            size_bytes = f.read(header_size)
            if not size_bytes:
                # No more data in the file.
                break
            size = int.from_bytes(size_bytes, byteorder="big")

            # Compute the position in the file where the next serialized data begins.
            data_offset = f.tell() + size

            # Read the serialized data.
            serialized_bytes = f.read(size)
            serialized_data = pickle.loads(serialized_bytes)

            # Yield the iteration number, nodes, particles, and walls from the deserialized data.
            yield serialized_data["nodes"], serialized_data["particles"], serialized_data["walls"]
