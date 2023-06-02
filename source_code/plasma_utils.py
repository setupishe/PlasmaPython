import numpy as np
from plasma_classes import *
import math
import random
import bisect
import pickle
import os
from tqdm import tqdm
import shutil
from datetime import datetime


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
    
    # if len_range % windows != 0:
    #     raise ValueError("The length of the neutral area must be divisible by \
    #                      the number of windows without a remainder")
    window = len_range/windows
    left = right = 0
    for i in range(windows):
        left = n_range[0] + window*i
        right = left + window# if left + window < n_range[1] else n_range[1]
        window_range = (left, right)
        N = 0
        for particles in particles_lst:
            mask = range_mask(particles, window_range)
            if np.sum(mask) == 0:
                continue
            slc = particles[mask]
            N += slc.n_macro
        delta = constant_n - N
        n = delta//2
        if n > 0:
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
    calculates sum kinetic energy of all particles
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

def calc_impulse(particles: Particles, h: float, tau: float):
    """
    calculates sum impulse of particles
    args:
    particles : sets of macroparticles
    h: h: cell size
    tau: time step
    """
    if particles.normalised:
        particles.denormalise(h, tau)
    res = np.sum(np.abs(particles.v))*particles.m*particles.concentration
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


def account_walls(particles_lst: Particles, walls: list[Wall], SEE=None, injection=None):
    for particles in particles_lst:
        params = (particles.concentration, particles.q, particles.m)
        for wall in walls:
            # Identifying the absorbed particles
            absorbed_mask = (particles.x <= wall.right) & (particles.x >= wall.left)
            if np.sum(absorbed_mask) == 0:
                continue
            SEE_success = False

            if SEE is not None and particles.q < 0:
                # Step 1: Discern particles capable of generating secondary electrons
                particles.denormalise(SEE["h"], SEE["tau"])
                energy = 0.5 * particles.m * particles.v ** 2
                particles.normalise(SEE["h"], SEE["tau"])

                # Step 2: Calculate the secondary electron emission yield (σ)
                if SEE["alpha"] is not None:
                    sigma = ((energy[absorbed_mask]) / SEE["E1"]) ** SEE["alpha"]
                else:
                    sigma = 0.5 + (energy[absorbed_mask]) / SEE["E1"]/2

                secondary_counts = np.floor(sigma).astype(int)
                # Step 3: Adding generated electrons to the system and ions to the wall
                probabilities = np.random.rand(len(sigma))
                secondary_counts += (probabilities < (sigma - np.floor(sigma))).astype(int)
                total_secondary = np.sum(secondary_counts)
                if total_secondary > 0:
                    SEE_success = True
                    new_electrons = Particles(total_secondary, *params)
                    new_coordinate = wall.right + 1 if wall.side == "left" else wall.left - 1
                    new_electrons.x = np.full(new_electrons.n_macro, new_coordinate)
                    new_electrons.normalised = True
                    set_distr(new_electrons, SEE["see_integral"], SEE["h"], SEE["tau"])

                    wall.charge += abs(particles.q*total_secondary)*particles.concentration
                    SEE["secondary_electrons"] = new_electrons.deepcopy()
                
                #print("emittion")
                # print(new_electrons.x)
                # print(new_electrons.v)

                # positrons = Particles(total_secondary, *params)
                # positrons.q *= -1
    
                # positrons.x = range_coordinates((wall.left, wall.right), np.ones(total_secondary))
                # positrons.v = np.zeros(positrons.n_macro)

                # wall.particles_lst.append(positrons)

            absorbed_particles = particles[absorbed_mask].deepcopy()

            sort = "ions" if particles.q > 0 else "electrons"
            name = "absorbed_" + sort
            SEE[name] = absorbed_particles

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

def save_system_state(iteration: int, nodes: Nodes, particles: Particles, walls: list[Wall], file_path: str, modes=('particles', 'nodes', 'walls')):
    """
    Serialize the system's state and append it to a binary file.

    iteration: int - the current iteration of the simulation.
    nodes: Nodes - the spatial grid of nodes in the system.
    particles: Particles - the set of particles in the system.
    walls: List[Wall] - the walls in the system.
    file_path: str - the path to the binary file to write to.
    modes: tuple - indicates which objects to save (default: ('particles', 'nodes', 'walls')).
    """
    serialized_data = {"iteration": iteration}

    if 'particles' in modes:
        serialized_data['particles'] = particles
    if 'nodes' in modes:
        serialized_data['nodes'] = nodes
    if 'walls' in modes:
        serialized_data['walls'] = walls

    serialized_bytes = pickle.dumps(serialized_data)

    with open(file_path, "ab+") as f:
        f.seek(0, os.SEEK_END)
        size_bytes = len(serialized_bytes).to_bytes(4, byteorder="big")
        f.write(size_bytes)
        f.write(serialized_bytes)


def load_system_state(file_path: str, iteration: int, modes=('particles', 'nodes', 'walls')):
    """
    Load the system's state from a binary file at the specified iteration.

    file_path: str - the path to the binary file to read from.
    iteration: int - the iteration to load from the file.
    modes: tuple - indicates which objects to load (default: ('particles', 'nodes', 'walls')).
    Returns:
        - The nodes object at the specified iteration (if 'nodes' in modes).
        - The particles object at the specified iteration (if 'particles' in modes).
        - The walls object at the specified iteration (if 'walls' in modes).
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

    result = [serialized_data["iteration"]]
    if 'nodes' in modes:
        result.append(serialized_data.get('nodes'))
    if 'particles' in modes:
        result.append(serialized_data.get('particles'))
    if 'walls' in modes:
        result.append(serialized_data.get('walls'))

    return tuple(result)


def loop_over_states(file_path: str, modes=('particles', 'nodes', 'walls')):
    """
    A generator that yields the system's state at each iteration from a binary file.

    file_path: str - the path to the binary file to read from.
    modes: tuple - indicates which objects to yield (default: ('particles', 'nodes', 'walls')).
    Yields:
        - A tuple containing the nodes object (if 'nodes' in modes), the particles object (if 'particles' in modes),
          and the walls object (if 'walls' in modes).
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

            result = [serialized_data["iteration"]]
            if 'nodes' in modes:
                result.append(serialized_data.get('nodes'))
            if 'particles' in modes:
                result.append(serialized_data.get('particles'))
            if 'walls' in modes:
                result.append(serialized_data.get('walls'))

            yield tuple(result)

def cut_states(file_path: str, iteration: int):
    """
    Cut the data after the specified iteration from a binary file.

    file_path: str - the path to the binary file.
    iteration: int - the iteration after which to cut the data.
    """
    # Open the file in read-write mode.
    with open(file_path, "rb+") as f:
        header_size = 4
        data_offset = 0
        current_iteration = -1
        while current_iteration < iteration:
            f.seek(data_offset, os.SEEK_SET)
            size_bytes = f.read(header_size)
            if not size_bytes:
                break
            size = int.from_bytes(size_bytes, byteorder="big")
            data_offset = f.tell() + size
            if data_offset == header_size:
                break
            current_iteration += 1

        # Truncate the file at the position of the last complete data block.
        if current_iteration >= iteration:
            f.truncate(data_offset)

def force_mkdir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)

def prepare_system(params):
    constants = params['constants']
    numerical = params['numerical']
    geometry = params['geometry']
    periods = params['periods']
    modes = params['modes']
    filenames = params['filenames']

    L = geometry['L']
    N_x = numerical['N_x']
    N_p = numerical['N_p']
    h = L / N_x
    tau = numerical['tau']
    n = numerical["time_iterations"]
    pumping_windows = numerical["pumping_windows"]

    n0 = constants['n0']
    left_border =  geometry["left_border"]
    right_border =  geometry["right_border"]
    factor = right_border - left_border
    n1 = n0 * N_x * factor / N_p

    q = constants['q']
    m_e = constants['m_e']
    m_i = constants['m_i']*constants['atomic mass']
    epsilon = constants['epsilon']

    eV = constants['eV']
    T_e = constants['T_e']*eV
    T_i = constants['T_i']
    k_b = constants['k_b']

    E1 = constants['E1']*q
    alpha = constants['alpha']



    # Calculating velocity parameters
    v_t_e = math.sqrt(3 * k_b * T_e / m_e)     # Thermal velocity of electrons
    vmin_e = -3 * v_t_e                        # Minimum electron velocity
    vmax_e = 3 * v_t_e                         # Maximum electron velocity
    v_t_i = math.sqrt(3 * k_b * T_i / m_i)     # Thermal velocity of ions
    vmin_i = -3 * v_t_i                        # Minimum ion velocity
    vmax_i = 3 * v_t_i                         # Maximum ion velocity

    # Calculating the Debye length and adjusting grid spacing if necessary
    r_d = math.sqrt(epsilon * k_b * T_e / (q * q * n0))
    debye_factor = numerical["debye_factor"]
    if r_d * debye_factor < h:
        raise ValueError(f"Spatial step h larger then Debye radius:\n \
                         h = {h}\n\
                         r_d = {r_d}")
        
    # Calculating the plasma time step size and adjusting if necessary
    tau_plasma = 1 / (math.sqrt(n0 * q * q / (m_e * epsilon)) / (2 * np.pi))
    oscill_factor = numerical["oscill_factor"]
    if tau_plasma * oscill_factor < tau:
        print("adjusting tau to plasma oscillations")
        tau = tau_plasma * oscill_factor
        params["numerical"]["tau"] = tau
        print(f"new tau = {tau}")

    # Calculating the Courant time step size and adjusting if necessary
    tau_courant = numerical['courant_factor'] * h / v_t_e

    if tau_courant < tau:
        print("adjusting tau to courant condition")
        tau = tau_courant
        params["numerical"]["tau"] = tau
        print(f"new tau = {tau}")
    
    dir_name = f"logs/Te{T_e/eV}Nx{N_x}_Np{N_p}_h{h}_tau{tau}_n{n}"
    if modes['add_datetime']:
        now = datetime.now()
        formatted_now = now.strftime("%d_%m_%Y_%H_%M_%S")
        dir_name += "_" + formatted_now

    force_mkdir(dir_name)
    for key in filenames:
        filenames[key] = os.path.join(dir_name, filenames[key])
    # Creating left and right wall objects
    left_wall = Wall(0, L*left_border, 0, h, "left")
    right_wall = Wall(L*right_border, L, 0, h, "right")
    walls = (left_wall, right_wall)

    # Creating particle objects for ions and electrons and nodes
    ions = Particles(N_p, n1, q, m_i)
    electrons = Particles(N_p, n1, -q, m_e)
    nodes = Nodes(N_x)


    # Setting initial homogeneous distributions for electrons and ions
    set_homogeneous(electrons, left_wall.right*h, right_wall.left*h)
    set_homogeneous(ions, left_wall.right*h, right_wall.left*h)

    # Calculating acceleration constants for electrons and ions
    A_e = (electrons.q*(tau**2))/(electrons.m*h)
    A_i = (ions.q*(tau**2))/(ions.m*h)

    # Calculating integrals for electron and ion distributions
    integral_points = numerical["integral_points"]
    e_integral = get_integral(Maxwell(T_e, k_b, m_e), vmin_e, vmax_e, integral_points)
    i_integral = get_integral(Maxwell(T_i, k_b, m_i), vmin_i, vmax_i, integral_points)
    see_integral = get_integral(Maxwell(1*eV, k_b, m_e), vmin_e, vmax_e, integral_points)

    # Setting electron and ion distributions based on calculated integrals
    set_distr(electrons, e_integral, h, tau)
    set_distr(ions, i_integral, h, tau) 

    # Calculating the charge density on the grid nodes
    get_rho(nodes, electrons)
    get_rho(nodes, ions)

    # Calculating the electric potential and electric field on the grid nodes
    calc_fields(nodes, h, epsilon, walls=walls)

    # Accelerating electrons and ions based on the electric field
    accel(electrons, nodes, A_e, zerostep=True)
    accel(ions, nodes, A_i, zerostep=True)

    debye_cells = r_d/h
    offset = (v_t_e*tau/h)*periods["pumping"]
    max_range = [left_wall.right + debye_cells*6, right_wall.left - debye_cells*6]
    (max_range)
    max_range[0] += offset
    max_range[1] -= offset

    neutral_range = geometry["neutral_range"]
    if neutral_range[0] < max_range[0] or neutral_range[1] > max_range[1]:
        print("neutral range too wide!")
        print(f"neutral range: {neutral_range}")
        print(f"maximum range: {max_range}")
        print("adjusting...")

        max_width = max_range[1] - max_range[0]
        int_width = max_width - max_width%pumping_windows
        center = N_x/2
        neutral_range = (center-int_width/2, center+int_width/2)
        print(f"adjusted neutral_range: {neutral_range}")


    #computing constant for pumping
    constant_n = 0
    for particles in (electrons, ions):
        w_left = int(N_x/2)
        neutral_width = neutral_range[1] - neutral_range[0]
        w_right = w_left + int(neutral_width/pumping_windows)
        mask = range_mask(particles, (w_left, w_right))
        slc = particles[mask]
        constant_n += slc.n_macro
    print(constant_n)
    #deleting log file if exists
    statespath = filenames["system_states"]
    if os.path.isfile(statespath):
        os.remove(statespath)


    see_dict = {"E1": E1, 
                "alpha": alpha, 
                "h": h, 
                "tau": tau, 
                "see_integral": see_integral
    }

    calc_dict = {
    'time_iterations': n,
    "periods": periods,
    "modes": modes,
    'n_range': neutral_range,
    'A_e': A_e,
    'A_i': A_i,
    'see_dict': see_dict,
    'filenames': filenames,
    'constant_n': constant_n,
    'h': h,
    'tau': tau,
    'epsilon': epsilon,
    'e_integral': e_integral,
    "pumping_windows": numerical["pumping_windows"]
    }

    with open(os.path.join(dir_name, "params.bin"), 'ab+') as f:
        pickle.dump(params, f)
    
    # Return the required objects
    return (electrons, ions), nodes, walls, calc_dict

def main_cycle(electrons, ions, nodes, walls, calc_dict):
    time_iterations = calc_dict['time_iterations']
    saving_period = calc_dict["periods"]['saving']
    pumping_period = calc_dict["periods"]['pumping']
    pumping_offset = calc_dict["periods"]['pumping_offset']
    maxwellise_period = calc_dict["periods"]['maxwellise']
    saving = calc_dict["modes"]['saving']
    pumping = calc_dict["modes"]['pumping']
    maxwellise = calc_dict["modes"]['maxwellise']

    system_states_path = calc_dict["filenames"]['system_states']

    n_range = calc_dict['n_range']
    A_e = calc_dict['A_e']
    A_i = calc_dict['A_i']
    see_dict = calc_dict['see_dict']
    
    constant_n = calc_dict['constant_n']
    see_dict = calc_dict['see_dict']
    h = calc_dict['h']
    tau = calc_dict['tau']
    epsilon = calc_dict['epsilon']
    e_integral = calc_dict['e_integral']
    pumping_windows = calc_dict["pumping_windows"]

    print(f"h = {h}, tau = {tau}")
    print("Launching calculations...")
    for t in tqdm(range(time_iterations)):
        try:
            move(electrons, nodes)
            move(ions, nodes)
        except Exception:
            print("Number of iteration:", t)
            break

        # Resetting charge densities
        nodes.rho *= 0
        nodes.conc_i *= 0
        nodes.conc_e *= 0

        # Accounting for wall interactions
        account_walls([electrons, ions], walls, SEE=see_dict)

        # Calculating charge densities
        get_rho(nodes, electrons)
        get_rho(nodes, ions)

        # Calculating electric fields and potentials
        calc_fields(nodes, h, epsilon, walls=walls)

        # Accelerating particles
        accel(electrons, nodes, A_e)
        accel(ions, nodes, A_i)

        # Saving system state
        if saving and t % saving_period == 0:
            save_system_state(t, nodes, (electrons, ions), walls, system_states_path)
        for kind in ["secondary_electrons", "absorbed_electrons", "absorbed_ions"]:
            if kind in see_dict:
                save_system_state(t, nodes, (see_dict[kind]), walls, calc_dict["filenames"][kind], modes=["particles"])
                see_dict.pop(kind)

        # Applying Maxwellian distribution
        if maxwellise and t % maxwellise_period == 0:
            set_distr(electrons, e_integral, h, tau, n_range)

        # Pumping particles
        if pumping and t % pumping_period == 0 and t > pumping_offset:
            pump_particles((electrons, ions), constant_n, n_range, windows=pumping_windows)
