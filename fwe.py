from plasma_classes import *
from plasma_utils import *
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

def account_walls(particles: Particles, walls: list[Wall], E1: float, alpha: float, SEE=False):
    for wall in walls:
        # Identifying the absorbed particles
        absorbed_mask = (particles.x <= wall.right) & (particles.x >= wall.left)
        absorbed_indices = np.where(absorbed_mask)

        # Step 1: Discern particles capable of generating secondary electrons
        energy = 0.5 * particles.m * particles.v ** 2
        emit_mask = energy > E1
        absorbed_emit_mask = absorbed_mask & emit_mask

        # Step 2: Calculate the secondary electron emission yield (σ)
        sigma = ((energy[absorbed_emit_mask]) / E1) ** alpha

        # Step 3: Adding generated electrons to the system and ions to the wall
        if SEE:
            n_secondary = int(np.sum(sigma))
            probabilities = np.random.rand(len(sigma))
            secondary_counts = (probabilities < (sigma - np.floor(sigma))).astype(int) + np.floor(sigma).astype(int)
            total_secondary = np.sum(secondary_counts)

            new_electron_params = (particles.concentration, particles.q, particles.m)
            new_electrons = Particles(total_secondary, *new_electron_params)
            new_electrons.x = np.repeat(particles.x[absorbed_emit_mask], secondary_counts)

            particles.x = np.concatenate([particles.x, new_electrons.x])
            particles.v = np.concatenate([particles.v, new_electrons.v])
            particles.n_macro += total_secondary
            wall.particles_lst.append(new_electrons)

        params = (particles.concentration, particles.q, particles.m)

        absorbed_particles = Particles(particles.n_macro, *params)
        absorbed_particles.n_macro = np.sum(absorbed_mask)
        absorbed_particles.x = np.random.uniform(wall.left+wall.h/10, wall.right-wall.h/10)

        # Excluding absorbed particles from the original set
        particles.x = particles.x[~absorbed_mask]
        particles.v = particles.v[~absorbed_mask]
        particles.n_macro = len(particles.x)


def test_see_implementation():
    # Test Setup
    particles = Particles(n_macro=100, concentration=1e20, q=-1.6e-19, m=9.11e-31)
    particles.x = np.linspace(-0.1, 1.1, particles.n_macro)
    particles.v = 1e6 * np.ones_like(particles.x)
    walls = [Wall(x=0.0, h=0.01), Wall(x=1.0, h=0.01)]
    E1 = 5e-17  # Secondary electron emission threshold energy
    alpha = 1.0

    # Step 1: Run the initial simulation and find absorbed electrons
    account_walls(particles, walls, E1, alpha, SEE=True)

    # Step 2: Calculate expected secondary electron emission values
    absorbed_mask = (particles.x <= 0) | (particles.x >= 1)
    energy = 0.5 * particles.m * particles.v ** 2
    emit_mask = energy > E1
    absorbed_emit_mask = absorbed_mask & emit_mask
    sigma = ((energy[absorbed_emit_mask]) / E1) ** alpha

    # Test 1: Check the total number of emitted secondary electrons
    n_secondary = int(np.sum(sigma))
    assert particles.n_macro - 100 == n_secondary, "The total number of emitted secondary electrons is incorrect."

    # Test 2: Check the distribution of emitted secondary electrons among the absorbed primary electrons
    probabilities = np.random.rand(len(sigma))
    secondary_counts = (probabilities < (sigma - np.floor(sigma))).astype(int) + np.floor(sigma).astype(int)
    total_secondary = np.sum(secondary_counts)
    assert n_secondary == total_secondary, "The distribution of emitted secondary electrons is incorrect."


# Executing the test
test_see_implementation()


