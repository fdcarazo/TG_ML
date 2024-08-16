#!/usr/bin/python3.9
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import chaospy as cp

from scipy.stats.qmc import Sobol, Halton, LatinHypercube

# Number of iterations for the loop
num_iterations = 5
samples_size = 2**10

# Loop to generate random numbers with different sequences
list_res= list()
for i in range(num_iterations):
    np.random.seed(np.random.randint(1000000))
    # Define the distribution (uniform between 300 and 400)
    distribution = cp.Uniform(300, 400)

    # Create a new sampler instance for each iteration
    # Definir el metodo QMC (Sobol en este caso)-.
    # i = np.random.randint(1000000)
    #sample = distribution.sample(samples_size,
    #                             rule='sobol',
    #                             seed=np.random.randint(1000000))


    # Create Sobol sequence with different seed for each iteration
    seed = np.random.randint(1000000)
    # distribution = cp.Iid(cp.Uniform(0, 1), 1)
    sample = distribution.sample(samples_size, seed=seed, rule="sobol")
    # sequence = cp.create_sobol_samples(distribution, samples_size, seed=seed)
    # Transform the numbers to fit the desired range
    random_numbers = sample * (400 - 300) + 300

    sobol_seq = Sobol(d=6, seed=seed, scramble=True)

    ## print(sobol_seq.random_base2(10)); input(11)
    # sobol_seq = Halton(d=1, seed=seed)
    # sobol_seq = LatinHypercube(d=1, seed=seed)
    
    # Generate Sobol sequence and scale to desired range (300 to 400)
    # sample = 300 + sobol_seq.random(samples_size) * 100
    # sample = sobol_seq[:,0]
    ## sample = sobol_seq.random_base2(10)
    sample = sobol_seq.random(400)
    print(np.shape(sample)); input(333333)
    
    # Print or use the random numbers for each iteration
    print(f"Iteration {i+1}: {sample}")

    #### list_res.append(random_numbers)
    list_res.append(sample)
    # del sobol_seq, sample
    # del distribution, sample, random_numbers
    ## Sobol.reset(sobol_seq)
    sobol_seq.reset()
#
x = list_res[1]
y = list_res[4]
print(np.shape(x), np.min(x[:,0]), np.max(x[:,0]))
plt.scatter(x[:, 1], y[:, 0], s=3, marker='o')
#plt.scatter(x, y)
plt.show()
#    print(x, y)


## https://stackoverflow.com/questions/76792777/my-two-dimensional-sobol-points-seem-wrong-from-the-scipy-qmc-module
## https://stackoverflow.com/questions/69855127/scipy-stats-qmc-how-to-do-multiple-randomized-quasi-monte-carlo
