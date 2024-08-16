#!/usr/bin/python3.9
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats.qmc import LatinHypercube, Sobol

import chaospy as cp
import numpy as np

def get_samples(samples_size, name_sampler, prop_bounds):

    '''
    TBC
    '''

    min_val, max_val = prop_bounds
    print(min_val, max_val, sep='\n'); input(77)
    if name_sampler == 'random':
        # @ complete_random_sampler-.
        np.random.seed(0)
        # generate 100 random numbers between 0 and 1-.
        random_numbers_01 = np.random.rand(samples_size)
        # linear transformaction to adjust at range [min_val,max_val]-.
        sample = min_val + random_numbers_01 * (max_val - min_val)
    elif name_sampler == 'random_uniform':
        # @ uniform_random_sampler-.
        sample = np.random.uniform(max_val, min_val, samples_size)
    elif name_sampler == 'stratified':
        # to generate stratified sample (stratified_sampler)-.
        n_stratas = 4
        strata_size = samples_size // n_stratas
        sample = np.concatenate([np.random.rand(strata_size)
                                 for _ in range(n_stratas)])
    elif name_sampler == 'latin_hypercube':
        # to generate hypercube_latine (hypercube_latine_sampler)-.
        latin_hypercube = LatinHypercube(d=2, seed=42)
        random_numbers_01 = latin_hypercube.random(n=samples_size)[:, 0]  # NOTE: MBC-.
        sample = min_val + random_numbers_01 * (max_val - min_val)
    elif name_sampler == 'sobol':
        # to generate sobol sequence to get sample (sobol_sampler)-.
        sobol = Sobol(d=2, seed=42)
        random_numbers_01 = sobol.random(n=samples_size)[:, 1]   # NOTE: MBC-.
        sample = min_val + random_numbers_01 * (max_val - min_val)
    return sample


features_dict = {'EM': (3.5, 10.),      # [GPa] to fiber properties Young modulus-.
                 'nuM': (0.25, 0.49),   # fiber properties Poisson ratio-.
                 'EF': (69., 786.),     # [GPa] # matrix properties Young modulus-.
                 'nuF': (0.2, 0.4),     # matrix properties Poisson ratio-.
                 'phiF': (0.0, 0.7)}    # fiber volume fraction-.
                     

# list or numpy-array with number of samples-.
min_samples, max_samples, interval = 100, 1000, 10
n_samples = np.linspace(start=min_samples, stop=max_samples,
                        num=interval, endpoint=True)
# print(n_samples); input(445566)
# list with names of samplers-.
samplers_names = ['random', 'random_uniform', 'stratified',
                  'latin_hypercube', 'sobol']


# calculate -.
for samples in n_samples:
    dict_to_df = {}

    
# main(features_dict, n_samples, samplers_names) -> int:
# main(features_dict, n_samples, samplers_names)

samplers_names = ['sobol']
for name in samplers_names:
    for i_prop, (prop_name, prop_bounds) in enumerate(features_dict.items()):
        print(int(samples), name, prop_bounds, sep='\n'); input(11111)
        sample = get_samples(int(samples), name, prop_bounds)
        # print(i_prop, prop_name, prop_bounds, sep='\n'); input(8899)
        dict_to_df[str(prop_name)] = sample
                
    df = pd.DataFrame(dict_to_df)
    # df['suma_total'] = df.apply(suma, axis=1)
    # df['suma_total_mod'], df['2*suma_total_mod'] = zip(*df.apply(suma_mod, axis=1))
    ## df['E11'], df['E22'], df['nu12'], df['nu23'], df['G12'], df['G23'] = zip(*df.apply(PMM, axis=1))
    
    ## df_name_to_save = os.getcwd()+'/ds/'+str(name)+str(int(samples))+'.csv'
    ## df.to_csv(df_name_to_save, sep=',', header=True, index=True, )


x, y = df.iloc[:, 1], df.iloc[:, 2]


min_val, max_val, samples_size = 100, 200, 1000

# x = np.random.uniform(0, 1, 1000)
# y = np.random.uniform(min_val, max_val, samples_size)

sobol = Sobol(d=2, seed=42)
random_numbers_01 = sobol.random(n=samples_size)[:, 1]   # NOTE: MBC-.
sample = min_val + random_numbers_01 * (max_val - min_val)

print(np.shape(sample))
# y = sample
plt.scatter(x, y)
plt.show()
print(x, y)



# Define the distribution (uniform between 300 and 400)
distribution = cp.Uniform(300, 400)

# Number of iterations for the loop
num_iterations = 5

# Loop to generate random numbers with different sequences
for i in range(num_iterations):
    # Create a new sampler instance for each iteration
    sampler = cp.Sampler(distribution, rule="S", seed=i)
    
    # Generate 10 random numbers using the sampler
    random_numbers = sampler.sample(10)
    
    # Transform the numbers to fit the desired range
    random_numbers = random_numbers * (400 - 300) + 300
    
    # Print or use the random numbers for each iteration
    print(f"Iteration {i+1}: {random_numbers}")


# Definir la distribucion uniforme en el rango deseado-.
distribution = cp.Uniform(min_val, max_val)
# u_samples = chaospy.Uniform(0, 1).sample(10000,
#                                          seed=np.random.seed(np.random.randint(1000000))
# Definir el metodo QMC (Sobol en este caso)-.
seed_num = np.random.randint(1000000)
sample = distribution.sample(samples_size,
                             rule='sobol',
                             seed=seed_num)
del distribution
