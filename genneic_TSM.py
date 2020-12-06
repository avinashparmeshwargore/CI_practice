import random
import numpy as np

random.seed(11)
np.random.seed(11)

def to_binary(i, n):
    b = '{0:b}'.format(i)
    while len(b) < n:
        b = '0' + b
    return b

def to_int(b):
    return int(b, 2)

## test to verify,
##    in: 4 -> in: '00100' -> out: 4
assert to_int(to_binary(4, 5)) == 4
def crossover(i1, i2):
    assert len(i1) == len(i2)
    n = len(i1); split_index = random.randint(0+1, n-1)
    return '{}{}'.format(i1[:split_index], i2[split_index:]), \
           '{}{}'.format(i1[split_index:], i2[:split_index])

## in: ('00010', '11000') out: ('00|000', '010|11') with split at 2.
def mutate(i1, threshold):
    def swap_bit(c):
        if c == '0':
            return '1'
        return '0'
    
    return ''.join([ \
        swap_bit(c) if random.uniform(0, 1) <= threshold \
                    else c for c in i1 \
    ])
def ranking_population(individuals):
    def calc_fitness(i1):
        ## as defined in the text = (1000 - | v^2 - 64 | )
        return 1000 - abs(to_int(i1)**2 - 64)
    
    return sorted( \
        [(i, calc_fitness(i)) for i in individuals], \
        key=lambda x: x[1], reverse=True)
n = 5
min_val = to_int('0'*n)  ## '00000'
max_val = to_int('1'*n)  ## '11111'

n_individuals = 4
individual_indexes = list(range(0, n_individuals))
population = [ to_binary(random.randint(min_val, max_val), 5) \
    for i in range(0, n_individuals) ]
n_matings = 5
n_iterations = 1000
mutation_threshold = .01

for iteration in range(0, n_iterations):
    
    new_population = population.copy()
    
    ## crossovers
    for _ in range(0, n_matings):
        items_to_mate = np.random.choice( \
            individual_indexes, size=2, replace=False)
        i1, i2 = crossover(population[items_to_mate[0]],population[items_to_mate[1]])
        new_population.append(i1)
        new_population.append(i2)
        
    ## mutates    
    for individual in new_population:
        individual = mutate(individual, mutation_threshold)
        
    ## rank the individuals in the population
    ranked_population = ranking_population(new_population)
    
    if ranked_population[0][1] == 1000:
        ### optimal found,
        break
    
    ## figure out the individuals that survived?
    
    ### take the top 2,
    population = [
        ranked_population[0][0],
        ranked_population[1][0]
    ]
    
    ### allow the others to fight it out,
    those_that_survived = np.random.choice( \
        range(2, len(ranked_population)), size=2, replace=False)
    for i in those_that_survived:
        population.append(ranked_population[i][0])
        