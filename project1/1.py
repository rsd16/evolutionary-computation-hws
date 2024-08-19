import numpy as np
import random


mutation_rate = 0.5
crossover_rate_gene = 0.5
crossover_rate_chromosome = 1.0 # Meaning, all of the parents can participate in Crossover process.
initial_population_size = 10

def population_initializer(initial_population_size):
    population = []
    for i in range(initial_population_size):
        number = np.random.randint(256)
        binary_form = bin(number).replace('0b','')
        binary_form = binary_form[::-1]
        while len(binary_form) < 8:
            binary_form += '0'

        binary_form = binary_form[::-1]
        population.append(binary_form) # For example, our population is: ['01010110', '10001101', ...]

    return population

def fitness_function(population):
    fitness_values = []
    for chromosome in population:
        number = int(chromosome, 2)
        fitness_value = number ** 2
        fitness_values.append(fitness_value)

    return fitness_values

def probability_function(fitness_values):
    sum_probabilities = np.sum(fitness_values)
    probabilities = []
    for fitness_value in fitness_values:
        probabilities.append(fitness_value / sum_probabilities)

    return probabilities

def expected_count_function(fitness_values):
    mean_fitness = np.mean(fitness_values)
    expected_counts = []
    for fitness_value in fitness_values:
        expected_counts.append(fitness_value / mean_fitness)

    return expected_counts

def parent_selection(population, probabilities):
    mating_pool = random.choices(population, probabilities, k=initial_population_size)
    return mating_pool

def one_point_crossover(mating_pool):
    offsprings = []
    for _ in range(int(initial_population_size / 2)):
        crossover_point = np.random.randint(8)
        parent_father = random.choices(mating_pool)[0]
        mating_pool.remove(parent_father)
        parent_mother = random.choices(mating_pool)[0]
        mating_pool.remove(parent_mother)
        parent_father = list(parent_father)
        parent_mother = list(parent_mother)
        for j in range(crossover_point, len(parent_father)):
            parent_father[j], parent_mother[j] = parent_mother[j], parent_father[j]

        parent_father = ''.join(parent_father)
        offsprings.append(parent_father)
        parent_mother = ''.join(parent_mother)
        offsprings.append(parent_mother)

    return offsprings

def two_point_crossover(mating_pool):
    def single_point(parent_father, parent_mother, j):
        parent_father_new = np.append(parent_father[:j], parent_mother[j:])
        parent_mother_new = np.append(parent_mother[:j], parent_father[j:])
        return parent_father_new, parent_mother_new

    offsprings = []
    for _ in range(int(initial_population_size / 2)):
        crossover_points = [np.random.randint(8), np.random.randint(8)]
        crossover_points = sorted(crossover_points)
        parent_father = random.choices(mating_pool)[0]
        mating_pool.remove(parent_father)
        parent_mother = random.choices(mating_pool)[0]
        mating_pool.remove(parent_mother)
        parent_father = list(parent_father)
        parent_mother = list(parent_mother)
        for j in crossover_points:
            parent_father, parent_mother = single_point(parent_father, parent_mother, j)

        parent_father = ''.join(parent_father)
        offsprings.append(parent_father)
        parent_mother = ''.join(parent_mother)
        offsprings.append(parent_mother)

    return offsprings

def uniform_crossover(mating_pool):
    offsprings = []
    for i in range(int(initial_population_size / 2)):
        crossover_probabilities = np.random.rand(8)
        parent_father = random.choices(mating_pool)[0]
        mating_pool.remove(parent_father)
        parent_mother = random.choices(mating_pool)[0]
        mating_pool.remove(parent_mother)
        parent_father = list(parent_father)
        parent_mother = list(parent_mother)
        for j in range(len(crossover_probabilities)):
            if crossover_probabilities[j] > crossover_rate_gene:
                temp = parent_father[j]
                parent_father[j] = parent_mother[j]
                parent_mother[j] = temp

        parent_father = ''.join(parent_father)
        offsprings.append(parent_father)
        parent_mother = ''.join(parent_mother)
        offsprings.append(parent_mother)

    return offsprings

def mutation(offsprings):
    for i, chromosome in enumerate(offsprings):
        chromosome = list(chromosome)
        for j, gene in enumerate(chromosome):
            gene_mutation_rate = random.uniform(0, 1)
            if gene_mutation_rate < mutation_rate:
                if gene == '0':
                    chromosome[j] = '1'
                else:
                    chromosome[j] = '0'

        chromosome = ''.join(chromosome)
        offsprings[i] = chromosome

    return offsprings

def main():
    for i in range(10):
        print('#####################################################################')
        print(f'Generation {i + 1} of Evolution Algorithm begins...')
        print()
        if i == 0:
            population = population_initializer(initial_population_size) # We initialize the population in first iteration.

        print('Our population is: ')
        for chromosome in population:
            print('The genotype is: ', chromosome, ' And, the phenotype is: ', int(chromosome, 2))

        print()

        # We calculated fitness value for each chromosome in our population:
        fitness_values = fitness_function(population)
        print('Before Operators... Fitness values are: ', fitness_values)

        # Maximum Fitness value:
        max_fitness = np.max(fitness_values)
        print('Before Operators... Maximum Fitness is: ', max_fitness)

        # Sum of Fitness value:
        sum_fitness = np.sum(fitness_values)
        print('Before Operators... Total sum of Fitness values is: ', sum_fitness)

        # Average of Fitness value:
        average_fitness = np.mean(fitness_values)
        print('Before Operators... Average of Fitness values is: ', average_fitness)

        # We calculate the probability for each chromosome, so we can choose parents:
        probabilities = probability_function(fitness_values)

        # We calculate the Expected Counts for each chromosome:
        #expected_counts = expected_count_function(fitness_values)
        #print('Before Operators... Expected counts are: ', expected_counts)

        # We calculate the Actual Counts for each chromosome:
        #actual_counts = np.round(expected_counts)
        #print('Before Operators... Actual counts are: ', actual_counts)

        # We select parents:
        mating_pool = parent_selection(population, probabilities)

        # One-Point Crossover:
        offsprings = one_point_crossover(mating_pool)

        # Two-Point Crossover:
        #offsprings = two_point_crossover(mating_pool)

        # Uniform Crossover:
        #offsprings = uniform_crossover(mating_pool)

        # Mutation:
        offsprings = mutation(offsprings)

        # At the end, the new population will be the offsprings.
        population = offsprings
        print()
        print('Our new population is: ', population)
        print()

        # We calculated fitness value for each chromosome in our population:
        fitness_values = fitness_function(population)
        print('After Operators... Fitness values are: ', fitness_values)

        # Maximum Fitness value:
        max_fitness = np.max(fitness_values)
        print('After Operators... Maximum Fitness is: ', max_fitness)

        # Sum of Fitness value:
        sum_fitness = np.sum(fitness_values)
        print('After Operators... Total sum of Fitness values is: ', sum_fitness)

        # Average of Fitness value:
        average_fitness = np.mean(fitness_values)
        print('After Operators... Average of Fitness values is: ', average_fitness)
        print('\n###################################################################')

if __name__ == '__main__':
    main()
