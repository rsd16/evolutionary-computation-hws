import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, other_city):
        x_distance = abs(self.x - other_city.x)
        y_distance = abs(self.y - other_city.y)
        distance = np.sqrt((x_distance ** 2) + (y_distance ** 2))
        return distance

    def show_coordinates(self):
        return self.x, self.y


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def route_distance(self):
        if self.distance == 0:
            route_distance = 0
            for i in range(0, len(self.route)):
                from_city = self.route[i]
                to_city = None
                if i + 1 < len(self.route):
                    to_city = self.route[i + 1]
                else:
                    to_city = self.route[0]

                route_distance += from_city.distance(to_city)

            self.distance = route_distance

        return self.distance

    def route_fitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.route_distance())

        return self.fitness


def initial_population(initial_population_size, cities):
    population = []
    for i in range(0, initial_population_size):
        population.append(random.sample(cities, len(cities)))

    return population

def fitness_function(population):
    fitness_values = {}
    for i in range(0, len(population)):
        fitness_values[i] = Fitness(population[i]).route_fitness()

    return sorted(fitness_values.items(), key=operator.itemgetter(1), reverse=True)

# Roulette Wheel Selection:
def selection(population_ranked, elite_size):
    parents = []
    df = pd.DataFrame(np.array(population_ranked), columns=['Index', 'Fitness'])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum() # Calculate the proability for each chromosome
    for i in range(0, elite_size):
        parents.append(population_ranked[i][0])

    for i in range(0, len(population_ranked) - elite_size):
        pick = 100 * random.random()
        for i in range(0, len(population_ranked)):
            if pick <= df.iat[i, 3]:
                parents.append(population_ranked[i][0])
                break

    return parents

def mating_pool(population, parents):
    mating_pools = []
    for i in range(0, len(parents)):
        index = parents[i]
        mating_pools.append(population[index])

    return mating_pools

def crossover_function(parent1, parent2):
    child = []
    child_parent1 = []
    child_parent2 = []
    gene1 = int(random.random() * len(parent1))
    gene2 = int(random.random() * len(parent1))
    start_gene = min(gene1, gene2)
    end_gene = max(gene1, gene2)
    for i in range(start_gene, end_gene):
        child_parent1.append(parent1[i])

    child_parent2 = [item for item in parent2 if item not in child_parent1]
    child = child_parent1 + child_parent2
    return child

def crossover(mating_pool, elite_size):
    children = []
    length = len(mating_pool) - elite_size
    pool = random.sample(mating_pool, len(mating_pool))
    for i in range(0, elite_size):
        children.append(mating_pool[i])

    for i in range(0, length):
        child = crossover_function(pool[i], pool[len(mating_pool) - i - 1])
        children.append(child)

    return children

def mutatation_function(chromosome, mutation_rate):
    for to_swap in range(len(chromosome)):
        if (random.random() < mutation_rate):
            swap_with = int(random.random() * len(chromosome))
            city1 = chromosome[to_swap]
            city2 = chromosome[swap_with]
            chromosome[to_swap] = city2
            chromosome[swap_with] = city1

    return chromosome

def mutatate(population, mutation_rate):
    mutated_population = []
    for i in range(0, len(population)):
        mutated_index = mutatation_function(population[i], mutation_rate)
        mutated_population.append(mutated_index)

    return mutated_population

def next_generation(current_population, elite_size, mutation_rate):
    # Calculate Fitness for the population and sort them:
    population_ranked = fitness_function(current_population)

    # Select best parents:
    selected_population = selection(population_ranked, elite_size)

    # Gather parents:
    parents = mating_pool(current_population, selected_population)

    # Ordered Crossover:
    offsprings = crossover(parents, elite_size)

    # Mutation:
    offsprings = mutatate(offsprings, mutation_rate)

    return offsprings


def genetic_algorithm(population, initial_population_size, elite_size, mutation_rate, generations):
    population = initial_population(initial_population_size, population)
    print('Initial distance: ' + str(1 / fitness_function(population)[0][1]))
    progress = []
    progress.append(1 / fitness_function(population)[0][1])
    acceptable_distance = 700
    for i in range(0, generations):
        population = next_generation(population, elite_size, mutation_rate)
        final_fitness = 1 / fitness_function(population)[0][1]
        progress.append(1 / fitness_function(population)[0][1])
        if final_fitness < acceptable_distance:
            print('Closed in ', i)
            print(final_fitness)
            break

    print('Final distance: ' + str(1 / fitness_function(population)[0][1]))
    best_route_index = fitness_function(population)[0][0]
    best_route = population[best_route_index]
    for coordinate in best_route:
        print(coordinate.show_coordinates())

    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()

def main():
    cities = []
    with open('TSP51.txt', 'r') as file:
        for line in file:
            digits = line.split(' ')
            cities.append(City(x=int(digits[1]), y=int(digits[2])))

    genetic_algorithm(population=cities, initial_population_size=100, elite_size=20, mutation_rate=0.01, generations=200)

if __name__ == '__main__':
    main()
