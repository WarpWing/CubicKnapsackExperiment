import numpy as np
import random

# Set the random seed for reproducibility
np.random.seed(500000)

# Problem specifications (I had to massively scale down search space for demonstration)
num_items = 1000
max_value = 100
max_weight = 40
max_volume = 20
knapsack_max_weight = 10000
knapsack_max_volume = 6000

# Generate items
items = [(random.randint(1, max_value), random.randint(1, max_weight), random.randint(1, max_volume)) for _ in range(num_items)]

# Genetic Algorithm parameters
population_size = 100
mutation_rate = 0.01
crossover_rate = 0.8
generation_limit = 100 # Adjusted for demonstration

def fitness(solution, items):
    total_value, total_weight, total_volume = 0, 0, 0
    for i, selected in enumerate(solution):
        if selected:
            value, weight, volume = items[i]
            total_value += value
            total_weight += weight
            total_volume += volume
    if total_weight > knapsack_max_weight or total_volume > knapsack_max_volume:
        return 1, total_value, total_weight, total_volume
    return 0, total_value, total_weight, total_volume 

def select(population, fitnesses):
    total_fitnesses = [1 - f[0] for f in fitnesses]  # Invert fitness for selection (higher is better)
    if sum(total_fitnesses) == 0: 
        return random.sample(population, 2)
    else:
        return random.choices(population, weights=total_fitnesses, k=2)

def crossover(parent1, parent2):
    if random.random() < crossover_rate:
        point = random.randint(1, len(parent1) - 2)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
    return parent1, parent2

def mutate(solution):
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            solution[i] = 1 - solution[i]
    return solution

def genetic_algorithm():
    population = [[random.randint(0, 1) for _ in range(num_items)] for _ in range(population_size)]
    generation_details = []

    for generation in range(generation_limit):
        fitnesses = [fitness(individual, items) for individual in population]
        best_index = max(range(len(fitnesses)), key=lambda i: fitnesses[i][0])
        best_fitness, best_value, best_weight, best_volume = fitnesses[best_index]
        
        generation_details.append((generation + 1, best_fitness, best_value, best_weight, best_volume))

        if best_fitness == 0:  
            break

        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])
        population = new_population

    for gen in generation_details:
        print(f"Generation {gen[0]}: Fitness = {gen[1]}, Value = {gen[2]}, Weight = {gen[3]}/{knapsack_max_weight}, Volume = {gen[4]}/{knapsack_max_volume}")

    if generation_details[-1][1] != 0:
        print("No feasible solution found within the generation limit.")

    return population[best_index]


best_solution = genetic_algorithm()
