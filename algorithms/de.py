import numpy as np

from src.utils import generer_solution_aleatoire
from src.contraintes import (penalisation_stat, degre_violation,  Bornes_minimum, Bornes_maximum)


def DE(max_iter, lambda_penalite, population_size, F, CR, epsilon, max_stagnation):

    # Initialisation population
    population = []

    for _ in range(population_size):
        x = np.array(generer_solution_aleatoire())
        population.append(x) # Chaque individu est une solution

    history = []
    nb_stagnation = 0

    for i in range(max_iter):

       
        old_best = min( penalisation_stat(ind[0], ind[1], ind[2], lambda_penalite) for ind in population)

        new_population = []

        for j in range(population_size): # On traite chaque individu

            # Choisir 3 individus distincts
            indices = list(range(population_size))
            indices.remove(j)
            # On ne veut pas utiliser l'individu courant pour la mutation


            a, b, c = np.random.choice(indices, 3, replace=False)

            # Mutation
            mutant = population[a] + F * (population[b] - population[c])
            mutant = np.clip(mutant, Bornes_minimum, Bornes_maximum)

            # Croisement
            trial = population[j].copy() # on prend l'individu courant comme base pour le trial
            for k in range(3):
                if np.random.rand() < CR: # avec une certaine probabilité, on remplace le gène par celui du mutant
                    trial[k] = mutant[k]
            trial_fitness = penalisation_stat( trial[0], trial[1], trial[2], lambda_penalite)
            target_fitness = penalisation_stat(population[j][0], population[j][1], population[j][2], lambda_penalite)

            if trial_fitness < target_fitness:
                new_population.append(trial)
            else:
                new_population.append(population[j])

        population = new_population

        best_fitness = min(penalisation_stat(ind[0], ind[1], ind[2], lambda_penalite)for ind in population)
        history.append(best_fitness)

        # Critère de stagnation
        if abs(old_best - best_fitness) < epsilon:
            nb_stagnation += 1
        else:
            nb_stagnation = 0

        if nb_stagnation >= max_stagnation:
            break

    best_x = min(population,key=lambda ind: penalisation_stat(ind[0], ind[1], ind[2], lambda_penalite))

    best_fitness = penalisation_stat(best_x[0], best_x[1], best_x[2], lambda_penalite)

    violation = degre_violation(best_x[0], best_x[1], best_x[2])

    return best_x, best_fitness, history, violation



def monte_carlo_de(n_runs, max_iter, lambda_penalite, population_size, F, CR, epsilon, max_stagnation):

    all_final_fitness = []
    all_histories = []
    all_violations = []

    for run in range(n_runs):
        np.random.seed(run)

        best_x, best_fitness, history, violation = DE( max_iter, lambda_penalite, population_size, F, CR, epsilon, max_stagnation )

        all_final_fitness.append(best_fitness)
        all_violations.append(violation)

        if len(history) < max_iter:
            history += [history[-1]] * (max_iter - len(history))

        all_histories.append(history)

    return (np.array(all_final_fitness), np.array(all_histories), np.array(all_violations))