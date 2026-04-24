import numpy as np
from src.utils import generer_solution_aleatoire
from src.contraintes import penalisation_stat, degre_violation, Bornes_minimum, Bornes_maximum


# Fonction du pso 
def pso(max_iter, lambda_penalite, swarm_size, w, c1, c2, espilon, max_stagnation): 

    # Initialisation de la population
    particules =[] # Liste des particules
    velocites = [] # Liste des vitesses
    best_positions = [] # Liste des meilleures positions
    best_scores = [] # Liste des meilleurs scores

    for _ in range(swarm_size):
        x = np.array(generer_solution_aleatoire()) # Générer une solution aléatoire
        v = np.random.uniform(-1, 1, size=3) # Générer une vitesse aléatoire
        fitness = penalisation_stat(x[0], x[1], x[2], lambda_penalite)

        # On stocke la particule, sa vitesse, sa meilleure position et son meilleur score
        particules.append(x)
        velocites.append(v)
        best_positions.append(x.copy())
        best_scores.append(fitness)

    # Initialisation de la meilleure position globale
    best_index = np.argmin(best_scores)
    global_best_position = best_positions[best_index].copy()
    global_best_score = best_scores[best_index]

    for i in range(max_iter) : # à chaque itération, toutes les particules se déplacent
        old_best = global_best_score # Ancienne meilleure fitness

        for j in range(swarm_size): # Pour chaque particule

            # On génère deux nombres aléatoires pour le déplacement vectioriel de la particule
            r1 = np.random.rand(3)
            r2 = np.random.rand(3)

            # Mise à jour vitesse
            velocities[j] = (
                w * velocities[j] # C'est l'inertie de la particule
                + c1 * r1 * (best_positions[j] - particles[j])
                + c2 * r2 * (global_best_position - particles[j]) # attiré par son meilleur personnel et le meilleur global
            ) # inertie + attraction personnelle + attraction globale

            # Mise à jour de la position
            particles[j] = particles[j] + velocities[j]

            # Gestion des bornes
            particles[j] = np.clip(particles[j], Bornes_minimum, Bornes_maximum)

            # Calcul de la nouvelle fitness
            fitness = penalisation_stat(
                particles[j][0],
                particles[j][1],
                particles[j][2],
                lambda_penalite
            )

            # Mise à jour meilleur personnel
            if fitness < best_scores[j]:
                best_scores[j] = fitness
                best_positions[j] = particles[j].copy()

            # Mise à jour meilleur global
            if fitness < global_best_score:
                global_best_score = fitness
                global_best_position = particles[j].copy()

        history.append(global_best_score)

        # Critère de stagnation
        if abs(old_best - global_best_score) < epsilon:
            nb_stagnation += 1
        else:
            nb_stagnation = 0

        if nb_stagnation >= max_stagnation:
            break

    violation = degre_violation(
        global_best_position[0],
        global_best_position[1],
        global_best_position[2]
    )

    return global_best_position, global_best_score, history, violation




def monte_carlo_pso(n_runs, max_iter, lambda_penalite, swarm_size, w, c1, c2, epsilon, max_stagnation):

    all_final_fitness = []
    all_histories = []
    all_violations = []

    for run in range(n_runs):
        np.random.seed(run)

        global_best_position, global_best_score, history, violation = PSO( max_iter, lambda_penalite, swarm_size, w, c1, c2, epsilon, max_stagnation )

        all_final_fitness.append(global_best_score)
        all_violations.append(violation)

        # compléter si arrêt anticipé
        if len(history) < max_iter:
            history += [history[-1]] * (max_iter - len(history))

        all_histories.append(history)

    return (np.array(all_final_fitness), np.array(all_histories), np.array(all_violations))