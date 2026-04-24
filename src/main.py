import argparse
import numpy as np
from algorithms import (
    monte_carlo_random_search,
    monte_carlo_hill_climbing,
    monte_carlo_generalized_hill_climbing,
    monte_carlo_recuit_simule,
)
from visualization.plots import plot_multiple_convergence, plot_convergence
from experiments.statistics import save_stats, save_history


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Comparaison d'algorithmes métaheuristiques")

    # Choix d'algorithme et paramètres
    parser.add_argument("--algo", type=str, default="all", help="Algorithme à exécuter (random, hill, generalized_hill, recuit, all)", choices=["random", "hill", "generalized_hill", "recuit", "all"])
    parser.add_argument("--n_runs", type=int, default=30, help="Nombre de runs Monte Carlo")

    parser.add_argument("--max_iter", type=int, default=1000, help="Nombre d'itérations")

    parser.add_argument("--lambda_penalite", type=float, default=1e5, help="Coefficient de pénalisation")

    # Pour Hill Climbing et Generalized Hill Climbing
    parser.add_argument("--epsilon", type=float, default=1e-6, help="Seuil d'amélioration")

    parser.add_argument("--max_stagnation", type=int, default=50, help="Nombre d'itérations de stagnation")

    # Pour Generalized Hill Climbing
    parser.add_argument("--lambda_voisins", type=int, default=10, help="Nombre de voisins (Generalized HC)")

    # Pour Recuit Simulé
    parser.add_argument("--t0", type=float, default=100, help="Température initiale")

    parser.add_argument("--alpha", type=float, default=0.95, help="Facteur de refroidissement")

    parser.add_argument("--strategy", default="geometric",choices=["linear", "geometric", "logarithmic"], help="Stratégie de refroidissement")

    parser.add_argument("--factor_rechauffement", type=float, default=1.5, help="Facteur de réchauffement")
    
    args = parser.parse_args()
    histories_dict = {}

    if args.algo in ["random", "all"] :
        final_fitness_random, histories_random = monte_carlo_random_search(args.n_runs, args.max_iter, args.lambda_penalite)
        histories_dict["Random Search"] = histories_random

        print("\n=== RANDOM SEARCH ===")
        print("Best:", np.min(final_fitness_random))
        print("Median:", np.median(final_fitness_random))
        
        print("Std:", np.std(final_fitness_random))

        save_stats("Random Search", np.min(final_fitness_random), np.median(final_fitness_random), np.std(final_fitness_random))
        save_history("random", histories_random)

    if args.algo in ["hill", "all"] :
        final_fitness_hill, histories_hill = monte_carlo_hill_climbing(args.n_runs, args.max_iter, args.lambda_penalite, args.epsilon, args.max_stagnation)
        histories_dict["Hill Climbing"] = histories_hill

        print("\n=== HILL CLIMBING ===")
        print("Best:", np.min(final_fitness_hill))
        print("Median:", np.median(final_fitness_hill))
        print("Std:", np.std(final_fitness_hill))
        save_stats("Hill Climbing", np.min(final_fitness_hill), np.median(final_fitness_hill), np.std(final_fitness_hill))
        save_history("hill", histories_hill)

    if args.algo in ["generalized_hill", "all"] :
        final_fitness_generalized_hill, histories_generalized_hill = monte_carlo_generalized_hill_climbing(args.n_runs, args.max_iter, args.lambda_penalite, args.lambda_voisins, args.epsilon, args.max_stagnation)
        histories_dict["Generalized Hill Climbing"] = histories_generalized_hill
        print("\n=== GENERALIZED HILL CLIMBING ===")
        print("Best:", np.min(final_fitness_generalized_hill))
        print("Median:", np.median(final_fitness_generalized_hill))
        print("Std:", np.std(final_fitness_generalized_hill))
        save_stats("Generalized Hill Climbing", np.min(final_fitness_generalized_hill), np.median(final_fitness_generalized_hill), np.std(final_fitness_generalized_hill))
        save_history("generalized_hill", histories_generalized_hill)

    if args.algo in ["recuit", "all"] :
        final_fitness_recuit, histories_recuit = monte_carlo_recuit_simule(args.n_runs, args.max_iter, args.lambda_penalite, args.t0, args.alpha, args.epsilon, args.max_stagnation, args.strategy, args.factor_rechauffement)
        histories_dict["Recuit Simulé"] = histories_recuit
        print("\n=== RECUIT SIMULÉ ===")
        print("Best:", np.min(final_fitness_recuit))
        print("Median:", np.median(final_fitness_recuit))
        print("Std:", np.std(final_fitness_recuit))
        save_stats("Recuit Simulé", np.min(final_fitness_recuit), np.median(final_fitness_recuit), np.std(final_fitness_recuit))
        save_history("recuit", histories_recuit)
    
    if len(histories_dict) > 1:
        plot_multiple_convergence(histories_dict)
    else:
        name = list(histories_dict.keys())[0]
        plot_convergence(histories_dict[name], name)
    

