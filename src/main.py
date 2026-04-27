import argparse
import numpy as np

# Import des algorithmes
from algorithms import monte_carlo_pso, monte_carlo_de

# Import des fonctions d'affichage et de sauvegarde
from visualization.plots import plot_multiple_convergence, plot_convergence
from experiments.statistics import save_stats, save_history


if __name__ == "__main__":

    # Parsing des arguments
    parser = argparse.ArgumentParser(description="Comparaison PSO et DE sur le problème du ressort")

    # Choix algorithme
    parser.add_argument("--algo", type=str, default="all", choices=["pso", "de", "all", "pso_compare_swarm", "pso_compare_w", "de_compare_pop", "de_compare_F", "de_compare_CR"], help="Algorithme à exécuter")

    # Paramètres généraux
    parser.add_argument("--n_runs", type=int, default=30, help="Nombre de runs Monte Carlo")
    parser.add_argument("--max_iter", type=int, default=1000, help="Nombre maximum d'itérations")
    parser.add_argument("--lambda_penalite", type=float, default=1e5, help="Coefficient de pénalisation")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="Seuil minimal d'amélioration")
    parser.add_argument("--max_stagnation", type=int, default=50, help="Nombre max d'itérations sans amélioration")

    # Paramètres PSO
    parser.add_argument("--swarm_size", type=int, default=30, help="Nombre de particules dans l'essaim")
    parser.add_argument("--w", type=float, default=0.7, help="Poids d'inertie")
    parser.add_argument("--c1", type=float, default=1.5, help="Attraction vers le meilleur personnel")
    parser.add_argument("--c2", type=float, default=1.5, help="Attraction vers le meilleur global")

    # Paramètres DE
    parser.add_argument("--pop_size", type=int, default=30, help="Taille de la population")
    parser.add_argument("--F", type=float, default=0.8, help="Facteur de mutation différentielle")
    parser.add_argument("--CR", type=float, default=0.9, help="Taux de crossover")

    args = parser.parse_args()

    # Dictionnaire pour stocker les historiques
    histories_dict = {}

    if args.algo in ["pso", "all"]:

        final_fitness_pso, histories_pso, violations_pso = monte_carlo_pso(args.n_runs, args.max_iter, args.lambda_penalite, args.swarm_size, args.w, args.c1, args.c2, args.epsilon, args.max_stagnation)

        histories_dict["PSO"] = histories_pso

        best = np.min(final_fitness_pso)
        worst = np.max(final_fitness_pso) 
        median = np.median(final_fitness_pso)
        std = np.std(final_fitness_pso)
        feasible_rate = np.mean(violations_pso == 0)

        print("\n=== PSO ===")
        print("Best :", best)
        print("Worst :", worst)
        print("Median :", median)
        print("Std :", std)
        print("Feasible rate :", feasible_rate)

        save_stats("PSO", best, median, std, feasible_rate, worst=worst)
        save_history("pso", histories_pso)

    if args.algo in ["de", "all"]:

        final_fitness_de, histories_de, violations_de = monte_carlo_de(args.n_runs, args.max_iter, args.lambda_penalite, args.pop_size, args.F, args.CR, args.epsilon, args.max_stagnation)

        histories_dict["DE"] = histories_de

        best = np.min(final_fitness_de)
        worst = np.max(final_fitness_de)
        median = np.median(final_fitness_de)
        std = np.std(final_fitness_de)
        feasible_rate = np.mean(violations_de == 0)

        print("\n=== DE ===")
        print("Best :", best)
        print("Worst :", worst)
        print("Median :", median)
        print("Std :", std)
        print("Feasible rate :", feasible_rate)

        save_stats("DE", best, median, std, feasible_rate, worst=worst)
        save_history("de", histories_de)

    if args.algo == "pso_compare_swarm":

        for swarm in [10, 30, 50, 100]:
            final_fitness, histories, violations = monte_carlo_pso(args.n_runs, args.max_iter, args.lambda_penalite, swarm, args.w, args.c1, args.c2, args.epsilon, args.max_stagnation)

            name = f"PSO swarm={swarm}"
            histories_dict[name] = histories

            best = np.min(final_fitness)
            worst = np.max(final_fitness)
            median = np.median(final_fitness)
            std = np.std(final_fitness)
            feasible_rate = np.mean(violations == 0)

            save_stats(name, best, median, std, feasible_rate, worst=worst)

            print(f"\n=== {name} ===")
            print("Best :", best)
            print("Worst :", worst)
            print("Median :", median)
            print("Std :", std)
            print("Feasible rate :", feasible_rate)

    if args.algo == "pso_compare_w":

        for inertia in [0.2, 0.5, 0.7, 0.9]:
            final_fitness, histories, violations = monte_carlo_pso(args.n_runs, args.max_iter, args.lambda_penalite, args.swarm_size, inertia, args.c1, args.c2, args.epsilon, args.max_stagnation)

            name = f"PSO w={inertia}"
            histories_dict[name] = histories

            best = np.min(final_fitness)
            worst = np.max(final_fitness)
            median = np.median(final_fitness)
            std = np.std(final_fitness)
            feasible_rate = np.mean(violations == 0)

            save_stats(name, best, median, std, feasible_rate, worst=worst)
            print(f"\n=== {name} ===")
            print("Best :", best)
            print("Worst :", worst)
            print("Median :", median)
            print("Std :", std)
            print("Feasible rate :", feasible_rate)

    if args.algo == "de_compare_pop":

        for pop in [10, 30, 50, 100]:
            final_fitness, histories, violations = monte_carlo_de(args.n_runs, args.max_iter, args.lambda_penalite, pop, args.F, args.CR, args.epsilon, args.max_stagnation)

            name = f"DE pop={pop}"
            histories_dict[name] = histories

            best = np.min(final_fitness)
            worst = np.max(final_fitness)
            median = np.median(final_fitness)
            std = np.std(final_fitness)
            feasible_rate = np.mean(violations == 0)

            save_stats(name, best, median, std, feasible_rate, worst=worst)
            print(f"\n=== {name} ===")
            print("Best :", best)
            print("Worst :", worst)
            print("Median :", median)
            print("Std :", std)
            print("Feasible rate :", feasible_rate)


    if args.algo == "de_compare_F":

        for facteur in [0.3, 0.5, 0.8, 1.0]:
            final_fitness, histories, violations = monte_carlo_de(args.n_runs, args.max_iter, args.lambda_penalite, args.pop_size, facteur, args.CR, args.epsilon, args.max_stagnation)

            name = f"DE F={facteur}"
            histories_dict[name] = histories

            best = np.min(final_fitness)
            worst = np.max(final_fitness)
            median = np.median(final_fitness)
            std = np.std(final_fitness)
            feasible_rate = np.mean(violations == 0)

            save_stats(name, best, median, std, feasible_rate, worst=worst)
            print(f"\n=== {name} ===")
            print("Best :", best)
            print("Worst :", worst)
            print("Median :", median)
            print("Std :", std)
            print("Feasible rate :", feasible_rate)

    if args.algo == "de_compare_CR":

        for crossover in [0.3, 0.5, 0.8, 1.0]:
            final_fitness, histories, violations = monte_carlo_de(args.n_runs, args.max_iter, args.lambda_penalite, args.pop_size, args.F, crossover, args.epsilon, args.max_stagnation)

            name = f"DE CR={crossover}"
            histories_dict[name] = histories

            best = np.min(final_fitness)
            worst = np.max(final_fitness)
            median = np.median(final_fitness)
            std = np.std(final_fitness)
            feasible_rate = np.mean(violations == 0)

            save_stats(name, best, median, std, feasible_rate, worst=worst)
            print(f"\n=== {name} ===")
            print("Best :", best)
            print("Worst :", worst)
            print("Median :", median)
            print("Std :", std)
            print("Feasible rate :", feasible_rate)

    if len(histories_dict) > 1:
        plot_multiple_convergence(histories_dict)
    elif len(histories_dict) == 1:
        name = list(histories_dict.keys())[0]
        plot_convergence(histories_dict[name], name)