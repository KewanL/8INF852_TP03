import matplotlib.pyplot as plt
import numpy as np
import os

FIG_DIR = "results/figures"

def plot_multiple_convergence(histories_dict):
    plt.figure(figsize=(8,5))
    for name, histories in histories_dict.items():
        median = np.median(histories, axis=0)
        q1 = np.percentile(histories, 25, axis=0)
        q3 = np.percentile(histories, 75, axis=0)
        plt.plot(median, label=name)
        plt.fill_between(range(len(median)), q1, q3, alpha=0.2)

    plt.xlabel("Itération")
    plt.ylabel("Fitness")
    plt.yscale("log")
    plt.legend()
    plt.grid()

    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(f"{FIG_DIR}/comparaison.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_convergence(histories, name):
    median = np.median(histories, axis=0)
    q1 = np.percentile(histories, 25, axis=0)
    q3 = np.percentile(histories, 75, axis=0)
    plt.figure(figsize=(8,5))
    plt.plot(median)
    plt.fill_between(range(len(median)), q1, q3, alpha=0.3)
    plt.title(name)
    plt.yscale("log")
    plt.grid()

    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(f"{FIG_DIR}/{name}.png", dpi=300, bbox_inches="tight")
    plt.show()