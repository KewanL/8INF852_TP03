import numpy as np
from src.contraintes import Bornes_minimum, Bornes_maximum

# Fonction pour générer une solution aléatoire dans les bornes
def generer_solution_aleatoire() :
    x1 = np.random.uniform(Bornes_minimum[0], Bornes_maximum[0])
    x2 = np.random.uniform(Bornes_minimum[1], Bornes_maximum[1])
    x3 = np.random.uniform(Bornes_minimum[2], Bornes_maximum[2])
    return x1, x2, x3