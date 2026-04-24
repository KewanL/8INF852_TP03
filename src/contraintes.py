import numpy as np 

# Fonctions d'objectif et de contraintes
def function_f(x1, x2, x3) : 
    return x1**2 * x2 * (2 + x3)

def function_g1(x1, x2, x3) : 
    x = (x2**3 * x3) / (71785 * x1**4) 
    return 1 - x

def function_g2(x1, x2, x3) :
    x = (4 * x2**2 - x1 * x2) / ( 12566 * (x2 * x1 ** 3 - x1 ** 4)) 
    y = 1 / (5108 * x1 **2)
    return x + y - 1

def function_g3(x1, x2, x3) :
    x = (140.45 * x1) / (x2**2 * x3)
    return 1 - x

def function_g4(x1, x2, x3) :
    x = (x1 + x2) / 1.5
    return x - 1

# Fonction de degré de violation de contrainte
def degre_violation(x1, x2, x3) :
    g1 = function_g1(x1, x2, x3)
    g2 = function_g2(x1, x2, x3)
    g3 = function_g3(x1, x2, x3)
    g4 = function_g4(x1, x2, x3)

    violation = 0
    if g1 > 0 : 
        violation += g1
    if g2 > 0 : 
        violation += g2
    if g3 > 0 : 
        violation += g3
    if g4 > 0 : 
        violation += g4

    return violation

# Fonction de pénalisation statique 
def penalisation_stat(x1, x2, x3, lambda_penalite) : 
    f = function_f(x1, x2, x3)
    violation = degre_violation(x1, x2, x3)
    return f + lambda_penalite * violation

# Bornes des variables 
Bornes_minimum = np.array([0.05, 0.25, 2.0])
Bornes_maximum = np.array([2.0, 1.3, 15.0])