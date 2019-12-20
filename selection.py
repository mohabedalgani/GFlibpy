'''
   This file is part of GFLIB toolbox
   First Version Sept. 2018

   Cite this project as:
   Mezher M., Abbod M. (2011) Genetic Folding: A New Class of Evolutionary Algorithms.
   In: Bramer M., Petridis M., Hopgood A. (eds) Research and Development in Intelligent Systems XXVII.
   SGAI 2010. Springer, London

   Copyright (C) 20011-2018 Mohd A. Mezher (mohabedalgani@gmail.com)
'''

import numpy as np


def selection(orgpop, fitness):
    '''
    Performs selection on the whole population.

    :param orgpop: Population of chromosomes
    :param fitness: Fitness for each chromosome
    :return: Filtered population, where some chromosomes might be replaced with other ones, based
    on their fitness
    '''

    fitness_sum = np.sum(fitness)
    maxnum = np.argmax(fitness)
    newpop = np.zeros_like(orgpop)

    for i in range(len(fitness)):
        if i == maxnum:
            newpop[i] = orgpop[i]
        else:
            value = np.random.rand() * fitness_sum
            if value < fitness[i]:
                newpop[i] = orgpop[i]
            else:
                newpop[i] = orgpop[maxnum]
    return newpop
