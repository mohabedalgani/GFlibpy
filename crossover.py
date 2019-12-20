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
from anytree import Node


def crossover(parent1, parent2, crossprob):
    '''
    Applies the crossover operation to parent1 and parent2 to produce 2 new child chromosomes.

    :param parent1:
    :param parent2:
    :param crossprob: probability of crossover operation
    :return: 2 children after crossover operation, or parent1, parent2 if the operation didn't happen
    '''

    tmp1, tmp2 = dict(), dict()
    if np.random.rand() < crossprob:
        tmp1['chromStr'], tmp1['chromNum'] = list(), list()
        tmp2['chromStr'], tmp2['chromNum'] = list(), list()
        length = min([len(parent1['chromStr']), len(parent2['chromStr'])])
        maxcrosspoint = 1
        for i in range(1, length):
            if parent1['chromNum'][i] != parent2['chromNum'][i]:
                maxcrosspoint = i
                break
        crosspoint = np.random.randint(1, maxcrosspoint + 1)
        for i in range(len(parent2['chromStr'])):
            if i < crosspoint:
                tmp1['chromStr'].append(parent1['chromStr'][i])
                tmp1['chromNum'].append(parent1['chromNum'][i])
            else:
                tmp1['chromStr'].append(parent2['chromStr'][i])
                tmp1['chromNum'].append(parent2['chromNum'][i])

        for i in range(len(parent1['chromStr'])):
            if i < crosspoint:
                tmp2['chromStr'].append(parent2['chromStr'][i])
                tmp2['chromNum'].append(parent2['chromNum'][i])
            else:
                tmp2['chromStr'].append(parent1['chromStr'][i])
                tmp2['chromNum'].append(parent1['chromNum'][i])

        tmp1['tree'] = [Node(tmp1['chromStr'][0])]
        for pos in range(len(tmp1['chromStr'])):
            numbers = tmp1['chromNum'][pos].split('.')
            if numbers[0] != '0':
                tmp1['tree'].append(Node(tmp1['chromStr'][int(numbers[0])]+numbers[0], parent=tmp1['tree'][pos]))
                tmp1['tree'].append(Node(tmp1['chromStr'][int(numbers[1])]+numbers[1], parent=tmp1['tree'][pos]))

        tmp2['tree'] = [Node(tmp2['chromStr'][0])]
        for pos in range(len(tmp2['chromStr'])):
            numbers = tmp2['chromNum'][pos].split('.')
            if numbers[0] != '0':
                tmp2['tree'].append(Node(tmp2['chromStr'][int(numbers[0])]+numbers[0], parent=tmp2['tree'][pos]))
                tmp2['tree'].append(Node(tmp2['chromStr'][int(numbers[1])]+numbers[1], parent=tmp2['tree'][pos]))
    else:
        tmp1 = parent1
        tmp2 = parent2
    return tmp1, tmp2

