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


def mutation(orgind, params):
    newind = dict()
    newind['chromNum'] = orgind['chromNum']
    newind['chromStr'] = list()
    length = len(orgind['chromStr'])
    if np.random.rand() < params['mutProb']:
        if orgind['chromStr'][0] == 'Plus_s':
            newind['chromStr'].append('Minus_s')
        else:
            newind['chromStr'].append('Plus_s')
    else:
        newind['chromStr'].append(orgind['chromStr'][0])

    for i in range(1, length):
        if np.random.rand() < params['mutProb']:
            numbers = orgind['chromNum'][i].split('.')
            if numbers[0] != '0':
                newind['chromStr'].append(params['opList'][np.random.randint(1-1, 4)])
            else:
                newind['chromStr'].append(params['opList'][np.random.randint(5-1, 6)])
        else:
            newind['chromStr'].append(orgind['chromStr'][i])
    newind['tree'] = list()
    newind['tree'].append(Node(newind['chromStr'][0]))
    length = len(newind['chromStr'])
    for pos in range(length):
        numbers = newind['chromNum'][pos].split('.')
        if numbers[0] != '0':
            newind['tree'].append(Node(newind['chromStr'][int(numbers[0])]+numbers[0], newind['tree'][pos]))
            newind['tree'].append(Node(newind['chromStr'][int(numbers[1])]+numbers[1], newind['tree'][pos]))
    return newind
