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


def inipop(params, max_chromosome_length):
    '''
    The function randomly generates the initial population

    :param params: Passed from binary.py, multi.py, regress.py with links
    to the dataset, problem types and initial params for GF algorithm
    :param max_chromosome_length: The maximum length of the chromosome

    :return: returns the list of chromosomes, their positions and the corresponding tree structure
    '''

    print('Initial Population ...\n')
    pop = list()

    if params['type'] == 'binary':
        for i in range(params['popSize']):
            pop.append(dict())
            pop[i]['chromStr'] = list()
            pop[i]['chromNum'] = list()
            pop[i]['chromStr'].append(params['opList'][np.random.randint(1-1, 2)])  # root op must be scalar
            pop[i]['tree'] = list()
            pop[i]['tree'].append(Node(pop[i]['chromStr'][0]))

            opLimit = int((max_chromosome_length - 3) / 4)  # the maximum numbers of operators in the chromosome is opLimit * 2 + 1
            # opLimit = 3
            opCount = 0  # the current number of operators in the chromosome
            count = 0  # guarantees that the last 2 values in chromStr array would be x,y variables
            pos = 0  # the final chromosome length
            len = 0  # used for tree nodes indexing

            while count != -1:
                if pop[i]['chromStr'][pos] != 'x' and pop[i]['chromStr'][pos] != 'y':
                    count += 2
                    opCount += 1
                    pop[i]['chromNum'].append(str(len+1) + '.' + str(len+2))
                    if opCount > opLimit:
                        pop[i]['chromStr'].append(params['opList'][np.random.randint(6-1, 7)])
                        pop[i]['chromStr'].append(params['opList'][np.random.randint(6-1, 7)])
                    else:
                        pop[i]['chromStr'].append(params['opList'][np.random.randint(1-1, 7)])
                        pop[i]['chromStr'].append(params['opList'][np.random.randint(1-1, 7)])
                    pop[i]['tree'].append(Node(pop[i]['chromStr'][len + 1]+ str(len+1), pop[i]['tree'][pos]))
                    pop[i]['tree'].append(Node(pop[i]['chromStr'][len + 2]+ str(len+2), pop[i]['tree'][pos]))
                    len += 2
                else:
                    pop[i]['chromNum'].append('0.' + str(pos))
                pos += 1
                count -= 1
    elif params['type'] == 'multi':
        for i in range(params['popSize']):
            pop.append(dict())
            pop[i]['chromStr'] = list()
            pop[i]['chromNum'] = list()
            pop[i]['chromStr'].append(params['opList'][np.random.randint(1-1, 2)])  # root op must be scalar
            pop[i]['tree'] = list()
            pop[i]['tree'].append(Node(pop[i]['chromStr'][0]))

            opLimit = int((max_chromosome_length - 3) / 4)  # the maximum numbers of operators in the chromosome is opLimit * 2 + 1
            # opLimit = 3
            opCount = 0  # the current number of operators in the chromosome
            count = 0  # guarantees that the last 2 values in chromStr array would be x,y variables
            pos = 0  # the final chromosome length
            len = 0  # used for tree nodes indexing

            while count != -1:
                if pop[i]['chromStr'][pos] != 'x' and pop[i]['chromStr'][pos] != 'y':
                    count += 2
                    opCount += 1
                    pop[i]['chromNum'].append(str(len+1) + '.' + str(len+2))
                    if opCount > opLimit:
                        pop[i]['chromStr'].append(params['opList'][np.random.randint(9-1, 10)])
                        pop[i]['chromStr'].append(params['opList'][np.random.randint(9-1, 10)])
                    else:
                        pop[i]['chromStr'].append(params['opList'][np.random.randint(1-1, 10)])
                        pop[i]['chromStr'].append(params['opList'][np.random.randint(1-1, 10)])
                    pop[i]['tree'].append(Node(pop[i]['chromStr'][len + 1]+str(len+1), pop[i]['tree'][pos]))
                    pop[i]['tree'].append(Node(pop[i]['chromStr'][len + 2]+str(len+2), pop[i]['tree'][pos]))
                    len += 2
                else:
                    pop[i]['chromNum'].append('0.' + str(pos))
                pos += 1
                count -= 1
    elif params['type'] == 'regress':
        for i in range(params['popSize']):
            pop.append(dict())
            pop[i]['chromStr'] = list()
            pop[i]['chromNum'] = list()
            pop[i]['chromStr'].append(params['opList'][np.random.randint(1-1, 2)])  # root op must be scalar
            pop[i]['tree'] = list()
            pop[i]['tree'].append(Node(pop[i]['chromStr'][0]))

            opLimit = int((max_chromosome_length - 3) / 4)  # the maximum numbers of operators in the chromosome is opLimit * 2 + 1
            # opLimit = 3
            opCount = 0  # the current number of operators in the chromosome
            count = 0  # guarantees that the last 2 values in chromStr array would be x,y variables
            pos = 0  # the final chromosome length
            len = 0  # used for tree nodes indexing

            while count != -1:
                if pop[i]['chromStr'][pos] != 'x' and pop[i]['chromStr'][pos] != 'y':
                    count += 2
                    opCount += 1
                    pop[i]['chromNum'].append(str(len+1) + '.' + str(len+2))
                    if opCount > opLimit:
                        pop[i]['chromStr'].append(params['opList'][np.random.randint(9-1, 10)])
                        pop[i]['chromStr'].append(params['opList'][np.random.randint(9-1, 10)])
                    else:
                        pop[i]['chromStr'].append(params['opList'][np.random.randint(1-1, 10)])
                        pop[i]['chromStr'].append(params['opList'][np.random.randint(1-1, 10)])
                    pop[i]['tree'].append(Node(pop[i]['chromStr'][len + 1] + str(len+1), pop[i]['tree'][pos]))
                    pop[i]['tree'].append(Node(pop[i]['chromStr'][len + 2] + str(len+2), pop[i]['tree'][pos]))
                    len += 2
                else:
                    pop[i]['chromNum'].append('0.' + str(pos))
                pos += 1
                count -= 1
    return pop

