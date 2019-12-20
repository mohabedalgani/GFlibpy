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


def Tanh(x1, x2):
    if (np.isscalar(x1)) & (np.isscalar(x2)):
        value = np.power(np.tanh(x1) * np.tanh(x2), 2)
    elif (~np.isscalar(x1)) & (~np.isscalar(x2)):
        value = np.sum(np.dot(np.tanh(x1), np.tanh(x2)))
    elif (np.isscalar(x1)) & (~np.isscalar(x2)):
        tmp = list(x2)
        tmp.append(x1)
        value = np.sum(np.power(np.tanh(tmp), 2))
    else:
        tmp = list(x1)
        tmp.append(x2)
        value = np.sum(np.power(np.tanh(tmp), 2))
    return value
