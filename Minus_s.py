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


def Minus_s(x1, x2):
    if type(x1) == list:
        x1 = np.array(x1)
    if type(x2) == list:
        x2 = np.array(x2)

    if (np.isscalar(x1)) & (np.isscalar(x2)):
        value = np.abs(x1 - x2)
    elif (~np.isscalar(x1)) & (~np.isscalar(x2)):
        value = np.linalg.norm(x1 - x2)
    elif (np.isscalar(x1)) & (~np.isscalar(x2)):
        tmp = np.zeros_like(x2)
        tmp[0] = x1
        value = np.linalg.norm(x2 - tmp)
    else:
        tmp = np.zeros_like(x1)
        tmp[0] = x2
        value = np.linalg.norm(tmp - x1)
    return value
