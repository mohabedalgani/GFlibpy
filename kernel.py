'''
   This file is part of GFLIB toolbox
   First Version Sept. 2018

   Cite this project as:
   Mezher M., Abbod M. (2011) Genetic Folding: A New Class of Evolutionary Algorithms.
   In: Bramer M., Petridis M., Hopgood A. (eds) Research and Development in Intelligent Systems XXVII.
   SGAI 2010. Springer, London

   Copyright (C) 20011-2018 Mohd A. Mezher (mohabedalgani@gmail.com)
'''

from kernelvalue import kernelvalue
import numpy as np


class Kernel:
    def __init__(self, ind):
        self.ind = ind

    def kernel(self, U, V):
        '''
        Used as a custom function to pass into SVM kernel parameter
        Refer to https://scikit-learn.org/stable/auto_examples/svm/plot_custom_kernel.html
        :param U: full X data passed to SVM fit
        :param V: full y target data passed to SVM fit
        :return: The Gram matrix a.k.a. Kernel Matrix (often abbreviated as K).
        '''

        m = U.shape[0]
        n = V.shape[0]
        G = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                G[i, j] = kernelvalue(U[i, :], V[j, :], self.ind['chromStr'], self.ind['chromNum'])
        return G
