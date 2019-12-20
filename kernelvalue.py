'''
   This file is part of GFLIB toolbox
   First Version Sept. 2018

   Cite this project as:
   Mezher M., Abbod M. (2011) Genetic Folding: A New Class of Evolutionary Algorithms.
   In: Bramer M., Petridis M., Hopgood A. (eds) Research and Development in Intelligent Systems XXVII.
   SGAI 2010. Springer, London

   Copyright (C) 20011-2018 Mohd A. Mezher (mohabedalgani@gmail.com)
'''


from Cosine import Cosine
from Log import Log
from Minus_s import Minus_s
from Minus_v import Minus_v
from Multi_s import Multi_s
from Plus_s import Plus_s
from Plus_v import Plus_v
from Sine import Sine
from Tanh import Tanh


def kernelvalue(x1, x2, str, num):
    '''
    Performs a certain operation on column vectors x1, x2 passed from kernel function.
    Calling kernelvalue on all combinations of i, j fills the Gram matrix, a.k.a Kernel Matrix

    :param x1: row i of X passed to SVM fit() method.
    :param x2: row j of X passed to SVM fit() method.
    :param str: string of operations for the current chromosome
    :param num: string of connections between operations for the current chromosome
    :return: Vector or scalar from a certain operation on x1, x2 values
    '''

    value = [0] * len(str)
    for i in range(len(str)-1, -1, -1):
        numbers = num[i].split('.')
        if numbers[0] == '0':
            if str[i] == 'x':
                value[i] = x1
            else:
                value[i] = x2
        else:
            x1 = value[int(numbers[0])]
            x2 = value[int(numbers[1])]
            if str[i] == 'Plus_s':
                value[i] = Plus_s(x1, x2)
            if str[i] == 'Minus_s':
                value[i] = Minus_s(x1, x2)
            if str[i] == 'Multi_s':
                value[i] = Multi_s(x1, x2)
            if str[i] == 'Plus_v':
                value[i] = Plus_v(x1, x2)
            if str[i] == 'Minus_v':
                value[i] = Minus_v(x1, x2)
            if str[i] == 'Sine':
                value.append(Sine(x1, x2))
            if str[i] == 'Cosine':
                value[i] = Cosine(x1, x2)
            if str[i] == 'Sine':
                value[i] = Sine(x1, x2)
            if str[i] == 'Tanh':
                value[i] = Tanh(x1, x2)
            if str[i] == 'Log':
                value[i] = Log(x1, x2)
    return value[0]






