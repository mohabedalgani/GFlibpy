'''
   This file is part of GFLIB toolbox
   First Version Sept. 2018

   Cite this project as:
   Mezher M., Abbod M. (2011) Genetic Folding: A New Class of Evolutionary Algorithms.
   In: Bramer M., Petridis M., Hopgood A. (eds) Research and Development in Intelligent Systems XXVII.
   SGAI 2010. Springer, London

   Copyright (C) 20011-2018 Mohd A. Mezher (mohabedalgani@gmail.com)
'''

# This script generates logic synthesis data
# 6-multiplexer, odd-3-parity, odd-5-parity

import numpy as np

# 6-multiplexer
inputs = np.random.randint(0, 2, 64*6).reshape((64,6))
outputs = []
for i in range(len(inputs)):
    if (inputs[i, 0] == 0) & (inputs[i, 1] == 0):
        outputs.append(inputs[i, 2])
    elif (inputs[i, 0] == 0) & (inputs[i, 1] == 1):
        outputs.append(inputs[i, 3])
    elif (inputs[i, 0] == 1) & (inputs[i, 1] == 0):
        outputs.append(inputs[i, 4])
    elif (inputs[i, 0] == 1) & (inputs[i, 1] == 1):
        outputs.append(inputs[i, 5])
outputs = np.array(outputs).reshape(-1,1)
data = np.hstack([inputs, outputs])
np.savetxt('data/binary/logic_6_multiplexer.txt', data, delimiter=',', fmt='%i')

# odd-3-parity
data = np.empty((8,4))
for i in range(0, 8):
    b = np.binary_repr(i, 3)
    n = 0
    for j in range(3):
        if b[j] == '1':
            n += 1
    data[i, :-1] = list(b)
    if (n == 1) | (n == 3):
        data[i, -1] = 1
    else:
        data[i, -1] = 0
np.savetxt('data/binary/odd_3_parity.txt', data, delimiter=',', fmt='%i')

# odd-7-parity
data = np.empty((127,8))
for i in range(0, 127):
    b = np.binary_repr(i, 7)
    n = 0
    for j in range(7):
        if b[j] == '1':
            n += 1
    data[i, :-1] = list(b)
    if (n == 1) | (n == 3) | (n == 5) | (n == 7):
        data[i, -1] = 1
    else:
        data[i, -1] = 0
np.savetxt('data/binary/odd_7_parity.txt', data, delimiter=',', fmt='%i')
