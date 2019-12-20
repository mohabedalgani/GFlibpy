'''
   This file is part of GFLIB toolbox
   First Version Sept. 2018

   Cite this project as:
   Mezher M., Abbod M. (2011) Genetic Folding: A New Class of Evolutionary Algorithms.
   In: Bramer M., Petridis M., Hopgood A. (eds) Research and Development in Intelligent Systems XXVII.
   SGAI 2010. Springer, London

   Copyright (C) 20011-2018 Mohd A. Mezher (mohabedalgani@gmail.com)
'''

from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# create folder for graphs generated during the run
if not os.path.exists('images/'):
    os.makedirs('images/')
else:
    files = glob.glob('images/*')
    for f in files:
        os.remove(f)

from inipop import inipop
from genpop import genpop
from tipicalsvm import typicalsvm

filterwarnings('ignore')
print('Running multi classification ...\n\n')

print('Type the maximum length of the chromosome: ')
max_chromosome_length = int(input())  # the maximum total length of the chromosome

DATA_PATH = 'data/multi/'  # Dataset path for binary classification

params = dict()
params['type'] = 'multi'  # problem type
params['data'] = 'iris_scale.txt'  # path to data file
params['kernel'] = 'rbf'  # rbf,linear,polynomial,gf
params['mutProb'] = 0.1  # mutation probability
params['crossProb'] = 0.5  # crossover probability
params['maxGen'] = 25  # max generation
params['popSize'] = 50  # population size
params['crossVal'] = 5  # number of cross validation slits
params['opList'] = ['Plus_s', 'Minus_s', 'Plus_v', 'Minus_v',
                    'Sine', 'Cosine', 'Tanh', 'Log', 'x', 'y']  # Operators and operands

print(f'''Data Set : {DATA_PATH + params['data']}\n\n''')
kernels = ['poly', 'rbf', 'linear', 'gf']
totalMSE = dict()
for ker in kernels:
    totalMSE[ker] = list()

for i in range(5):
    temp = []
    for index, kernel in enumerate(kernels):
        params['kernel'] = kernel
        print(f'''SVM Kernel : {params['kernel']} \n''')
        if kernel == 'gf':
            print(f'''Max Generation : {params['maxGen']}\n''')
            print(f'''Population Size : {params['popSize']}\n''')
            print(f'''CrossOver Probability : {params['crossProb']}\n''')
            print(f'''Mutation Probability : {params['mutProb']}\n\n''')
            pop = inipop(params, max_chromosome_length)
            mse = genpop(pop, params, i)
        else:
            mse = typicalsvm(params)
        totalMSE[kernel].append(mse)
        print('\n')

# Boxplot of errors for each kernel
plt.boxplot([totalMSE['poly'], totalMSE['rbf'], totalMSE['linear'], totalMSE['gf']])
plt.xticks(np.arange(1,5), kernels)
plt.title('MSE for each svm kernel')
plt.xlabel('SVM kernel')
plt.ylabel('Test Error Rate')
plt.ioff()
plt.savefig('images/mse.png')
plt.show()

