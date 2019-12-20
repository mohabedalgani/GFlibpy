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
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.metrics import accuracy_score, mean_squared_error

from kernel import Kernel

PATH = 'data/'
rnd_seed = 2019
MAX_ITER = 100


def calcfitness(pop, params):
    '''
    Reads the data of type params['type'] and with the data path params['data'],
    then fits the SVC or SVR model depending on the params['type'] with the custom
    kernel, determined by the "pop" parameter. Calculates the resulting metrics for the
    input population.

    :param pop: Population, which will determine the custom kernel for SVM model
    :param params: Parameters, containing the info about population and about the task we are solving
    :return: -MSE for regression task, Accuracy * 100 for the binary and multi classification tasks
    '''

    models = []
    fitness = []
    if params['type'] == 'binary':
        if params['data'] == 'spam.txt':
            with open(PATH + 'binary/' + params['data']) as f:
                M = np.array([])
                file = f.read().split('\n')
                for val in file:
                    tmp = np.array(val.split(','))
                    if len(tmp) <= 1:
                        continue
                    if M.shape[0] == 0:
                        M = tmp
                    else:
                        M = np.vstack([M, tmp])
                tmpX = M[:, :-1]
                tmpY = M[:, -1]
        elif params['data'] == 'german credit.txt':
            with open(PATH + 'binary/' + params['data']) as f:
                M = np.array([])
                file = f.read().split('\n')
                for val in file:
                    tmp = np.array(val.split())
                    if len(tmp) <= 1:
                        continue
                    if M.shape[0] == 0:
                        M = tmp
                    else:
                        M = np.vstack([M, tmp])
                tmpX = M[:, :-1]
                tmpY = M[:, -1]
        elif (params['data'] == 'logic_6_multiplexer.txt') | (params['data'] == 'odd_7_parity.txt') | (params['data'] == 'odd_3_parity.txt'):
            with open(PATH + 'binary/' + params['data']) as f:
                M = np.array([])
                file = f.read().split('\n')
                for val in file:
                    tmp = np.array(val.split(','))
                    if len(tmp) <= 1:
                        continue
                    if M.shape[0] == 0:
                        M = tmp
                    else:
                        M = np.vstack([M, tmp])
                tmpX = M[:, :-1]
                tmpY = M[:, -1]
        elif params['data'] == 'credit approval.txt':
            with open(PATH + 'binary/' + params['data']) as f:
                M = np.array([])
                file = f.read().split('\n')
                for val in file:
                    tmp = np.array(val.split(','))
                    if len(tmp) <= 1:
                        continue
                    if M.shape[0] == 0:
                        M = tmp
                    else:
                        M = np.vstack([M, tmp])
            M = pd.DataFrame(M)
            M = M[~(M == '?').any(axis=1)]
            for col in M.columns:
                try:
                    M.loc[:, col] = M[col].map(float)
                except:
                    M.loc[:, col] = LabelEncoder().fit_transform(M[col])
            M = M.values
            tmpX = M[:, :-1]
            tmpY = M[:, -1]
        else:
            with open(PATH + 'binary/' + params['data']) as f:
                lenConst = 0
                if params['data'] == 'sonar_scale.txt':
                    lenConst = 61
                if params['data'] == 'ionosphere_scale.txt':
                    lenConst = 34
                if params['data'] == 'heart_scale.txt':
                    lenConst = 13
                M = np.array([])
                file = f.read().split('\n')
                for val in file:
                    tmp = np.array(val.split(':'))
                    tmpp = []
                    for t in tmp:
                        tmpp.append(t.split(' ')[0])
                    if len(tmpp) != lenConst:
                        continue
                    if M.shape[0] == 0:
                        M = np.array(tmpp)
                    else:
                        M = np.vstack([M, tmpp])

                tmpX = M[:, 1:]
                tmpY = M[:, 0]
        tmpX = pd.DataFrame(tmpX)

        # For each feature in data tmpX, encode feature with label encoder in case of categorical variable
        for i in range(tmpX.shape[1]):
            try:
                tmpX.iloc[:, i] = tmpX.iloc[:, i].map(float)
            except:
                tmpX.iloc[:, i] = LabelEncoder().fit_transform(tmpX.iloc[:, i])

        tmpY = LabelEncoder().fit_transform(tmpY)  # Transforms the label column (Y) in case it is a categorical feature
        tmpX = tmpX.values
        tmpX = StandardScaler().fit_transform(tmpX)  # Scales the data, so all variables will be in the same range

        trainX, testX, trainY, testY = train_test_split(tmpX, tmpY, train_size = .75, random_state=rnd_seed)
        for i in range(params['popSize']):
            ind = pop[i]  # Population consists of params['popSize'] kernel variations
            k = Kernel(ind)
            svm = SVC(max_iter=MAX_ITER, kernel=k.kernel, probability=True)  # create an SVM model with custom kernel
            svm.fit(trainX, trainY)
            label = svm.predict(testX)
            models.append(svm)
            fitness.append(accuracy_score(testY, label) * 100)

    if params['type'] == 'multi':
        if params['data'] == 'zoo.txt':
            with open(PATH + 'multi/' + params['data']) as f:
                M = np.array([])
                file = f.read().split('\n')
                for val in file:
                    tmp = np.array(val.split(','))
                    if len(tmp) <= 1:
                        continue
                    if M.shape[0] == 0:
                        M = tmp
                    else:
                        M = np.vstack([M, tmp])
                tmpX = M[:, :-1]
                tmpY = M[:, -1]
        else:
            with open(PATH + 'multi/' + params['data']) as f:
                if params['data'] == 'wine_scale.txt':
                    lenConst = 14
                if params['data'] == 'iris_scale.txt':
                    lenConst = 5
                M = np.array([])
                file = f.read().split('\n')
                for val in file:
                    tmp = np.array(val.split(':'))
                    tmpp = []
                    for t in tmp:
                        tmpp.append(t.split(' ')[0])
                    if len(tmpp) != lenConst:
                        continue
                    if M.shape[0] == 0:
                        M = np.array(tmpp)
                    else:
                        M = np.vstack([M, tmpp])

                tmpX = M[:, 1:]
                tmpY = M[:, 0]

        tmpX = pd.DataFrame(tmpX)
        # For each feature in data tmpX, encode feature with label encoder in case of categorical variable
        for i in range(tmpX.shape[1]):
            try:
                tmpX.iloc[:, i] = tmpX.iloc[:, i].map(float)
            except:
                tmpX.iloc[:, i] = LabelEncoder().fit_transform(tmpX.iloc[:, i])
        tmpY = LabelEncoder().fit_transform(tmpY)  # Transforms the label column (Y) in case it is a categorical feature
        tmpX = tmpX.values
        tmpX = StandardScaler().fit_transform(tmpX)  # Scales the data, so all variables will be in the same range

        trainX, testX, trainY, testY = train_test_split(tmpX, tmpY, train_size=.75, random_state=rnd_seed)
        for i in range(params['popSize']):
            ind = pop[i]  # Population consists of params['popSize'] kernel variations
            k = Kernel(ind)
            svm = SVC(max_iter=MAX_ITER, kernel=k.kernel, probability=True)  # create an SVM model with custom kernel
            svm.fit(trainX, trainY)
            label = svm.predict(testX)
            models.append(svm)
            fitness.append(accuracy_score(testY, label) * 100)
    if params['type'] == 'regress':
        with open(PATH + 'regress/'+ params['data']) as f:
            if params['data'] == 'abalone_scale.txt':
                lenConst = 9
            if params['data'] == 'housing_scale.txt':
                lenConst = 14
            if params['data'] == 'mpg_scale.txt':
                lenConst = 8
            M = np.array([])
            file = f.read().split('\n')
            for val in file:
                tmp = np.array(val.split(':'))
                tmpp = []
                for t in tmp:
                    tmpp.append(t.split(' ')[0])
                if len(tmpp) != lenConst:
                    continue
                if M.shape[0] == 0:
                    M = np.array(tmpp)
                else:
                    M = np.vstack([M, tmpp])

            tmpX = M[:, 1:]
            tmpY = M[:, 0]

        tmpX = pd.DataFrame(tmpX)
        # For each feature in data tmpX, encode feature with label encoder in case of categorical variable
        for i in range(tmpX.shape[1]):
            try:
                tmpX.iloc[:, i] = tmpX.iloc[:, i].map(float)
            except:
                tmpX.iloc[:, i] = LabelEncoder().fit_transform(tmpX.iloc[:, i])
        tmpY = LabelEncoder().fit_transform(tmpY)  # Transforms the label column (Y) in case it is a categorical feature
        tmpX = tmpX.values
        tmpX = StandardScaler().fit_transform(tmpX)  # Scales the data, so all variables will be in the same range

        trainX, testX, trainY, testY = train_test_split(tmpX, tmpY, train_size=.75, random_state=rnd_seed)
        for i in range(params['popSize']):
            ind = pop[i]   # Population consists of params['popSize'] kernel variations
            k = Kernel(ind)
            svm = SVR(max_iter=2*MAX_ITER, kernel=k.kernel)  # create an SVM model with custom kernel
            svm.fit(trainX, trainY)
            label = svm.predict(testX)
            models.append(svm)
            fitness.append(-mean_squared_error(testY, label))
    return fitness, models, testX, testY
