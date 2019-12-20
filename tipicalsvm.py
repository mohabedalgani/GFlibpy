'''
   This file is part of GFLIB toolbox
   First Version Sept. 2018

   Cite this project as:
   Mezher M., Abbod M. (2011) Genetic Folding: A New Class of Evolutionary Algorithms.
   In: Bramer M., Petridis M., Hopgood A. (eds) Research and Development in Intelligent Systems XXVII.
   SGAI 2010. Springer, London

   Copyright (C) 20011-2018 Mohd A. Mezher (mohabedalgani@gmail.com)
'''

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error

PATH = 'data/'

def typicalsvm(params):
    '''
    Loads the dataset specified in the params dictionary, applies preprocessing and scaling and fits
    the SVM with the kernel type specified in the params
    :param params: Parameters including kernel types, number of crossvalidation splits, current dataset path and problem type.
    :return: Metrics measured on the test set with the current SVM model
    '''

    if params['type'] == 'binary':
        if params['data'] == 'spam.txt':
            with open(PATH + 'binary/'+ params['data']) as f:
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
            with open(PATH + 'binary/'+ params['data']) as f:
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
            with open(PATH + 'binary/'+ params['data']) as f:
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
            with open(PATH + 'binary/'+ params['data']) as f:
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
            with open(PATH + 'binary/'+ params['data']) as f:
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
        for i in range(tmpX.shape[1]):
            try:
                tmpX.iloc[:, i] = tmpX.iloc[:, i].map(float)
            except:
                tmpX.iloc[:, i] = LabelEncoder().fit_transform(tmpX.iloc[:, i])
        tmpY = LabelEncoder().fit_transform(tmpY)
        tmpX = tmpX.values
        tmpX = StandardScaler().fit_transform(tmpX)

        trainX, testX, trainY, testY = train_test_split(tmpX, tmpY, train_size = .75)
        model = SVC(kernel=params['kernel'])
        model.fit(trainX, trainY)
        # calculate classification loss
        label = model.predict(testX)
        L = np.sum(label != testY) / len(testY)
        print(f'The value of loss {L}\n')
        fitness = accuracy_score(testY, label) * 100
        mse = (100 - fitness) / 100
        print(f'Accuracy : {fitness}\n')

    if params['type'] == 'multi':
        if params['data'] == 'zoo.txt':
            with open(PATH + 'multi/'+ params['data']) as f:
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
            with open(PATH + 'multi/'+ params['data']) as f:
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
        for i in range(tmpX.shape[1]):
            try:
                tmpX.iloc[:, i] = tmpX.iloc[:, i].map(float)
            except:
                tmpX.iloc[:, i] = LabelEncoder().fit_transform(tmpX.iloc[:, i])
        tmpY = LabelEncoder().fit_transform(tmpY)
        tmpX = tmpX.values
        tmpX = StandardScaler().fit_transform(tmpX)

        trainX, testX, trainY, testY = train_test_split(tmpX, tmpY, train_size=.75)
        model = SVC(kernel=params['kernel'])
        model.fit(trainX, trainY)
        # calculate classification loss
        label = model.predict(testX)
        L = np.sum(label != testY) / len(testY)
        print(f'The value of loss {L}\n')
        fitness = accuracy_score(testY, label) * 100
        mse = (100 - fitness) / 100
        print(f'Accuracy : {fitness}\n')
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
        for i in range(tmpX.shape[1]):
            try:
                tmpX.iloc[:, i] = tmpX.iloc[:, i].map(float)
            except:
                tmpX.iloc[:, i] = LabelEncoder().fit_transform(tmpX.iloc[:, i])

        tmpY = LabelEncoder().fit_transform(tmpY)
        tmpX = tmpX.values
        tmpX = StandardScaler().fit_transform(tmpX)

        trainX, testX, trainY, testY = train_test_split(tmpX, tmpY, train_size=.75)
        model = SVR(kernel=params['kernel'])
        model.fit(trainX, trainY)
        label = model.predict(testX)
        fitness = -mean_squared_error(testY, label)
        mse = -fitness
        print(f'MSE : {fitness}\n')
    return mse
