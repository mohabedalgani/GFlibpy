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
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from anytree.exporter import DotExporter
import pydotplus
plt.ion()

from selection import selection
from calcfitness import calcfitness
from crossover import crossover
from mutation import mutation


def genpop(pop, params, runNumber):
    '''
    Generates the new population of chromosomes from the initial population. Plots the graphs of metrics
    for different generations.

    :param pop: The initial chromosome population
    :param params: Parameters, containing the info about population and about the task we are solving
    :return: -MSE for regression, Accuracy * 100 for classification of the new generated chromosomes
    '''

    print('Calculate Fitness Before Generation ...\n')
    fitness, models, _, _ = calcfitness(pop, params)  # fitness of the initial population
    print(f'Initial Fitness\n')
    print(fitness)
    print('\n')

    # save the best chromosome for the initial population
    bestfit = pop[np.argmax(fitness)]
    bestmodel = models[np.argmax(fitness)]

    # Figure 1: Fitness
    f1 = plt.figure(figsize=(7, 7))
    f1_plot = f1.add_subplot(111)
    f1_plot.set_xlim(0, params['maxGen'])
    f1_plot.set_xticks(np.arange(1, params['maxGen']))
    f1_plot.set_xlabel('generation')
    if params['type'] == 'regress':
        f1_plot.set_ylabel('MSE negative value')
    else:
        f1_plot.set_ylabel('fitness/accuracy %')
        f1_plot.set_ylim(0, 100)
    f1_plot.set_title('Fitness')
    maximum_fit = [np.max(fitness)]  # maximum fitness
    average_fit = [np.mean(fitness)]  # average fitness
    median_fit = [np.median(fitness)]  # median fitness
    std_fit = [np.std(fitness)]  # standard deviation
    overall_maximum_fit = [np.max(maximum_fit)]

    s1 = f'maximum: {maximum_fit[-1]}'
    s2 = f'average: {average_fit[-1]}'
    s3 = f'median: {median_fit[-1]}'
    s4 = f'avg - std: {(np.array(average_fit) - np.array(std_fit))[-1]}'
    s5 = f'avg + std: {(np.array(average_fit) + np.array(std_fit))[-1]}'
    s6 = f'bestsofar: {overall_maximum_fit[-1]}'
    f1_plot.legend([s1, s2, s3, s4, s5, s6])
    f1_plot.plot(maximum_fit, color='b', linewidth=1, marker='o')
    f1_plot.plot(average_fit, color='y', marker='o')
    f1_plot.plot(median_fit, color='g', marker='o')
    f1_plot.plot(np.array(average_fit) - np.array(std_fit), color='r', linestyle=':', marker='o')
    f1_plot.plot(np.array(average_fit) + np.array(std_fit), color='r', linestyle=':', marker='o')
    f1_plot.plot(np.max(maximum_fit), color='b', linewidth=2, marker='o')
    plt.pause(0.01)

    # Figure 2: Structure Complexity
    f2 = plt.figure(figsize=(7, 7))
    f2_plot = f2.add_subplot(111)
    f2_plot.set_xlim(0, params['maxGen'])
    f2_plot.set_xticks(np.arange(1, params['maxGen']))
    f2_plot.set_xlabel('generation')
    f2_plot.set_ylabel('tree depth/size')
    f2_plot.set_title('Structure Complexity')
    maxnum = np.argmax(fitness)

    treedepth, treenodes = [0] * len(pop), [0] * len(pop)
    for i in range(len(pop)):
        treenodes[i] = len(pop[i]['chromNum'])
        treedepth[i] = 1
        childnum = treenodes[i]
        for j in range(treenodes[i] - 1, -1, -1):
            numbers = pop[i]['chromNum'][j].split('.')
            if (numbers[0] != 0) & ((numbers[0] == str(childnum)) | (numbers[1] == str(childnum))):
                treedepth[i] += 1
                childnum = j

    maximum_depth = [np.max(treedepth)]
    old_max_depth = [treedepth[maxnum]]
    max_size = [treenodes[maxnum]]

    s1 = f'maximum depth: {maximum_depth[-1]}'
    s2 = f'bestsofar depth: {old_max_depth[-1]}'
    s3 = f'bestsofar size: {max_size[-1]}'

    f2_plot.legend([s1, s2, s3])
    f2_plot.plot(maximum_depth, color='m', marker='*')
    f2_plot.plot(old_max_depth, color='y', marker='*')
    f2_plot.plot(max_size, color='g', marker='*')
    plt.pause(0.01)

    # Figure 3: Population Diversity

    f3 = plt.figure(figsize=(7, 7))
    f3_plot = f3.add_subplot(111)
    f3_plot.set_xlim(0, params['maxGen'])
    if params['type'] != 'regress':
        f3_plot.set_ylim(0, 100)
    else:
        f3_plot.set_ylim(-1e5, 1e5)
    f3_plot.set_xticks(np.arange(1, params['maxGen']))
    f3_plot.set_xlabel('generation')
    f3_plot.set_ylabel('fitness distribution')
    f3_plot.set_title('Population Diversity')

    xgen = np.zeros_like(fitness)
    f3_plot.plot(xgen, fitness, color = 'b', marker = '.')
    plt.pause(0.01)

    # Figure 4: Accuracy versus Complexity

    f4 = plt.figure(figsize=(7, 7))
    f4_plot = f4.add_subplot(111)
    f4_plot.set_xlim(0, params['maxGen'])
    f4_plot.set_ylim(0, 100)
    f4_plot.set_xticks(np.arange(1, params['maxGen']))
    f4_plot.set_xlabel('generation')
    f4_plot.set_ylabel('fitness, nodes')
    f4_plot.set_title('Accuracy versus Complexity')
    f4_plot.plot(np.arange(len(maximum_fit)), maximum_fit, color='b', marker='.')
    f4_plot.plot(np.arange(len(max_size)), max_size, color='r', marker='.')

    s1 = 'fitness'
    s2 = 'nodes'
    f4_plot.legend([s1, s2])
    plt.pause(0.01)

    # generation
    for i in range(params['maxGen']):
        print(f'Generation {i}\n')
        print(f'Selection in generation {i}\n')
        pop = selection(pop, fitness)

        print(f'Crossover in generation {i}\n')
        # apply crossover to the first half of population
        for j in range(0, int(np.ceil(params['popSize']/2)), 2):
            pop[j], pop[j+1] = crossover(pop[j], pop[j+1], params['crossProb'])

        print(f'Mutation in generation {i}\n')
        # apply mutation to the second part of population
        for j in range(int(np.ceil(params['popSize'] / 2)), params['popSize']):
            pop[j] = mutation(pop[j], params)

        # calculating fitness after new population generated
        print(f'Calculate fitness in generation {i}\n')
        fitness, models, testX, testY = calcfitness(pop, params)

        # save the best chromosome string and chromosome numbers for all time
        if np.max(fitness) > overall_maximum_fit[-1]:
            bestfit = pop[np.argmax(fitness)]
            bestmodel = models[np.argmax(fitness)]

        # Update Figure 1
        maximum_fit.append(np.max(fitness))  # maximum fitness
        average_fit.append(np.mean(fitness))  # average fitness
        median_fit.append(np.median(fitness))  # median itness
        std_fit.append(np.std(fitness))  # standard deviation
        overall_maximum_fit.append(np.max(maximum_fit))

        s1 = f'maximum: {maximum_fit[-1]}'
        s2 = f'average: {average_fit[-1]}'
        s3 = f'median: {median_fit[-1]}'
        s4 = f'avg - std: {(np.array(average_fit) - np.array(std_fit))[-1]}'
        s5 = f'avg + std: {(np.array(average_fit) + np.array(std_fit))[-1]}'
        s6 = f'bestsofar: {overall_maximum_fit[-1]}'

        f1_plot.plot(maximum_fit, color='b', linewidth=1, marker='o')
        f1_plot.plot(average_fit, color='y', marker='o')
        f1_plot.plot(median_fit, color='g', marker='o')
        f1_plot.legend([s1, s2, s3, s4, s5, s6])
        f1_plot.plot(np.array(average_fit) - np.array(std_fit), color='r', linestyle=':', marker='o')
        f1_plot.plot(np.array(average_fit) + np.array(std_fit), color='r', linestyle=':', marker='o')
        f1_plot.plot(overall_maximum_fit, color='b', linewidth=2, marker='o')
        plt.pause(0.01)

        # Update Figure 2
        for j in range(len(pop)):
            treenodes[j] = len(pop[j]['chromNum'])
            treedepth[j] = 1
            childnum = treenodes[j]
            for k in range(treenodes[j] - 1, -1, -1):
                numbers = pop[j]['chromNum'][k].split('.')
                if (numbers[0] != 0) & ((numbers[0] == str(childnum)) | (numbers[1] == str(childnum))):
                    treedepth[j] += 1
                    childnum = k

        maximum_depth.append(np.max(treedepth))
        old_max_depth.append(treedepth[maxnum])
        max_size.append(treenodes[maxnum])

        s1 = f'maximum depth: {maximum_depth[-1]}'
        s2 = f'bestsofar depth: {old_max_depth[-1]}'
        s3 = f'bestsofar size: {max_size[-1]}'

        f2_plot.plot(maximum_depth, color='m', marker='*')
        f2_plot.plot(old_max_depth, color='y', marker='*')
        f2_plot.plot(max_size, color='g', marker='*')
        f2_plot.legend([s1, s2, s3])
        plt.pause(0.01)

        # Update Figure 3
        xgen = (i+1) * np.ones_like(fitness)
        print(xgen)
        print(fitness)
        f3_plot.plot(xgen, fitness, color='b', marker='.')
        plt.pause(0.01)

        # Update Figure 4
        f4_plot.plot(maximum_fit, color='b', marker='.')
        f4_plot.plot(max_size, color='r', marker='.')
        plt.pause(0.01)

    plt.close('all')

    # display results after generation
    if params['type'] != 'regress':
        print(f'Max Fitness(Accuracy) in all generations: {overall_maximum_fit[-1]}\n')
        mse = (100 - overall_maximum_fit[-1]) / 100
    else:
        print(f'Best MSE in all generations: {-overall_maximum_fit[-1]}\n')
        mse = -overall_maximum_fit[-1]
    print('Best chromosome string: \n')
    print(bestfit['chromStr'])

    print('Best chromosome number: \n')
    print(bestfit['chromNum'])

    f1.savefig('images/fitness'+str(runNumber)+'.png')
    f2.savefig('images/complexity' + str(runNumber) + '.png')
    f3.savefig('images/diversity' + str(runNumber) + '.png')
    f4.savefig('images/accVScomp' + str(runNumber) + '.png')

    # show graph
    DotExporter(bestfit['tree'][0]).to_dotfile('images/tree.dot')
    graph = pydotplus.graph_from_dot_file('images/tree.dot')
    graph.write_png("images/tree"+str(runNumber)+'.png')
    img = cv2.imread('images/tree'+str(runNumber)+'.png')

    f5 = plt.figure(figsize=(7, 7))
    f5_plot = f5.add_subplot(111)
    f5_plot.set_title('Best Expression Tree')
    f5_plot.imshow(img)


    if params['type'] == 'binary':
        if len(np.unique(testY)) > 1:
            y_pred_proba = bestmodel.predict_proba(testX)[:, 1]
            fpr, tpr, _ = roc_curve(testY, y_pred_proba)
            auc = roc_auc_score(testY, y_pred_proba)
            plt.plot(fpr, tpr, label="AUC=" + str(auc))
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
            plt.pause(5)
            plt.savefig('images/auc'+str(runNumber)+'.png')
            print(f'ROC AUC Score: {roc_auc_score(testY, y_pred_proba)}')
    plt.close('all')
    return mse







