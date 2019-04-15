"""
@author: Aiping
"""

import pandas as pd
import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00', '#000000']
plt.rcParams.update({'font.size': 12})


# import data

# infile = './Hard Q-Learning L0.9 q-100.0 E0.1.csv'
# infile = '../Solution2/@World@ Q-Learning L@L@ q@q@ E@E@.csv'
infile = './@World@ Q-Learning L@L@ q@q@ E@E@.csv'

worlds = ['Easy','Hard']
lr = [0.1,0.9]  # learning rate
qInit = [-100.0,0.0,100.0]
epsilon = [0.1,0.3,0.5]
columns = ['iter','time','reward','steps','convergence']
data_list = pd.DataFrame(columns=columns)


for w in worlds:
    for l in lr:
        for q in qInit:
            for e in epsilon:
                fname = infile.replace('@World@', w).replace('@L@', str(l)).replace('@q@', str(q)).replace('@E@', str(e))
                data = pd.read_csv(fname, sep=',', header='infer')
                data['world'] = w
                data['learning_rate'] = l
                data['gInit'] = q
                data['epsilon'] = e
                data_list = pd.concat([data_list, data], ignore_index=True)


worlds = ['Easy','Hard']

##########
# learning rate
##########

for w in worlds:
    for l in [0.1, 0.9]:
        for q in [0.0]:
            for e in [0.1]:
                plt.plot(data_list[(data_list['world'] == w) &
                                   (data_list['learning_rate'] == float(l)) &
                                   (data_list['gInit'] == float(q)) &
                                   (data_list['epsilon'] == float(e))]['iter'],
                         data_list[(data_list['world'] == w) &
                                   (data_list['learning_rate'] == float(l)) &
                                   (data_list['gInit'] == float(q)) &
                                   (data_list['epsilon'] == float(e))]['reward'],
                         '.-', label='learning_rate_' + str(l) + '_gInit_' + str(q) + '_epsilon_' + str(e))

    plt.rc('axes', prop_cycle=(cycler('color', colors)))
    plt.xlabel('iteration')
    plt.ylabel('reward')
    plt.ylim(-800, 100)
    plt.grid(True)
    plt.legend(loc="best")
    plt.title('{}--{}'.format(w, ' Q-Learning'))
    plt.show()

##########
# q-initialization
##########

for w in worlds:
    for l in [0.1]:
        for q in [-100.0, 0.0, 100.0]:
            for e in [0.1]:
                plt.plot(data_list[(data_list['world'] == w) &
                                   (data_list['learning_rate'] == float(l)) &
                                   (data_list['gInit'] == float(q)) &
                                   (data_list['epsilon'] == float(e))]['iter'],
                         data_list[(data_list['world'] == w) &
                                   (data_list['learning_rate'] == float(l)) &
                                   (data_list['gInit'] == float(q)) &
                                   (data_list['epsilon'] == float(e))]['reward'],
                         '.-', label='learning_rate_' + str(l) + '_gInit_' + str(q) + '_epsilon_' + str(e))

    plt.rc('axes', prop_cycle=(cycler('color', colors)))
    plt.xlabel('iteration')
    plt.ylabel('reward')
    plt.ylim(-800, 100)
    plt.grid(True)
    plt.legend(loc="best")
    plt.title('{}--{}'.format(w, ' Q-Learning'))
    plt.show()



##########
# epsilon
##########

for w in worlds:
    for l in [0.1]:
        for q in [0.0]:
            for e in [0.1,0.3,0.5]:
                plt.plot(data_list[(data_list['world'] == w) &
                                   (data_list['learning_rate'] == float(l)) &
                                   (data_list['gInit'] == float(q)) &
                                   (data_list['epsilon'] == float(e))]['iter'],
                         data_list[(data_list['world'] == w) &
                                   (data_list['learning_rate'] == float(l)) &
                                   (data_list['gInit'] == float(q)) &
                                   (data_list['epsilon'] == float(e))]['reward'],
                         '.-', label='learning_rate_' + str(l) + '_gInit_' + str(q) + '_epsilon_' + str(e))

    plt.rc('axes', prop_cycle=(cycler('color', colors)))
    plt.xlabel('iteration')
    plt.ylabel('reward')
    plt.ylim(-800, 100)
    plt.grid(True)
    plt.legend(loc="best")
    plt.title('{}--{}'.format(w, ' Q-Learning'))
    plt.show()





##########
# compare value iteration and policy iteration
##########

# input data
infile = './@World@ @Iteration@.csv'
worlds = ['Easy','Hard']
iterations = ['Value','Policy']

columns = ['iter','time','reward','steps','convergence']
data_list = pd.DataFrame(columns=columns)


for w in worlds:
    for i in iterations:
        fname = infile.replace('@World@', w).replace('@Iteration@', i)
        data = pd.read_csv(fname, sep=',', header='infer')
        data['world'] = w
        data['iteration'] = i
        data_list = pd.concat([data_list, data], ignore_index=True)


# add Q reinforcement, if only compare two policies, comment out this section
infile = './@World@ Q-Learning L@L@ q@q@ E@E@.csv'

lr = [0.1]  # learning rate
qInit = [0.0]
epsilon = [0.1]

iterations = ['Value','Policy', 'Q-learning']  # if compare three

for w in worlds:
    for l in lr:
        for q in qInit:
            for e in epsilon:
                fname = infile.replace('@World@', w).replace('@L@', str(l)).replace('@q@', str(q)).replace('@E@', str(e))
                data = pd.read_csv(fname, sep=',', header='infer')
                data['world'] = w
                data['iteration'] = 'Q-learning'
                data_list = pd.concat([data_list, data], ignore_index=True)


# plot
for y in ['reward', 'time', 'steps']:
    for w in worlds:
        for i in iterations:
            plt.semilogx(data_list[(data_list['world'] == w) &
                               (data_list['iteration'] == i)]['iter'],
                     data_list[(data_list['world'] == w) &
                               (data_list['iteration'] == i)][y],
                     '.-', label=str(w) + '_world_' + str(i) + '_iteration_')

        plt.rc('axes', prop_cycle=(cycler('color', colors)))
        plt.xlabel('iteration')
        plt.ylabel(y)
        plt.grid(True)
        plt.legend(loc="best")
        plt.title('{}--{}--{}'.format(w, 'Iteration', y))
        plt.show()

for w in worlds:
    for i in iterations:
        plt.plot(data_list[(data_list['world'] == w) &
                           (data_list['iteration'] == i)]['time'],
                 data_list[(data_list['world'] == w) &
                           (data_list['iteration'] == i)]['reward'],
                 '.-', label=str(w) + '_world_' + str(i) + '_iteration_')

    plt.rc('axes', prop_cycle=(cycler('color', colors)))
    plt.xlabel('time')
    plt.ylabel('reward')
    plt.grid(True)
    plt.legend(loc="best")
    plt.title('{}--{}--{}'.format(w, 'reward', 'time'))
    plt.show()


# ##########
# # vary size
# ##########

# input data
infile = './size @Iteration@.csv'
iterations = ['Value','Policy']
columns = ['iter','time','reward','steps','shape']
data_list = pd.DataFrame(columns=columns)


for i in iterations:
    fname = infile.replace('@Iteration@', i)
    data = pd.read_csv(fname, sep=',', header='infer')
    data['iteration'] = i
    data_list = pd.concat([data_list, data], ignore_index=True)

# add Q

infile = './size Q-Learning L0.1 E@e@.csv'
epsilon = [0.1]

for e in epsilon:
    fname = infile.replace('@e@', str(e))
    data = pd.read_csv(fname, sep=',', header='infer')
    data['iteration'] = 'Q-Learning '+ str(e)
    data_list = pd.concat([data_list, data], ignore_index=True)
iterations = ['Value','Policy','Q-Learning 0.1']

# plot

# time vs states
for i in iterations:
    plt.plot(data_list[(data_list['iteration'] == i)]['shape']**2,
             data_list[(data_list['iteration'] == i)]['time'],
             '.-', label=str(i) + '_iteration_')

plt.rc('axes', prop_cycle=(cycler('color', colors)))
plt.xlabel('states')
plt.ylabel('time /seconds')
plt.grid(True)
plt.legend(loc="best")
plt.title('{}--{}'.format('States', 'time'))
plt.show()

# rewards vs states
for i in iterations:
    plt.plot(data_list[(data_list['iteration'] == i)]['shape']**2,
             data_list[(data_list['iteration'] == i)]['reward'],
             '.-', label=str(i) + '_iteration_')

plt.rc('axes', prop_cycle=(cycler('color', colors)))
plt.xlabel('states')
plt.ylabel('reward')
plt.grid(True)
plt.legend(loc="best")
plt.title('{}--{}'.format('States', 'reward'))
plt.show()

# iterations vs states
for i in iterations:
    plt.plot(data_list[(data_list['iteration'] == i)]['shape']**2,
             data_list[(data_list['iteration'] == i)]['iter'],
             '.-', label=str(i) + '_iteration_')

plt.rc('axes', prop_cycle=(cycler('color', colors)))
plt.xlabel('states')
plt.ylabel('iterations to converge')
plt.grid(True)
plt.legend(loc="best")
plt.title('{}--{}'.format('States', 'iterations'))
plt.show()



###########
# learning rate alpha with decay
###########

# # import data
# infile = '../Solution2/@World@ Q-Learning L@L@ q@q@ E@E@.csv'
#
# worlds = ['Easy','Hard']
# lr = [0.1] # learning rate
# qInit = [0.0]
# epsilon = [0.1]
# columns = ['iter','time','reward','steps','convergence']
# data_list_2 = pd.DataFrame(columns=columns)
#
# for w in worlds:
#     for l in lr:
#         for q in qInit:
#             for e in epsilon:
#                 fname = infile.replace('@World@', w).replace('@L@', str(l)).replace('@q@', str(q)).replace('@E@', str(e))
#                 data = pd.read_csv(fname, sep=',', header='infer')
#                 data['world'] = w
#                 data['learning_rate'] = l
#                 data['gInit'] = q
#                 data['epsilon'] = e
#                 data_list_2 = pd.concat([data_list, data], ignore_index=True)
#
#
# # plot
#
# for w in worlds:
#     for l in [0.1]:
#         for q in [0.0]:
#             for e in [0.1]:
#                 plt.plot(data_list_2[(data_list_2['world'] == w) &
#                                    (data_list_2['learning_rate'] == float(l)) &
#                                    (data_list_2['gInit'] == float(q)) &
#                                    (data_list_2['epsilon'] == float(e))]['iter'],
#                          data_list_2[(data_list_2['world'] == w) &
#                                    (data_list_2['learning_rate'] == float(l)) &
#                                    (data_list_2['gInit'] == float(q)) &
#                                    (data_list_2['epsilon'] == float(e))]['reward'],
#                          '.-', label='with decay '+'learning_rate_' + str(l) + '_gInit_' + str(q) + '_epsilon_' + str(e))
#
#                 plt.plot(data_list[(data_list['world'] == w) &
#                                    (data_list['learning_rate'] == float(l)) &
#                                    (data_list['gInit'] == float(q)) &
#                                    (data_list['epsilon'] == float(e))]['iter'],
#                          data_list[(data_list['world'] == w) &
#                                    (data_list['learning_rate'] == float(l)) &
#                                    (data_list['gInit'] == float(q)) &
#                                    (data_list['epsilon'] == float(e))]['reward'],
#                          '.-', label='no decay '+'learning_rate_' + str(l) + '_gInit_' + str(q) + '_epsilon_' + str(e))
#
#     plt.rc('axes', prop_cycle=(cycler('color', colors)))
#     plt.xlabel('iteration')
#     plt.ylabel('reward')
#     plt.ylim(-800, 100)
#     plt.grid(True)
#     plt.legend(loc="best")
#     plt.title('{}--{}'.format(w, ' Q-Learning with or without decay'))
#     plt.show()
#
