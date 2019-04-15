# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 22:57:10 2017

@author: Jon
"""

import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

# infiles = ['./Policy Easy Iter 43 Policy Map.pkl',
#            './Policy Hard Iter 22 Policy Map.pkl',
#            './Value Easy Iter 78 Policy Map.pkl',
#            './Value Hard Iter 61 Policy Map.pkl']


# infiles = ['../Solution2/QL Q-Learning L0.9 q-100.0 E0.5 Hard Iter 43 Policy Map.pkl',
#            '../Solution2/QL Q-Learning L0.1 q-100.0 E0.5 Hard Iter 15 Policy Map.pkl',
#            '../Solution2/QL Q-Learning L0.1 q-100.0 E0.3 Hard Iter 19 Policy Map.pkl',
#            '../Solution2/QL Q-Learning L0.1 q-100.0 E0.1 Hard Iter 12 Policy Map.pkl']

infiles = ['./QL Q-Learning L0.1 q0.0 E0.1 Hard Iter 3000 Policy Map.pkl',
           './QL Q-Learning L0.1 q0.0 E0.1 Easy Iter 791 Policy Map.pkl']

# infile = './Policy Easy Iter 43 Policy Map.pkl'
# infile = './Policy Hard Iter 22 Policy Map.pkl'
# infile = './Value Hard Iter 61 Policy Map.pkl'
# infile = './Value Easy Iter 78 Policy Map.pkl'


# infile = './QL Q-Learning L0.1 q0.0 E0.1 Easy Iter 791 Policy Map.pkl'
# infile = './QL Q-Learning L0.1 q0.0 E0.1 Hard Iter 3000 Policy Map.pkl'

for infile in infiles:
    with open(infile,'rb') as f:
        # arr = pkl.load(f, encoding='latin1')
        arr = pkl.load(f)

    lookup = {'None': (0,0),
              '>': (1,0),
            'v': (0,-1),
            '^':(0,1),
            '<':(-1,0)}

    n= len(arr)
    arr = np.array(arr)
    X, Y = np.meshgrid(range(1,n+1), range(1,n+1))
    U = X.copy()
    V = Y.copy()
    for i in range(n):
        for j in range(n):
            U[i,j]=lookup[arr[n-i-1,j]][0]
            V[i,j]=lookup[arr[n-i-1,j]][1]

    plt.figure()
    #plt.title('Arrows scale with plot width, not view')
    plt.title(infile)
    Q = plt.quiver(X, Y, U, V,headaxislength=5,pivot='mid',angles='xy', scale_units='xy', scale=1)
    plt.xlim((0,n+1))
    plt.ylim((0,n+1))
    plt.tight_layout()
    plt.show()
