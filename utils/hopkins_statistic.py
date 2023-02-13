from sklearn.neighbors import NearestNeighbors
from random import sample
import numpy as np
from numpy.random import uniform 
from math import isnan 

from argparse import ArgumentParser

def hopkins(X) : 
    '''Calculates the clustering tendency of a 2d array using hopkin's statistic. 
    
    Input : 
        X : input 2d np array 
    Output : 
        hopkin's statistic    
    '''
    d = X.shape[1]
    n = len(X)
    m = int(0.1 * n)

    nbrs = NearestNeighbors(n_neighbors = 1).fit(X)
    rand_X = sample(range(0,n,1),m)

    ujd = []
    wjd = []

    for j in range(0,m) :
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X, axis=0), np.amax(X,axis=0),d).reshape(1,-1),2,return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X[rand_X[j]].reshape(1,-1),2,return_distance=True)
        wjd.append(w_dist[0][1])

    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H) : 
        print(ujd, wjd)
        H = 0 

    return round(H,3)

def hopkin_iter(X,iterations=10) : 
    '''Runs hopkin statistic for n iterations and reports mean and std of hopkin's statistic'''
    hopkins_stats = []
    for i in range(iterations) : 
        hopkins_stats.append(hopkins(X))

    return round(np.mean(hopkins_stats),3), round(np.std(hopkins_stats),3)
if __name__ == '__main__' : 
    parser = ArgumentParser()

    parser.add_argument('--test', type=str, default = 'false')

    args = parser.parse_args()

    if args.test == 'true' : 
        x = np.random.uniform(size=(1000,1000))
        iterations = 10
        mean, std = hopkin_iter(x, iterations)
        print(f'Test Hopkin statistic run for {iterations} iterations - mean +/- std: {mean} +/- {std}')