"""
k-means algorithm
"""

import numpy as np
from sklearn.metrics import pairwise_distances_argmin
from collections import deque

log = []
def fitness(X,y,C):
    """fitness function that is supposed to be maximized"""
    global log
    distances = ((X-C[y])**2).sum(axis=1)**0.5
    average_distance = distances.mean()
    log.append(average_distance)
    b = log == sorted(log)[::-1]  # is continuously decreasing?
    return -average_distance  # negative because this is a MAXIMAZATION function


def kmeans(X, n_clusters=2, max_iter=100):
    try: m,n = X.shape
    except AttributeError: m,n = len(X),len(X[0])
    k = n_clusters
    
    # container for the current centroids
    q = deque(maxlen=2)       
    
    #C = centroids matrix
    C = X[np.random.permutation(len(X))[:k]]
    
    #LOOP
    for loop in range(max_iter):
        #EXPECTATION (i.e. update expectation of which cluster each point must belong to)
        labels = pairwise_distances_argmin(X,C)
        
        #if one centroid is left without any points assigned to it
        if len(set(labels))<k:
            print("One (or) more centroids is left without any points assigned to it. Initializing random centroids again..")
            C = X[np.random.permutation(len(X))[:k]]
            continue
        
        #MAXIMAZATION
        C = np.array([X[labels==label].mean(axis=0) for label in range(k)])
        f = fitness(X,labels,C)
        print("loop {} maximizing the fitness function: {:.4f}".format(loop,f))
        
        #save current centroids
        q.append(C)
        
        #check for convergence
        if len(q)>1 and np.allclose(*q):
            print("\nbreaking after loop#", loop)
            break
            
    else: print("failed to converge")    
    return(labels,C)  # C = centroids
    

# DEMO ############################

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
X, y = make_blobs()
labels,centroids = kmeans(X, n_clusters=3)
plt.scatter(*X.T, c=labels)

    
def load_from_github(url):
    from urllib.request import urlopen
    from os import remove
    
    obj = urlopen(path)
    assert obj.getcode()==200,"unable to open"

    s = str(obj.read(), encoding="utf-8")
    NAME = "_temp.py"
    with open(NAME, mode='wt', encoding='utf-8') as fh: fh.write(s)
    module = __import__(NAME[:-3])
    remove(NAME)
    return module


path = r"https://raw.githubusercontent.com/leztien/synthetic_datasets/master/make_data_for_classification.py"
module = load_from_github(path)
n_clusters = 5
X,y = module.make_data_for_classification(m=100, n=10, k=n_clusters, blobs_density=0.9)
labels,centroids = kmeans(X, n_clusters=n_clusters)
print(np.bincount(y), np.bincount(labels))
