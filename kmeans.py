"""
k-means algorithm 
(if you provide the true labels array, this function will calculate the accuracy based on entropy)
"""

import numpy as np
def kmeans(X, k_or_y:'k is the number of clusters, y is the true labels array'):
    try: m,n = X.shape
    except AttributeError: m,n = len(X),len(X[0])
    
    try: #assume k_or_y is the target-array
        k = len(set(k_or_y))
        y = tuple(k_or_y)
    except TypeError: # assume k_or_y is an int
        k = int(k_or_y)
        y = None
    
    
    # container for the current centroids
    from collections import deque
    q = deque(maxlen=2)       
    
    #initialize centroids
    mn = X.min(0)
    mx = X.max(0)
    r = (X.max(0)-X.min(0)).max() # range along the longest dimension
    epsilon = r/100
    
    #C = centroids matrix
    C = np.random.uniform(low=mn, high=mx, size=(k,n))
    
    
    for loop in range(100):
        #D = distance matrix
        D = np.square(np.expand_dims(X, axis=0) - np.expand_dims(C, axis=1)).sum(axis=-1).T
        
        #assign labels
        labels = D.argmin(axis=1)
        if len(set(labels))<k:
            C = np.random.uniform(low=mn, high=mx, size=(k,n))
            continue
        
        #new centroids
        C = np.array([X[labels==label].mean(axis=0) for label in range(k)])
        
        #save current centroids
        q.append(C)
        
        d = np.square(np.subtract.reduce(q)).sum(axis=1).max()
        if d < epsilon: 
            print("\nbreaking after loop#", loop)
            break
    else: print("failed to converge")
    
    #asign the datapoints to the centroids
    D = np.square(np.expand_dims(X, axis=0) - np.expand_dims(C, axis=1)).sum(axis=-1).T
    labels = D.argmin(axis=1)
    #bincounts = np.bincount(labels)
    d = {c:i for i,c in enumerate(np.bincount(labels).argsort())}
    labels = [d[k] for k in labels]
    

    if y:
        #sort the Centroids matrix by the distance from the origin
        nx = np.square(C).sum(1).argsort()
        C = C[nx]
        
        if len(set(y)) != k:print("number of clusters do not match")
        Creal = np.array([X[labels==label].mean(axis=0) for label in set(y)])
        
        nx = np.square(Creal).sum(1).argsort()
        Creal = Creal[nx]
        
        d = np.sqrt(np.square(C-Creal).sum(axis=1)).mean()
        print("average distance between the true and fitted centroids =", round(d,3))
        

        def entropy(a):
            from math import log2
            pp = [a.count(n)/len(a) for n in set(a)]
            entropy = -sum(p*log2(p) for p in pp)  # the less the better
            return(entropy)
        
        arrays = (tuple(np.array(labels)[y==c]) for c in sorted(set(y)))
        from statistics import mean
        entropy = mean(entropy(a) for a in arrays)
        print("average entropy =", round(entropy,3), "(the less the better)")
        
    return(labels)



#demo
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
X, y = make_blobs()


labels = kmeans(X, y)
plt.scatter(*X.T, c=labels)



from untitled2 import make_data_for_classification
X,y = make_data_for_classification(m=150, n=5, k=4, blobs_density=0.5)

labels = kmeans(X, y)

