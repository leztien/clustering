
"""
demo of image partitioning based on  linear space partitioning  vs  kmeans
"""

import numpy as np
from matplotlib import pyplot as plt


def download_ndarray(url:"url of .npy file") -> "numpy-array":
    """download a (numpy-pickled) ndarray from the web"""
    from numpy import load
    from urllib.request import urlopen
    from urllib.error import URLError
    from tempfile import TemporaryFile
    from shutil import copyfileobj
    from sys import exit
    
    try: rs = urlopen(URL)   # rs = response-object
    except URLError: print("unable to download"); exit()
    
    with rs as rs, TemporaryFile(mode='w+b', suffix='.npy', delete=True) as fh:
        if rs.getcode() != 200: print("unable to download"); exit()
        copyfileobj(rs, fh)  
        fh.seek(0)
        nd = load(fh)
    #just in case:
    rs.close(); fh.close(); del rs, fh
    return(nd)

#======================================================================================

#plt
plt.rcParams['figure.dpi'] = 110
fig,(sp1,sp2) = plt.subplots(2,1, figsize=(15,15), subplot_kw={'xticks':[],'yticks':[]})

#get data
URL = r"https://github.com/leztien/datasets/blob/master/korean.npy?raw=true"
pn = download_ndarray(URL)
X = pn.reshape(pn.shape[0]*pn.shape[1], pn.shape[2]).astype(np.uint64)


"""IMAGE COLOUR COMPRESSION BASED ON LINEAR SPACE PARTITIONING"""
n = 5 # n_colours per dimension
r = np.linspace(0,255, num=n).round().astype('uint8')
from itertools import product
C = np.array(tuple(product(r,r,r)), dtype=np.uint64)

nx = ((X[None,...] - C[:,None,:])**2).sum(axis=-1).argmin(axis=0)
pn = C[nx].reshape(*pn.shape)
im = sp1.imshow(pn)
sp1.set_title("linear space partitioning")


"""IMAGE COLOUR COMPRESSION BASED ON KMEANS"""
n_clusters = 16   # n_colors
from sklearn.cluster import KMeans
md = KMeans(n_clusters=n_clusters, n_init=1, tol=0.1, max_iter=50).fit(X)
y = md.predict(X)
C = md.cluster_centers_.round().astype('uint64')

pn = C[y].reshape(*pn.shape)
im = sp2.imshow(pn)
sp2.set_title("kmeans")

"""
as you can see kmeans-compression is better than linear-space-partitioning
"""
