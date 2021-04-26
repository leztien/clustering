
"""
AgglomerativeHierarchicalClustering
"""




def load_from_github(url):
    from urllib.request import urlopen
    from os import remove
    obj = urlopen(url)
    assert obj.getcode()==200,"unable to open"
    s = str(obj.read(), encoding="utf-8")
    NAME = "_temp.py"
    with open(NAME, mode='wt', encoding='utf-8') as fh: fh.write(s)
    module = __import__(NAME[:-3])
    remove(NAME)
    return module



url = r"https://raw.githubusercontent.com/leztien/synthetic_datasets/master/make_data_for_classification.py"
module = load_from_github(url)
X,y = module.make_data_for_classification(m=50, n=2, k=3, blobs_density=0.5)

m,n,k = len(X), len(X[0]), len(set(y))


##############################################################################



class AgglomerativeHierarchicalClustering:
    def __init__(self, n_clusters, linkage='single'):
        self.k = n_clusters
        self.linkage = str(linkage or 'single').lower()
        self.ypred = None
    
    def fit_predict(self, data):
        self.d = self.distance_matrix(data)
        clusters = tuple({i} for i in range(len(data)))
        
        C = [clusters,]
        for k in range(len(X)-1, 0, -1):
            clusters = self.agglomerate_clusters(clusters)
            C.append(clusters)
        
        self.ypred = self.classify(C, k=3)
        return self.ypred
        
    def predict(self, data=None):
        if not self.ypred:
            self.fit_predict(data)
        return self.ypred


    @staticmethod
    def squared_distance(a,b):
        return sum((a-b)**2 for a,b in zip(a,b))
        
    
    def distance_matrix(self, data):
        d = {(i,j): self.squared_distance(data[i], data[j]) for i in range(len(data)) for j in range(i+1, len(data))}
        d.update({k[::-1]:v for k,v in d.items()})
        return d
        
    
    def cluster_distance(self, c1, c2):
        switch = {'single':min, 'complete':max, 'average': lambda a: sum(a)/len(a)}
        distances = [self.d[(i,j)] for i in c1 for j in c2]
        return switch[self.linkage](distances)
        
    
    def agglomerate_clusters(self, clusters):
        clusters = list(clusters)
        cluster_distances = [((i, j), self.cluster_distance(clusters[i], clusters[j])) 
             for i in range(len(clusters)) for j in range(i+1, len(clusters))]
        i,j = closest = sorted(cluster_distances, key=lambda t: t[-1])[0][0]
        
        # Create a new (merged) cluster
        try:
            merged_cluster = clusters[i].union(clusters[j])
        except AttributeError:
            merged_cluster = clusters[i] + clusters[j]
        
        # Remove the two clusters
        clusters[i] = None; clusters[j] = None
        clusters.remove(None); clusters.remove(None)
        
        # Append the new cluster
        clusters.append(merged_cluster)
        return clusters
    
    @staticmethod
    def classify(C, k):
        assert 0 < k <= len(C)
        clusters = sorted(C[len(C)-k], key=len, reverse=True)
        y = [None] * sum(len(c) for c in clusters)
        for klass,cluster in enumerate(clusters):
            for ix in cluster:
                y[ix] = klass
        return tuple(y)
            
    




md = AgglomerativeHierarchicalClustering(n_clusters=3, linkage='complete')
ypred = md.fit_predict(X)




# Demo
if len(X[0]) == 2:
    import matplotlib.pyplot as plt
    plt.scatter(*zip(*X), c=ypred)
    
    from sklearn.cluster import AgglomerativeClustering
    md = AgglomerativeClustering(n_clusters=3, linkage='complete')
    ypred = md.fit_predict(X)
    plt.figure()
    plt.scatter(*zip(*X), c=ypred)

