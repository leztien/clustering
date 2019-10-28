
"""
demo on Univariate Gaussian Mixture Classifier
"""

import numpy as np
from numpy import exp, sqrt
import matplotlib.pyplot as plt
from scipy.stats import norm


def make_data(): #makes 3 classes
    from numpy import array, concatenate
    from numpy.random import normal
    data = normal(loc=[0,10,15], scale=[2,1,0.5], size=(10,3))
    x0 = data[:,0]; x1 = data[:-3,1]; x2 = data[:-5,2]
    y = array([0]*len(x0) + [1]*len(x1) + [2]*len(x2), dtype="uint8")
    x = concatenate((x0,x1,x2))
    x -= x.min()-1
    return(x,y)
    

def visualize_points_and_their_gaussians(x,y):
    for k in sorted(set(y)):
        color = ["g","orange","b"]
        mu,sd = x[y==k].mean(), x[y==k].std(ddof=0)
        pdf = norm(mu,sd).pdf
        xx = np.linspace(x[y==k].min()-10, x[y==k].max()+10, 100)
        yy = pdf(xx)
        mask = yy > 0.001
        xx,yy = (a[mask] for a in (xx,yy))
        plt.plot(xx,yy, color='#999999')
        plt.fill(xx,yy, alpha=0.5, color=color[k])
        plt.plot(x[y==k], [0]*len(x[y==k]), 'o', color=color[k], mec='#999999', ms=8)
    return(plt.gca())


def cost(x,y,μ,σ):
    from numpy import log as ln
    return -ln(pdf(x, loc=μ[y], scale=σ[y])).sum()/m
    
#################################################################    

x,y = make_data()
X = x.reshape(-1,1)


#prepare
pdf = norm.pdf
m = len(x)
K = len(set(y))
max_iter = 1000

#initialize
def initialize(x,K):
    m = len(x)
    nx = np.random.choice(m, K, replace=False)
    μ = x[nx]
    σ = np.array([1]*K, dtype="float64")
    priors = np.array([1/K]*K)
    return μ,σ,priors

μ,σ,priors = initialize(x,K)

log = list()
for epoch in range(max_iter):
    #expectation
    likelihoods = pdf(X, loc=μ, scale=σ)
    marginals = (likelihoods*priors).sum(1, keepdims=True)
    posteriors = likelihoods*priors / marginals
    
    #maximization
    μ = (posteriors*X).sum(0) / posteriors.sum(0)
    σ = sqrt( ((X-μ)**2*posteriors).sum(0) / posteriors.sum(0) )
    priors = posteriors.sum(0)/m

    #check for missing classes
    ypred = posteriors.argmax(1)
    if len(set(ypred))<3: 
        μ,σ,priors = initialize(x,K)
        continue
    
    #check convergence
    J = -np.log((likelihoods * priors).sum(1)).sum()/m
    log.append(J)
    if len(log)>7 and len(set([round(c,7) for c in log[-7:]]))==1: break
    

sp = visualize_points_and_their_gaussians(x,ypred)
sp.text(0.4, 0.9, f"epoch: {epoch}", transform=sp.transAxes)


