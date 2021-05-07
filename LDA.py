
"""
LDA
"""


import numpy as np
import matplotlib.pyplot as plt


def random_word_distribution(n_words, n_topics=1, heating=None):
    """word distributions over topics"""
    heating = heating or 1.9   # try 5+
    D = np.zeros(shape=(n_topics+1, n_words), dtype=np.float64)
    
    for i in range(0, n_topics, 2):
        pp = np.exp((np.log(np.random.uniform(0, 1, size=n_words)) * heating))
        pp = pp / pp.sum()
        D[i] = pp
        pp = pp.max()+(1E-10) - pp
        D[i+1] = pp / pp.sum()
    return D[:n_topics][0 if n_topics==1 else ...]
        

def random_topic_distribution(n_topics, n_documents=1, heating=None):
    """topic-mix for a document"""
    heating = heating or 3     # the higher the steeper
    rate = np.log(n_topics) * heating    
    std = rate / 10
    rates = np.random.normal(rate, std, size=n_documents)[:,None]
    xx = np.linspace(0.01, 1, n_topics)
    yy = np.exp(xx*rates)
    P = yy / yy.sum(axis=1, keepdims=True)
    nx = np.random.rand(*P.shape).argsort(axis=1)
    P = np.take_along_axis(P, nx, axis=1)
    return P[0 if n_documents==1 else ...]


def sample(probabilities, n_points=1):
    """sample a value for a multinomial distribution"""
    if n_points == 1:
        probabilities = np.array(probabilities)
        probabilities = probabilities / probabilities.sum()  # normalize so that the probabilities sum up to 1.0
        cum = probabilities.cumsum()
        r = np.random.random()
        for i,p in enumerate(cum):
            if p >= r: return i
    return [sample(probabilities, 1) for _ in range(n_points)]


def make_data(m:'n_documents', n:'n_words in each document',
              k:'n_topics', v:'length of the vovabulary'):
    """generate documents for LDA"""
    topics = random_word_distribution(n_words=v, n_topics=k)
    topic_mixes = Y = random_topic_distribution(n_topics=k, n_documents=m)
    X = np.zeros(shape=(m,n), dtype=np.uint16)
    
    for i in range(m):
        for j in range(n):
            X[i,j] = sample(topics[sample(topic_mixes[i])])
    return (X+1, Y.astype(np.float16))  #  "x+1" turns zeros into ones etc  

############################################################################


def p_topics_given_document(n_topics, document_index, word_index, assignements):
    smoother = 1
    k,i,A = n_topics, document_index, assignements
    d = np.delete(A[i], word_index)  # treat the current assignement as unknown
    pp = np.append(np.bincount(d)+smoother, [smoother]*k)[:k] / (len(d)+k*smoother)
    return pp
    

def p_word_given_topics(word, word_distributions):
    smoother = 1
    M = word_distributions
    pp = (M[:, word] + smoother) / (M[:, -1] + smoother)
    return pp


def topic_mixes(assignements, n_topics=None):
    A = assignements
    m,n = len(A), len(A[0])
    k = n_topics or len(set(A.ravel()))
    Y = np.zeros(shape=(m,k), dtype=np.float16)
    for i in range(m):
        pp = np.append(np.bincount(A[i]), [0]*k)[:k] / n
        Y[i] = pp
    return Y


def clustering_accuracy(ytrue, ypred):
    from scipy.stats import mode
    d = dict()
    for c in sorted(set(ytrue)):
        mask = [ytrue==c]
        d[c] = mode(ypred[tuple(mask)])[0][0]
    ytrue = [d[y] for y in ytrue]
    acc = np.equal(ytrue, ypred).sum() / len(ytrue)
    return acc


#####################################################################

### DATA ###
    
m = 300 # number of documents
n = 250  # number of words in each documents
k = 3   # number of topics
v = 1000 # length of the vocabulary


X,Y = make_data(m, n, k, v)
y = ytrue = Y.argmax(axis=1)

#______________________________________________________________


def lda_clustering(data, n_topics, n_iter=10):
    """LDA clustering"""
    X = data
    k = n_topics
    m,n = X.shape
    
    ### PREPARATION ###
    # Make a dictionary to enlabel words in X (incrementally)
    vocabulary = sorted(set(X.ravel()))   # actual vocabulary
    d = {w:i for (i,w) in enumerate(vocabulary)}
    
    # New X with enlabled words
    X = np.vectorize(lambda w: d[w])(X)
    
    # Make an Assignement matrix  (word -> topic)
    A = np.random.randint(0, k, size=(m,n))
    
    # Multinomial distributions:  (topics x words)
    M = np.zeros(shape=(k, len(vocabulary)+1), dtype=np.int32)  # +1 is for the marginals
    
    # Fill in the M-matrix
    for i in range(m):
        for j in range(n):
            M[A[i,j], X[i,j]] += 1
    
    # Marginals
    M[:,-1] = np.array(M.sum(axis=1))
    
    
    ### LOOPING ###
    for epoch in range(n_iter):
        for i in range(m):
            for j in range(n):
                # Asignement, word  as indeces
                a = A[i,j]
                w = X[i,j]
                # Subtract one for the current
                M[a,w] -= 1
                M[a,-1] -= 1
                assert M[a,w] >= 0 and M[a,-1] >= 0, "bug detected"
                # Compute the two probabilities
                p1 = p_topics_given_document(n_topics=k, document_index=i, word_index=j, assignements=A)
                p2 = p_word_given_topics(word=w, word_distributions=M)
                # Probabilities of topics given word
                pp = p1 * p2    # these probs are not normalized
                # Sample from that multinomial distribution
                a = sample(pp) # New assignement to the current word
                A[i,j] = a
                # Update the M-matrix by incrementing by one
                M[a,w] += 1
                M[a,-1] += 1
    # Save the Assignements matrix for the possibility of post-analysis
    lda_clustering.assignements = A
    lda_clustering.word_distributions = M
    # Return predicted topic mixes     
    return topic_mixes(assignements=A, n_topics=k) 



# Test and evaluate
Ypred = lda_clustering(X, n_topics=k, n_iter=20)
ypred = Ypred.argmax(axis=1)
acc = clustering_accuracy(ytrue, ypred)
print("my lda accuracy:", acc)


### SKLEARN ###
from sklearn.decomposition import LatentDirichletAllocation
md = LatentDirichletAllocation(k, max_iter=20)
P = md.fit_transform(X)
ypred_sklearn = P.argmax(axis=1)
acc = clustering_accuracy(ytrue, ypred_sklearn)
print("sklearn accuracy:", acc)

