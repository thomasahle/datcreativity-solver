import numpy as np
import random

print('Loading data')
# Load Glove vectors preprocessed by
# https://github.com/FALCONN-LIB/FALCONN/blob/a4c0288e/src/examples/glove/convert.py
X = np.load('data/glove.840B.300d.npy')

print('Loading words')
with open('data/words84') as file:
    wid = {w.rstrip(): i for i,w in enumerate(file)}

print('Loading nouns')
# Loading Dat-Creativity word list
with open('data/words-common.txt') as file:
    nouns = [w.rstrip() for w in file]
#with open('nouns') as file:
#    restricted_nouns = {w.split('/')[0] for w in file}
#    nouns = [w for w in nouns if w in restricted_nouns]
nouns = [w for w in nouns if w in wid]

# Throw away word vectors not needed
X = X[[wid[w] for w in nouns]]
wid = {w:i for i,w in enumerate(nouns)}
N = len(nouns)
print(f'Read {N} words')

print('Normalizing')
X /= np.linalg.norm(X, axis=1, keepdims=True)

K = 7
D = X.shape[1]

#
# Improvement strategies
#

def improve_indi(ws):
    ''' Greedy replace for each word, one at a time '''
    new = []
    for i in range(K):
        other = new + ws[i+1:]
        new.append((X @ X[other].T).sum(axis=1).argmin())
    return new

def improve_all(ws):
    ''' Greedy replace for all words at the same time '''
    ips = X @ X[ws].T
    new = []
    total = ips.sum(axis=1)
    for i in range(len(ws)):
        new_i = (total - ips[:,i]).argmin()
        new.append(new_i)
    return new

#
# Seed strategies
#

def sample_randn():
    ''' Sample K points by vicinity to random gaussians '''
    return list((X @ np.random.randn(D, K)).argmax(axis=0))

def sample_kmeans():
    ''' Sample K points by vicinity to kmeans '''
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=K, n_init=1, max_iter=1).fit(X)
    return list((X @ kmeans.cluster_centers_.T).argmax(axis=0))

def sample_simple():
    ''' Sample K points uniformly at random '''
    return random.sample(range(N), K)

#
# Main
#

def score(ws):
    # Since the vectors are normalized, it doesn't matter
    # that we include the diagonal entries, as they are always
    # the same
    return -np.sum(X[ws] @ X[ws].T)

def print_table(ws):
    pairs = X[ws] @ X[ws].T
    print((100*(1-pairs)).round())
    print('Score:', (100*(1-pairs)).sum() / K / (K-1))

def main():
    best_ws, best_score = [], -100
    while True:
        rand_ws = sample_simple()
        #rand_ws = sample_randn()
        #rand_ws = sample_kmeans()
        rand_score = score(rand_ws)

        progress = True
        while progress:
            progress = False
            #ws = improve_all(rand_ws)
            ws = improve_indi(rand_ws)
            if (new_score := score(ws)) > rand_score:
                rand_ws, rand_score = ws, new_score
                progress = True

        if rand_score > best_score:
            best_ws, best_score = rand_ws, rand_score
            print(', '.join(nouns[w] for w in best_ws))
            print_table(best_ws)

if __name__ == '__main__':
    main()


