import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from numpy import genfromtxt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import neighbors

def genDataSet(N):
    X= np.random.normal(0,1,N)
    ytrue = (np.cos(X) + 2)/(np.cos(X*1.4)+2)
    noise = np.random.normal(0,0.2,N)
    y = ytrue+noise
    return X, y, ytrue

X, y, ytrue = genDataSet(1000)
plt.plot(X, y, '.')
plt.plot(X, ytrue, 'rx')
plt.show()

X = X.reshape(len(X), 1)
bestk = []
kc = 0

for n_neighbors in range(1,999):
    kf = KFold(n_splits=10)
    kscore = []
    k = 0
    for train, test in kf.split(X):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        #we create an instance of Neighbors regressor and fit the data.
        clf = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
        clf.fit(X_train, y_train)
        kscore.append(clf.score(X_test, y_test))
        k = k + 1
        #print n_neighbors
        bestk.append(sum(kscore) / len(kscore))
        kc += 1
# to do here: given this array of E_outs in CV, find the max, its
# corresponding index, and its corresponding value of n_neighbors
    #this method allows me to control cv
    scores = cross_val_score(clf, X, y, cv=10)
    print 'kscore:'
    print kscore
    print 'scores:'
    print scores


nbk = sorted(bestk,reverse = True)
getindex = [nbk[0], nbk[1], nbk[2]]
print 'getindex'
print getindex
#eout = clf.score(clf,X,y)
#print 'eout:'
#print eout

#idx = sorted(range(len(bestk), key = bestk.__getiten__))
#print (idx[0] + 2)+1
#print (idx[1] + 2)+1
#print (idx[2] + 2)+1
print 'bestk:'
print bestk

