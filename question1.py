import random
import matplotlib.pyplot as plt
import numpy as np
import copy
from numpy import genfromtxt

class Perceptron:
    def __init__(self, N):
        self.X = self.generate_points(N)

    def generate_points(self, N):
        X, y = self.make_data(N)
        bX = []
        for k in range(0, N):
            bX.append((np.concatenate(([1], X[k, :])), y[k]))
        # this will calculate linear regression at this point
        X = np.concatenate((np.ones((N, 1)), X), axis=1);  # adds 1 as a constant
        self.linRegW = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
        print self.linRegW
        return bX

    def make_data(self, N):
        dataset = genfromtxt('features.csv', delimiter=' ')
        y = dataset[0:N, 0]
        X = dataset[0:N, 1:]
        y[y<>1] = -1
        y[y==1] = +1
        c0 = plt.scatter(X[y == -1, 0], X[y == -1, 1], s=20, color='r', marker='x')
        c1 = plt.scatter(X[y == 1, 0], X[y == 1, 1], s=20, color='b', marker='o')
        plt.legend((c0,c1), ('All other numbers -1', 'Number zero +1'),
                   loc='upper right', scatterpoints=1, fontsize=11)
        plt.xlabel(r'$x1$')
        plt.ylabel(r'$x2$')
        plt.title(r'Intensity and Symmetry of Digits')
        plt.savefig('midterm.plot.png', bbox_inches='tight')
        plt.show
        return X, y

    def plot(self, mispts=None, vec=None, save=False):
        fig = plt.figure(figsize=(5, 5))
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        l = np.linspace(-1.5, 2.5)
        V = self.linRegW
        a, b = -V[1] / V[2], -V[0] / V[2]
        plt.plot(l, a * l + b, 'k-')
        cols = {1: 'r', -1: 'b'}
        for x, s in self.X:
            plt.plot(x[1], x[2], cols[s] + '.')
        if mispts:
            for x, s in mispts:
                plt.plot(x[1], x[2], cols[s] + 'x')
        if vec.size:
            aa, bb = -vec[1] / vec[2], -vec[0] / vec[2]
            plt.plot(l, aa * l + bb, 'g-', lw=2)
        if save:
            if not mispts:
                plt.title('N = %s' % (str(len(self.X))))
            else:
                plt.title('N = %s with %s test points' \
                          % (str(len(self.X)), str(len(mispts))))
            plt.savefig('p_N%s' % (str(len(self.X))), \
                        dpi=200, bbox_inches='tight')

    def classification_error(self, vec, pts=None):
        if not pts:
            pts = self.X

        M = len(pts)
        n_mispts = 0

        for x, s in pts:
            if int(np.sign(vec.T.dot(x))) != s:
                n_mispts += 1
        error = n_mispts / float(M)
        return error

    def choose_miscl_point(self, vec):
        pts = self.X
        mispts = []
        for x, s in pts:
            if int(np.sign(vec.T.dot(x))) != s:
                mispts.append((x, s))
        return mispts[random.randrange(0, len(mispts))]

    def pla(self, save=False, linear=False):
        #set w=0 to start if we are running pocket algorithm and set it equal to the solution found by linear regression if we are running linear regression
        if linear != False:
            w=self.linRegW
        else:
            w = np.zeros(3)

        self.keepW = copy.deepcopy(w)
        self.plaError = []
        self.pocketError = []
        X, N = self.X, len(self.X)
        #set number of iterations = 0
        it = 0
        lastIT = 0
        self.plaError.append(self.classification_error(w))
        self.pocketError.append(self.plaError[it])
        #keep running until all points are correctly classified or we exceed 10 iterations of pocket
        while self.plaError[it] != 0 and lastIT < 1000:
            it += 1
            # Pick random misclassified point
            x, y = self.choose_miscl_point(w)
            # Update weights
            w += y * x
            # Update if new w is better
            self.plaError.append(self.classification_error(w))
            #increment the count to check if there is a change in w. this will allow us to stop running if w is not changing
            if self.pocketError[it-1] < self.plaError[it]:
                lastIT+=1

            if (self.pocketError[it - 1] > self.plaError[it]):  # for Pocket
                self.pocketError.append(self.plaError[it])
                self.keepW = copy.deepcopy(w)
            else:
                self.pocketError.append(self.pocketError[it - 1])


        #    if (save==True) and it % 100 == 0 or it ==1:
         #       self.plot(vec=w)
          #      plt.title('N = %s, Iteration %s\n' \
           #               % (str(N), str(it)))
            #    plt.savefig('p_N%s_it%s' % (str(N), str(it)), \
             #               dpi=200, bbox_inches='tight')

        self.w = w
        print 'PLA Error:'
        print self.plaError
        print 'Pocket Error:'
        print self.pocketError
        print 'Best w:'
        print self.keepW
        print lastIT

        return it

    def check_error(self, M, vec):
        check_pts = self.generate_points(M)
        return self.classification_error(vec, pts=check_pts)

def main():
    it = np.zeros(1)
    for x in range(0, 1):
        p = Perceptron(7291)
        it[x] = p.pla(save=True)
        print it
main()