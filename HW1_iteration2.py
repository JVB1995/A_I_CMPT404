import numpy as np
import random
import matplotlib.pyplot as plt
import os, subprocess
class Perceptron:
    histArray = []
    bins = [10, 20, 30, 40, 50, 60]

    def __init__(self, N):
        # Random linearly separated data
        #xA, yA, xB, yB, xC, yC,xD, yD, xE, yE, xF, yF = [random.uniform(-1, 1) for i in range(12)]
        #self.V = np.array([(xB * yA - xA * yB), (yB - yA), (xA - xB), (xC * yD - xD * yC), (yC - yD),(xD - xC),
                          # (xF * yE - xE * yF), (yF - yE), (xE -xF), 0, 1])
        #self.V = np.array([(xB * yA - xA * yB), (yB - yA), (xA - xB), (xC * yB - xB * yC), (yC - yB),
          #                 (xB - xC), (xD * yC - xC * yD), (yD - yC), (xC - xD), (xE * yD - xD * yE),
           #                (yE - yD), (xD - xE), (xF * yE - xE * yF), (yF - yE), (xE - xF)])
       # xA, yA, xB, yB = [random.uniform(-1, 1) for i in range(4)]
       # self.V = np.array([xB * yA - xA * yB, yB - yA, xA - xB])
        self.V = (np.random.rand(11)*2)-1
        self.X = self.generate_points(N)

    def generate_points(self, N):
        X = []
        #for i in range(N):
           # x1,x2,x3,x4,x5,x6,x7,x8,x9,x10 = [random.uniform(-1, 1) for i in range(10)]
           # x = np.array([1,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10])
           # s = int(np.sign(self.V.T.dot(x)))
           # X.append((x, s))
        for i in range(N):
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = [random.uniform(-1, 1) for i in range(10)]
            x = np.array([1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
            s = int(np.sign(self.V.T.dot(x)))
            X.append((x, s))
        return X

    def plot(self, mispts=None, vec=None, save=False):
        fig = plt.figure(figsize=(5, 5))
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        V = self.V
        a, b = -V[1] / V[2], -V[0] / V[2]
        l = np.linspace(-1, 1)
        plt.plot(l, a * l + b, 'k-')
        cols = {1: 'r', -1: 'b'}
        for x, s in self.X:
            plt.plot(x[1], x[2], cols[s] + 'o')
        if mispts:
            for x, s in mispts:
                plt.plot(x[1], x[2], cols[s] + '.')
        if vec != None:
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
   # plt.clf()
  #  plt.cla()
    #plt.close()
    def classification_error(self, vec, pts=None):
        # Error defined as fraction of misclassified points
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
        # Choose a random point among the misclassified
        pts = self.X
        mispts = []
        for x, s in pts:
            if int(np.sign(vec.T.dot(x))) != s:
                mispts.append((x, s))
        return mispts[random.randrange(0, len(mispts))]

    def pla(self, save=False):
        # Initialize the weigths to zeros
        w = np.zeros(11)
        X, N = self.X, len(self.X)
        it = 0
        # Iterate until all points are correctly classified
        while self.classification_error(w) != 0:
            it += 1
            # Pick random misclassified point
            x, s = self.choose_miscl_point(w)
            # Update weights
            w += s * x
            if save:
                self.plot(vec=w)
                plt.title('N = %s, Iteration %s\n' \
                          % (str(N), str(it)))
                plt.savefig('p_N%s_it%s' % (str(N), str(it)),dpi=200, bbox_inches='tight')
        self.w = w
        self.histArray.append(it)
        print self.histArray
        x = len(self.histArray)
        return it

    def check_error(self, M, vec):
        check_pts = self.generate_points(M)
        return self.classification_error(vec, pts=check_pts)

    def hist(self):
        return self.histArray
        #for x in self.histArray:
           # plt.hist(x)
            #plt.show()

    def bins(self):
        return self.bins

def main(self):
    testing = None
    for x in range(0, 2):
        p = Perceptron(10)
        it = p.pla(save=False)
        testing = p.hist()
        bins = p.bins()
        print it
        p.plot()
    results = map(int, testing)
    plt.hist(results)
    plt.show()
    plt.hist(testing, bins, histtype='bar', rwidth=0.8)
    for x in testing:
        plt.hist(x)
        plt.show()

def main(self):
    i = 0
    while i < 100:
        i += 1
        p = Perceptron(1000)
        it = p.pla(save=False)
        p.plot()
main(1)