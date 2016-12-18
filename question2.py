import numpy as np

def question2(N, dvc, delta, epsilon):
    Ni = N
    for i in range (1, 1000):
        Ni = (8/epsilon ** 2)*np.log((4*(2*N ** dvc)+ 1)/(epsilon))
        N = Ni
        print Ni


def main():
    question2(1000, 10, 0.05, 0.05)

main()

