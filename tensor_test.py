import numpy as np
import scipy as sp

class updater(object):
    def __init__(self, V):
        self.M, self.N = V.shape
        self.phi = self.generate_phi(self.M)
        self.theta = self.generate_theta(self.N)

    def generate_phi(self, M):
        phi = np.array([np.zeros(M) for _ in range(M - 1)])
        for i in range(M - 1):
            phi[i][i] = 0
            phi[i][i + 1] = 1
        return sp.sparse.coo_array(phi)
    
    def generate_theta(self, N):
        theta = np.array([np.zeros(N) for _ in range(N - 1)])
        for i in range(N - 1):
            theta[i][i] = - 1
            theta[i][i + 1] = 1
        return sp.sparse.coo_array(theta)


    def update(self, V):
        M, N  = V.shape
        H = np.array([np.zeros(M - 1) for _ in range(N - 1)])

        for i in range(N - 1):
            for j in range(M - 1):
                H[i][j] = V[i + 1][j + 1] - V[i][j + 1]

        print(H)
        return H
    
    def update_tensor(self, V):
        V = sp.sparse.coo_array(V)

        H = self.phi.tensordot(V, axes=([1], [1]))
        H = H.tensordot(self.theta, axes=([1], [1])).toarray()
        print(H)
        return H    


def update_tensor(V):
    M, N = V.shape

    theta = np.array([np.zeros(N) for _ in range(N - 1)])
    for i in range(N - 1):
        theta[i][i] = - 1
        theta[i][i + 1] = 1

    phi = np.array([np.zeros(M) for _ in range(M - 1)])
    for i in range(M - 1):
        phi[i][i] = 0
        phi[i][i + 1] = 1


    


def main():
    SIZE = 1000
    E = np.arange(0, SIZE * SIZE, 1)
    E = E.reshape(SIZE, SIZE)
    updater_obj = updater(E)
    print(E)
    updater_obj.update(E)
    updater_obj.update_tensor(E)
main()