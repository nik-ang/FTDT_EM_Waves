from typing import List
import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors
import scipy as sp
import time
import concurrent.futures


def forwards_matrix(rows, cols):
	F = np.zeros((rows, cols))
	index = min(rows, cols)
	for i in range(index):
		F[i, i] = -1
		if (i < cols-1):
			F[i, i + 1] = 1
	return F

def backwards_matrix(rows, cols):
	B = np.zeros((rows, cols))
	index = min(rows, cols)
	for i in range(index):
		B[i, i] = 1
		if (i < rows - 1):
			B[i + 1, i] = -1
	return B


SIZE = 5
F = forwards_matrix(SIZE, SIZE)
B = backwards_matrix(SIZE, SIZE)
C = np.tensordot(B, F, axes=([1], [0]))


print(np.matmul(B.T, B))
eigenvaluesB, eigenvectorsB = np.linalg.eig(np.matmul(B.T, B))
eigenvaluesF, eigenvectorsF = np.linalg.eig(np.matmul(F, F.T))
eigenvaluesC, eigenvectorsC = np.linalg.eig(np.matmul(C, C.T))
print(np.sqrt(np.max(eigenvaluesB)))
print(np.sqrt(np.max(eigenvaluesF)))
print(np.sqrt(np.max(eigenvaluesC)))
