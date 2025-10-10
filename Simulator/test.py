import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy as sp
import scipy.constants
import scipy.sparse

A = np.array([[1, 3, 5, 0],
		   	[54, 23, 65, 10],
		   	[65,762,9877,9]])

B = np.array([[54, 6950, 29, 29],
		   	[68, 49, 103, 139],
		   	[760, 101, 120, 7]])

C1 = np.tensordot(A, B, axes = (1, 1))
C2 = np.tensordot(B, A, axes = (1, 1)).T

print(C1 - C2)

