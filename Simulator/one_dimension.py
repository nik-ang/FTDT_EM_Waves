import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy
import scipy.constants
import scipy.sparse

DELTA_X = 1
DELTA_T = 0.2
DELTA_RATIO = DELTA_T / DELTA_X
MU_0 = scipy.constants.mu_0
EPSILON_0 = scipy.constants.epsilon_0
ETA_0 = 377
COURANT_E = 1 / EPSILON_0 * (DELTA_T / DELTA_X)
COURANT_H = 1 / MU_0 * (DELTA_T / DELTA_X)

MAX_X = 500

time = 0
x = np.arange(0, MAX_X, DELTA_X)
ez = np.zeros(x.shape[0])
hy = np.zeros(x.shape[0])

# PLOT
fig, ax = plt.subplots()
line, = ax.plot(x, ez)

def init_plot():
	ax.set_xlim(0, 200)
	ax.set_ylim(-1.2, 1.2)
	return line,

def generate_H_matrix():
	N = len(x)
	H = np.identity(N) * -1
	H[N-1, N-1] = 0
	for i in range(N - 1):
		H[i, i + 1] = 1

	return H

def generate_E_matrix():
	N = len(x)
	E = np.identity(N)
	E[0, 0] = 1
	for i in range(1, N):
		E[i, i-1] = -1

	return E

def backwards_matrix(rows, cols):
	B = np.zeros((rows, cols))
	index = min(rows, cols)
	for i in range(index):
		B[i, i] = 1
		if (i < rows - 1):
			B[i + 1, i] = -1

	B[0, 0] = 0
	B[-1, -1] = 0
	return B


H = scipy.sparse.csr_matrix(generate_H_matrix() * DELTA_T / DELTA_X)
E = scipy.sparse.csr_matrix(generate_E_matrix() * DELTA_T / DELTA_X)

def update(frame):
	global time
	global hy
	global ez

	time += DELTA_T
	hy += H @ ez
	ez += E @ hy
	#ez[int(len(x) / 2)] += np.exp(-((time - 50)**2) / 400) * np.cos(time)
	ez[int(len(x) / 2)] += 0.1*np.exp(-((time - 10)**2) / 10)
	line.set_data(x, ez)
	return line,

ani = animation.FuncAnimation(fig, update, frames = np.arange(0, 100, 1), interval = 10	, init_func = init_plot, blit = True)
plt.show()