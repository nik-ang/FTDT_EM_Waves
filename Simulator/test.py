import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy as sp
import scipy.constants
import scipy.sparse

SIZE_X = 1000
SIZE_Y = 1000

DELTA_X = 0.1
DELTA_Y = 0.1
DELTA_T = 0.1

MU_0 = scipy.constants.mu_0
EPSILON_0 = scipy.constants.epsilon_0

time = 0


Ez = np.zeros((SIZE_X, SIZE_Y))
source_mesh_x, source_mesh_y = np.meshgrid(np.arange(- SIZE_X / 2, SIZE_X / 2), np.arange(- SIZE_Y / 2, SIZE_Y / 2), indexing="ij")

def source(x, y, t):
	return 10*np.exp((-(x**2 + y**2))/10*(t-10)**2)

fig, ax = plt.subplots()
heatmap = ax.pcolorfast(Ez.T, vmin=0, vmax = 15, cmap = 'plasma', rasterized=False)

def update(frames):

	global time
	global Ez
	
	Ez += source(source_mesh_x, source_mesh_y, time) - source(source_mesh_x, source_mesh_y, time - DELTA_T)
	time += DELTA_T

	heatmap.set_array(Ez.T)
	return heatmap,

ani = animation.FuncAnimation(fig, update, frames = np.arange(0, 100, 1), interval = 1, blit = True)

plt.show()