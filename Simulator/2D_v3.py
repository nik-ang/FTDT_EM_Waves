import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors
import scipy as sp
import cupy as cp
import time

class FTDT_EM(object):

	MU_0 = 1
	EPSILON_0 = 1

	def __init__(self, size_x: int, size_y: int, delta_x: float = 1, delta_y: float = 1, delta_t: float = 0.2):
		self.SIZE_X = size_x
		self.SIZE_Y = size_y
		self.DELTA_X = delta_x
		self.DELTA_Y = delta_y
		self.DELTA_T = delta_t
		self.time = 0
		#FIELDS
		self.Ez = cp.zeros((self.SIZE_X, self.SIZE_Y))
		self.Ex = cp.zeros((self.SIZE_X - 1, self.SIZE_Y))
		self.Ey = cp.zeros((self.SIZE_X, self.SIZE_Y - 1))
		self.Hz = cp.zeros((self.SIZE_X - 1, self.SIZE_Y - 1))
		self.Hx = cp.zeros((self.SIZE_X, self.SIZE_Y - 1))
		self.Hy = cp.zeros((self.SIZE_X - 1, self.SIZE_Y))
		#MATERIALS
		self.epsilon = cp.ones_like(self.Ez) # Eps_0 * Eps_r
		self.conductivity = cp.zeros_like(self.Ez)
		self.material_patches = []
		#TENSOR OPERATORS
		self.Hx_Ez = sp.sparse.coo_array(self.forwards_matrix(self.SIZE_Y - 1, self.SIZE_Y) * self.DELTA_T / self.DELTA_Y)
		self.Hy_Ez = sp.sparse.coo_array(self.forwards_matrix(self.SIZE_X - 1, self.SIZE_X) * self.DELTA_T / self.DELTA_X)
		self.Ez_Hy = sp.sparse.coo_array(self.backwards_matrix(self.SIZE_X, self.SIZE_X - 1) * self.DELTA_T / self.DELTA_X)
		self.Ez_Hx = sp.sparse.coo_array(self.backwards_matrix(self.SIZE_Y, self.SIZE_Y - 1) * self.DELTA_T / self.DELTA_Y)
		#CONSTANTS
		self.C_EField = (1 - (self.conductivity * self.DELTA_T) / (2 * self.epsilon)) /(1 + (self.conductivity * self.DELTA_T) / (2 * self.epsilon))
		self.C_HField = 1 / (self.epsilon * (1 + (self.conductivity * self.DELTA_T) / (2 * self.epsilon)))
		self.C_EzE = self.C_EField
		self.C_EzH = self.C_HField

	def forwards_matrix(self, rows, cols):
		F = np.zeros((rows, cols))
		index = min(rows, cols)
		for i in range(index):
			F[i, i] = -1
			if (i < cols-1):
				F[i, i + 1] = 1
		return F

	def backwards_matrix(self, rows, cols):
		B = np.zeros((rows, cols))
		index = min(rows, cols)
		for i in range(index):
			B[i, i] = 1
			if (i < rows - 1):
				B[i + 1, i] = -1

		B[0, 0] = 0
		B[-1, -1] = 0
		return B
	
	def generate_constants(self):
		self.C_EField = (1 - (self.conductivity * self.DELTA_T) / (2 * self.epsilon)) /(1 + (self.conductivity * self.DELTA_T) / (2 * self.epsilon))
		self.C_HField = 1 / (self.epsilon * (1 + (self.conductivity * self.DELTA_T) / (2 * self.epsilon)))
		self.C_EzE = self.C_EField
		self.C_EzH = self.C_HField
		
	def add_rectangle_material(self, xy, x_size, y_size, eps = 1, sigma = 0, color = '#ffffff', alpha = 0):
		x_start = xy[0]
		y_start = xy[1]

		if x_start < 0 or y_start < 0: return

		x_size = int(x_size)
		y_size = int(y_size)

		for i in range(x_start, min(x_start + x_size, self.SIZE_X)):
			for j in range(y_start, min(y_start + y_size, self.SIZE_Y)):
					self.epsilon[i, j] = eps
					self.conductivity[i, j] = sigma
		self.material_patches.append(plt.Rectangle(xy, x_size, y_size, color = color, alpha = alpha, zorder = 5))	
		self.generate_constants()

	def update(self):
		self.Hx = self.Hx - cp.asarray(self.Hx_Ez.tensordot(self.Ez.get(), axes=([1], [1])).T)
		self.Hy = self.Hy + cp.asarray(self.Hy_Ez.tensordot(self.Ez.get(), axes=([1], [0])))
		self.Ez = self.C_EzE * self.Ez + self.C_EzH * cp.asarray((self.Ez_Hy.tensordot(self.Hy.get(), axes=([1], [0])) - self.Ez_Hx.tensordot(self.Hx.get(), axes=([1], [1])).T))
		self.Ez[int(self.SIZE_X / 2), int(self.SIZE_Y / 2)] = 20*cp.exp(-((self.time - 10)**2) / 50) * cp.cos(0.5*self.time)
		self.time += self.DELTA_T

# CREATE NEW PALETTE WITH TRANSPARENCY
ncolors = 256 
color_array = plt.get_cmap('seismic')(range(ncolors))
color_array[:, -1] = np.abs(np.linspace(1.0, -1.0, ncolors))
map_object = matplotlib.colors.LinearSegmentedColormap.from_list(name="seismic_alpha", colors = color_array)
plt.colormaps.register(cmap=map_object)


### MAIN ===================================================================================================
simulator = FTDT_EM(1000, 1000, 1, 1, 0.2)
simulator.add_rectangle_material((0,277), 117, 100, 1.33, 0, "#191981", alpha=0.1)

fig, ax = plt.subplots()
plt.style.use('dark_background')
heatmap = ax.pcolorfast(simulator.Ez.T.get(), vmin=-1, vmax= 1, cmap = plt.colormaps['seismic_alpha'], rasterized=True, zorder = 1)

def init_plot():
	#ax.imshow(background_image)
	return heatmap,

ax.set_aspect(1)
for p in simulator.material_patches:
	ax.add_patch(p)

def update(frames):
	start = time.time()
	for _ in range(10):
		simulator.update()
	end = time.time()
	print(end - start)
	heatmap.set_array(simulator.Ez.T.get())
	return heatmap,

'''

fig, ax = plt.subplots()
x = np.arange(0, Ez[int(SIZE_X / 2), :].shape[0], 1)
line, = ax.plot(x, Ez[int(SIZE_X / 2), :])

def init_plot():
	ax.set_ylim(-1.2, 1.2)
	return line,


def update(frames):
	global Hx
	global Hy
	global Ez
	global time

	Hx = Hx - Hx_Ez.tensordot(Ez, axes=([1], [1])).T
	Hy = Hy + Hy_Ez.tensordot(Ez, axes=([1], [0]))
	Ez = C_EzE * Ez + C_EzH * (Ez_Hy.tensordot(Hy, axes=([1], [0])) - Ez_Hx.tensordot(Hx, axes=([1], [1])).T)
	#Ez += source(source_mesh_x, source_mesh_y, time) - source(source_mesh_x, source_mesh_y, time - DELTA_T)
	
	Ez[:, int(SIZE_Y / 2)] += 0.1*np.exp(-((time - 100)**2) / 1000)
	#Ez[:, int(SIZE_Y / 2)] += np.cos(0.25*time) - np.cos(0.25*(time - DELTA_T))

	time += DELTA_T

	line.set_data(x, Ez[int(SIZE_X / 2), :])
	return line,
'''

ani = animation.FuncAnimation(fig, update, frames = np.arange(0, 2, 1), interval = 0.001, init_func = init_plot, blit = True)
plt.show()












