from typing import List
import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors
import scipy as sp
import time
import concurrent.futures

class EM_Source(object):
	
	def __init__(self, indices, func, amp_phase = None):
		self.indices: tuple[np.ndarray] = indices
		self.func: function = func
		self.amplitude_phase: np.ndarray
		if amp_phase is None:
			self.amplitude_phase = np.ones_like(indices[0])
		else:
			self.amplitude_phase = amp_phase
			

class FTDT_EM(object):

	MU_0 = 1
	EPSILON_0 = 1

	def __init__(self, size_x: int, size_y: int, delta_x: float = 1, delta_y: float = 1, delta_t: float = 0.2):
		#PARAMETERS
		self.SIZE_X = size_x
		self.SIZE_Y = size_y
		self.DELTA_X = delta_x
		self.DELTA_Y = delta_y
		self.DELTA_T = delta_t
		self.time = 0
		#FIELDS
		self.Ez = np.zeros((self.SIZE_X, self.SIZE_Y))
		self.Ex = np.zeros((self.SIZE_X - 1, self.SIZE_Y))
		self.Ey = np.zeros((self.SIZE_X, self.SIZE_Y - 1))
		self.Hz = np.zeros((self.SIZE_X - 1, self.SIZE_Y - 1))
		self.Hx = np.zeros((self.SIZE_X, self.SIZE_Y - 1))
		self.Hy = np.zeros((self.SIZE_X - 1, self.SIZE_Y))
		#MATERIALS
		self.epsilon = np.ones_like(self.Ez) # Eps_0 * Eps_r
		self.conductivity = np.zeros_like(self.Ez)
		self.material_patches = []
		#SOURCES
		self.sources: List[EM_Source] = []
		#TENSOR OPERATORS
		self.Hx_Ez = sp.sparse.coo_array(self.__forwards_matrix(self.SIZE_Y - 1, self.SIZE_Y) * self.DELTA_T / self.DELTA_Y)
		self.Hy_Ez = sp.sparse.coo_array(self.__forwards_matrix(self.SIZE_X - 1, self.SIZE_X) * self.DELTA_T / self.DELTA_X)
		self.Ez_Hy = sp.sparse.coo_array(self.__backwards_matrix(self.SIZE_X, self.SIZE_X - 1) * self.DELTA_T / self.DELTA_X)
		self.Ez_Hx = sp.sparse.coo_array(self.__backwards_matrix(self.SIZE_Y, self.SIZE_Y - 1) * self.DELTA_T / self.DELTA_Y)
		#CONSTANTS
		self.C_EField = (1 - (self.conductivity * self.DELTA_T) / (2 * self.epsilon)) /(1 + (self.conductivity * self.DELTA_T) / (2 * self.epsilon))
		self.C_HField = 1 / (self.epsilon * (1 + (self.conductivity * self.DELTA_T) / (2 * self.epsilon)))
		self.C_EzE = self.C_EField
		self.C_EzH = self.C_HField
		#MULTITHREADING
		self.thread_executor = concurrent.futures.ThreadPoolExecutor(max_workers = 2)
		self.futures: List[concurrent.futures.Future] = [None, None]

	def __forwards_matrix(self, rows, cols):
		F = np.zeros((rows, cols))
		index = min(rows, cols)
		for i in range(index):
			F[i, i] = -1
			if (i < cols-1):
				F[i, i + 1] = 1
		return F

	def __backwards_matrix(self, rows, cols):
		B = np.zeros((rows, cols))
		index = min(rows, cols)
		for i in range(index):
			B[i, i] = 1
			if (i < rows - 1):
				B[i + 1, i] = -1

		B[0, 0] = 0
		B[-1, -1] = 0
		return B
	
	def __generate_constants(self):
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
		self.__generate_constants()

	def point_source(self, point, function):
		index = (point[0], point[1])
		self.sources.append(EM_Source(index, function))

	def line_source(self, point_a, point_b, function):
		rr, cc, weight = ski.draw.line_aa(point_a[0], point_a[1], point_b[0], point_b[1])
		index = (rr, cc)
		self.sources.append(EM_Source(index, function, weight))

	def line_source_fade(self, point_a, point_b, function):
		rr, cc, weight = ski.draw.line_aa(point_a[0], point_a[1], point_b[0], point_b[1])
		index = (rr, cc)
		x0 = 0.5*(point_a[0] + point_b[0])
		y0 = 0.5*(point_a[1] + point_b[1])
		distances = np.power(rr - x0, 2) + np.power(cc - y0, 2)
		weight = weight * np.exp( - distances / (1000))
		self.sources.append(EM_Source(index, function, weight))

	def gaussian_beam(self, starting_point, direction, width, frequency):
		pass

	def phase_array(self, starting_point, direciton, width, frequency, phase):
		pass

	def __updateHx(self):
		self.Hx = self.Hx - self.Hx_Ez.tensordot(self.Ez, axes=([1], [1])).T

	def __updateHy(self):
		self.Hy = self.Hy + self.Hy_Ez.tensordot(self.Ez, axes=([1], [0]))

	def __updateEz(self):
		self.Ez = self.C_EzE * self.Ez + self.C_EzH * (self.Ez_Hy.tensordot(self.Hy, axes=([1], [0])) - self.Ez_Hx.tensordot(self.Hx, axes=([1], [1])).T)

	def update(self):
		self.futures[0] = self.thread_executor.submit(self.__updateHx)
		self.futures[1] = self.thread_executor.submit(self.__updateHy)
		concurrent.futures.wait(self.futures, timeout=None, return_when=concurrent.futures.FIRST_EXCEPTION)
		self.__updateEz()
		for source in self.sources:
			self.Ez[source.indices] = source.amplitude_phase * source.func(self.time)
		self.time += self.DELTA_T

# CREATE NEW PALETTE WITH TRANSPARENCY
ncolors = 256 
color_array = plt.get_cmap('seismic')(range(ncolors))
color_array[:, -1] = np.abs(np.linspace(1.0, -1.0, ncolors))
map_object = matplotlib.colors.LinearSegmentedColormap.from_list(name="seismic_alpha", colors = color_array)
plt.colormaps.register(cmap=map_object)


### MAIN ===================================================================================================
simulator = FTDT_EM(500, 500, 1, 1, 0.2)
simulator.add_rectangle_material((0,0), simulator.SIZE_X, simulator.SIZE_Y/2, 1.33, 0.01, "#191981", alpha=0.1)
#simulator.point_source(point=(250, 250), function = lambda t: 10*np.exp(-((t - 10)**2) / 50) * np.cos(0.5*t))
#simulator.point_source(point=(100, 400), function = lambda t: 10*np.exp(-((t - 10)**2) / 50) * np.cos(0.5*t))
simulator.line_source_fade(point_a=(250, 250), point_b=(100, 400), function = lambda t: 10 * np.cos(0.2*t))

fig, ax = plt.subplots()
plt.style.use('dark_background')
heatmap = ax.pcolorfast(simulator.Ez.T, vmin=-1, vmax= 1, cmap = plt.colormaps['seismic_alpha'], rasterized=True, zorder = 1)

def init_plot():
	return heatmap,

ax.set_aspect(1)
for p in simulator.material_patches:
	ax.add_patch(p)

def update(frames):
	#start = time.time()
	for _ in range(5):
		simulator.update()
	#end = time.time()
	#print(end - start)
	
	heatmap.set_array(simulator.Ez.T)
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












