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
	
	def __init__(self, indices, func, amp_phase = None, time_shift = None):
		self.indices: tuple[np.ndarray] = indices
		self.func: function = np.vectorize(func)
		self.amplitude_phase: np.ndarray
		self.time_shift: np.ndarray
		if amp_phase is None:
			self.amplitude_phase = np.ones_like(indices[0])
		else:
			self.amplitude_phase = amp_phase
		if time_shift is None:
			self.time_shift = np.zeros_like(indices[0])
		else:
			self.time_shift = time_shift

class FTDT_EM(object):

	MU_0 = sp.constants.mu_0
	EPSILON_0 = sp.constants.epsilon_0
	#MU_0 = 1
	#EPSILON_0 = 1

	def __init__(self, size_x: int, size_y: int, delta_x: float = 1, delta_y: float = 1, delta_t: float = 0.2, mur_abc = True):
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
		self.epsilon = np.ones_like(self.Ez) * self.EPSILON_0
		self.conductivity = np.zeros_like(self.Ez)
		self.material_patches = []
		#SOURCES
		self.sources: List[EM_Source] = []
		#TENSOR OPERATORS
		self.Hx_Ez = sp.sparse.coo_array(self.__forwards_matrix(self.SIZE_Y - 1, self.SIZE_Y) * self.DELTA_T / self.DELTA_Y)
		self.Hy_Ez = sp.sparse.coo_array(self.__forwards_matrix(self.SIZE_X - 1, self.SIZE_X) * self.DELTA_T / self.DELTA_X)
		self.Ez_Hy = sp.sparse.coo_array(self.__backwards_matrix(self.SIZE_X, self.SIZE_X - 1) * self.DELTA_T / self.DELTA_X)
		self.Ez_Hx = sp.sparse.coo_array(self.__backwards_matrix(self.SIZE_Y, self.SIZE_Y - 1) * self.DELTA_T / self.DELTA_Y)
		# PROPAGATION CONSTANTS
		self.C_EzE = (1 - (self.conductivity * self.DELTA_T) / (2 * self.epsilon)) /(1 + (self.conductivity * self.DELTA_T) / (2 * self.epsilon))
		self.C_EzH = 1 / (self.epsilon * (1 + (self.conductivity * self.DELTA_T) / (2 * self.epsilon)))
		#BOUNDARY CONDITION CONSTANTS
		self.mur_abc = mur_abc
		self.C0x0 = (1 / np.sqrt(self.epsilon[0, :] * self.MU_0) * self.DELTA_T - self.DELTA_X) / ((1 / np.sqrt(self.epsilon[0, :] * self.MU_0) * self.DELTA_T + self.DELTA_X))
		self.CNx0 = (1 / np.sqrt(self.epsilon[self.SIZE_X - 1, :] * self.MU_0) * self.DELTA_T - self.DELTA_X) / ((1 / np.sqrt(self.epsilon[self.SIZE_X - 1, :] * self.MU_0) * self.DELTA_T + self.DELTA_X))
		self.C0y0 = (1 / np.sqrt(self.epsilon[:, 0] * self.MU_0) * self.DELTA_T - self.DELTA_Y) / ((1 / np.sqrt(self.epsilon[:, 0] * self.MU_0) * self.DELTA_T + self.DELTA_Y))
		self.CNy0 = (1 / np.sqrt(self.epsilon[:, self.SIZE_Y - 1] * self.MU_0) * self.DELTA_T - self.DELTA_Y) / ((1 / np.sqrt(self.epsilon[:, self.SIZE_Y - 1] * self.MU_0) * self.DELTA_T + self.DELTA_Y))
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
		self.C_EzE = (1 - (self.conductivity * self.DELTA_T) / (2 * self.epsilon)) /(1 + (self.conductivity * self.DELTA_T) / (2 * self.epsilon))
		self.C_EzH = 1 / (self.epsilon * (1 + (self.conductivity * self.DELTA_T) / (2 * self.epsilon)))
		self.C0x0 = (1 / np.sqrt(self.epsilon[0, :] * self.MU_0) * self.DELTA_T - self.DELTA_X) / ((1 / np.sqrt(self.epsilon[0, :] * self.MU_0) * self.DELTA_T + self.DELTA_X))
		self.CNx0 = (1 / np.sqrt(self.epsilon[self.SIZE_X - 1, :] * self.MU_0) * self.DELTA_T - self.DELTA_X) / ((1 / np.sqrt(self.epsilon[self.SIZE_X - 1, :] * self.MU_0) * self.DELTA_T + self.DELTA_X))
		self.C0y0 = (1 / np.sqrt(self.epsilon[:, 0] * self.MU_0) * self.DELTA_T - self.DELTA_Y) / ((1 / np.sqrt(self.epsilon[:, 0] * self.MU_0) * self.DELTA_T + self.DELTA_Y))
		self.CNy0 = (1 / np.sqrt(self.epsilon[:, self.SIZE_Y - 1] * self.MU_0) * self.DELTA_T - self.DELTA_Y) / ((1 / np.sqrt(self.epsilon[:, self.SIZE_Y - 1] * self.MU_0) * self.DELTA_T + self.DELTA_Y))
		
	def add_rectangle_material(self, xy, x_size, y_size, eps = 1, sigma = 0, color = '#ffffff', alpha = 0):
		x_start = xy[0]
		y_start = xy[1]

		if x_start < 0 or y_start < 0: return

		x_size = int(x_size)
		y_size = int(y_size)

		for i in range(x_start, min(x_start + x_size, self.SIZE_X)):
			for j in range(y_start, min(y_start + y_size, self.SIZE_Y)):
					self.epsilon[i, j] = self.EPSILON_0 * eps
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

	def gaussian_beam(self, starting_point, direction, width, function):
		line_direction = np.array([-1 * direction[1], direction[0]])
		starting_point = np.array([starting_point[0], starting_point[1]])
		point_a = starting_point + 2 * width * line_direction
		point_b = starting_point - 2 * width * line_direction
		rr, cc, weight = ski.draw.line_aa(point_a[0], point_a[1], point_b[0], point_b[1])
		index = (rr, cc)
		x0 = 0.5*(point_a[0] + point_b[0])
		y0 = 0.5*(point_a[1] + point_b[1])
		distances = np.power(rr - x0, 2) + np.power(cc - y0, 2)
		weight = weight * np.exp( - distances / (width ** 2))
		self.sources.append(EM_Source(index, function, weight))

	def phased_array(self, starting_point, direction, width, function):
		line_direction = np.array([-1 * direction[1], direction[0]])
		starting_point = np.array([starting_point[0], starting_point[1]])
		point_a = starting_point + 2 * width * line_direction
		point_b = starting_point - 2 * width * line_direction
		rr, cc, weight = ski.draw.line_aa(point_a[0], point_a[1], point_b[0], point_b[1])
		index = (rr, cc)
		x0 = 0.5*(point_a[0] + point_b[0])
		y0 = 0.5*(point_a[1] + point_b[1])
		distances = np.power(rr - x0, 2) + np.power(cc - y0, 2)
		weight = weight * np.exp( - distances / (width ** 2))
		t_shifts = np.linspace(0, 4*np.pi, rr.shape[0])
		self.sources.append(EM_Source(index, function, weight, t_shifts))


	def __updateHx(self):
		self.Hx = self.Hx - self.Hx_Ez.tensordot(self.Ez, axes=([1], [1])).T * (1 / self.MU_0)

	def __updateHy(self):
		self.Hy = self.Hy + self.Hy_Ez.tensordot(self.Ez, axes=([1], [0])) * (1 / self.MU_0)

	def __updateEz(self):
		self.Ez = self.C_EzE * self.Ez + self.C_EzH * (self.Ez_Hy.tensordot(self.Hy, axes=([1], [0])) - self.Ez_Hx.tensordot(self.Hx, axes=([1], [1])).T)

	def update(self):
		self.futures[0] = self.thread_executor.submit(self.__updateHx)
		self.futures[1] = self.thread_executor.submit(self.__updateHy)
		concurrent.futures.wait(self.futures, timeout=None, return_when=concurrent.futures.FIRST_EXCEPTION)
		if self.mur_abc: 
			E_x_boundary = np.array([self.Ez[0,:], self.Ez[1, :], self.Ez[self.SIZE_X - 1, :], self.Ez[self.SIZE_X - 2, :]])
			E_y_boundary = np.array([self.Ez[:,0], self.Ez[:, 1], self.Ez[:, self.SIZE_Y - 1], self.Ez[:, self.SIZE_Y - 2]])
			self.__updateEz()
			self.Ez[0, :] = E_x_boundary[1] + self.C0x0 * (self.Ez[1, :] - E_x_boundary[0])
			self.Ez[self.SIZE_X - 1, :] = E_x_boundary[-1] + self.CNx0 * (self.Ez[self.SIZE_X - 2, :] - E_x_boundary[-2])
			self.Ez[:, 0] = E_y_boundary[1] + self.C0y0 * (self.Ez[:, 1] - E_y_boundary[0])
			self.Ez[:, self.SIZE_Y - 1] = E_y_boundary[-1] + self.CNy0 * (self.Ez[:, self.SIZE_Y - 2] - E_y_boundary[-2])
		else:
			self.__updateEz()
		for source in self.sources:
			self.Ez[source.indices] = np.real(source.amplitude_phase * source.func(self.time - source.time_shift))
		self.time += self.DELTA_T

# CREATE NEW PALETTE WITH TRANSPARENCY
ncolors = 256 
color_array = plt.get_cmap('seismic')(range(ncolors))
color_array[:, -1] = np.abs(np.linspace(1.0, -1.0, ncolors))
map_object = matplotlib.colors.LinearSegmentedColormap.from_list(name="seismic_alpha", colors = color_array)
plt.colormaps.register(cmap=map_object)


### MAIN ===================================================================================================
DELTA_T = 0.2
DELTA_X = DELTA_T * 5 * sp.constants.c
DELTA_Y = DELTA_X
simulator = FTDT_EM(500, 500, DELTA_X, DELTA_Y, DELTA_T, mur_abc=True)
simulator.add_rectangle_material((0,0), simulator.SIZE_X, simulator.SIZE_Y / 2, 5, 0, "#191981", alpha=0.1)
#simulator.point_source(point=(400, 250), function = lambda t: 10*np.exp(-((t - 10)**2) / 50) * np.cos(0.5*t))
#simulator.point_source(point=(100, 400), function = lambda t: 10*np.exp(-((t - 10)**2) / 50) * np.cos(0.5*t))

def source_func(t):
	return 10 * np.cos(0.2*t)

#simulator.gaussian_beam(starting_point = (100, 400), direction = (1, -1), width = 40, function = source_func)
simulator.phased_array(starting_point = (100, 400), direction = (1, -1), width = 40, function = source_func)




### PLOTY STUFF =============================================================================================
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

ani = animation.FuncAnimation(fig, update, frames = np.arange(0, 2, 1), interval = 0.001, init_func = init_plot, blit = True)
plt.show()

## MEASUREMENTS ============================================================================================

fig, (ax0, ax1) = plt.subplots(1, 2)
field_slice = simulator.Ez.T

fourier_2D = np.fft.fft2(field_slice)
fourier_2D = np.abs(np.fft.fftshift(fourier_2D))
ax0.pcolorfast(fourier_2D, cmap = plt.colormaps['turbo'])
ax1.pcolorfast(field_slice, cmap = plt.colormaps['seismic'])
ax0.set_aspect(1)
ax1.set_aspect(1)

print(f'Total Parseval Energy: {1 / (500 * 500) * np.sum(np.power(fourier_2D, 2))}')
print(f'Total Field Energy: {np.sum(np.power(field_slice, 2))}')

refraction_angle = np.arcsin(np.sin(np.pi/4) * np.sqrt(1 / 2))
fresnel_T_te = (np.cos(np.pi / 4) - np.sqrt(2) * np.cos(refraction_angle)) / (np.cos(np.pi / 4) + np.sqrt(2) * np.cos(refraction_angle))
fresnel_R_te = (2 * np.cos(np.pi / 4)) / (np.cos(np.pi / 4) + np.sqrt(2) * np.cos(refraction_angle))

print(f'Theoretical refraction angle: {refraction_angle * 180 / np.pi} degrees')
print(f'Theoretical Fresnel Transmission Coefficient: {fresnel_T_te}')
print(f'Theoretical Fresnel Reflection Coefficient: {fresnel_R_te}')
print(f'E field boundary condition {simulator.Ez.T[250, 249]}')

plt.show()