import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors
import scipy as sp
import scipy.constants


SIZE_X = 400
SIZE_Y = 400

DELTA_X = 1
DELTA_Y = 1
DELTA_T = 0.2

#MU_0 = scipy.constants.mu_0
#EPSILON_0 = scipy.constants.epsilon_0

MU_0 = 1
EPSILON_0 = 1

time = 0

# OPERATOR TYPES
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

	B[0, 0] = 0
	B[-1, -1] = 0
	return B

# FIELDS
Ez = np.zeros((SIZE_X, SIZE_Y))
Ex = np.zeros((SIZE_X - 1, SIZE_Y))
Ey = np.zeros((SIZE_X, SIZE_Y - 1))
Hz = np.zeros((SIZE_X - 1, SIZE_Y - 1))
Hx = np.zeros((SIZE_X, SIZE_Y - 1))
Hy = np.zeros((SIZE_X - 1, SIZE_Y))

# MATERIALS
epsilon = np.ones_like(Ez) # Eps_0 * Eps_r
conductivity = np.zeros_like(Ez)

material_patches = []

def add_rectangle_material(xy, x_size, y_size, eps = 1, sigma = 0, color = '#ffffff', alpha = 0):
	x_start = xy[0]
	y_start = xy[1]

	if x_start < 0 or y_start < 0: return

	x_size = int(x_size)
	y_size = int(y_size)

	for i in range(x_start, min(x_start + x_size, SIZE_X)):
		for j in range(y_start, min(y_start + y_size, SIZE_Y)):
				epsilon[i, j] = eps
				conductivity[i, j] = sigma
	material_patches.append(plt.Rectangle(xy, x_size, y_size, color = color, alpha = alpha, zorder = 5))

# SOURCE?
source_mesh_x, source_mesh_y = np.meshgrid(np.arange(- SIZE_X / 2, SIZE_X / 2), np.arange(- SIZE_Y / 2, SIZE_Y / 2), indexing="ij")

def source(x, y, t):
	return 10*np.exp((-(x**2 + y**2))/100*(t-10)**2)

# OPERATORS
Hx_Ez = sp.sparse.coo_array(forwards_matrix(SIZE_Y - 1, SIZE_Y) * DELTA_T / DELTA_Y)
Hy_Ez = sp.sparse.coo_array(forwards_matrix(SIZE_X - 1, SIZE_X) * DELTA_T / DELTA_X)
Ez_Hy = sp.sparse.coo_array(backwards_matrix(SIZE_X, SIZE_X - 1) * DELTA_T / DELTA_X)
Ez_Hx = sp.sparse.coo_array(backwards_matrix(SIZE_Y, SIZE_Y - 1) * DELTA_T / DELTA_Y)


# CREATE NEW PALETTE WITH TRANSPARENCY
ncolors = 256 
color_array = plt.get_cmap('seismic')(range(ncolors))
color_array[:, -1] = np.abs(np.linspace(1.0, -1.0, ncolors))
map_object = matplotlib.colors.LinearSegmentedColormap.from_list(name="seismic_alpha", colors = color_array)
plt.colormaps.register(cmap=map_object)


fig, ax = plt.subplots()
plt.style.use('dark_background')
heatmap = ax.pcolorfast(Ez.T, vmin=-1, vmax= 1, cmap = plt.colormaps['seismic_alpha'], rasterized=True, zorder = 1)

### CODE GOES HERE ===================================================================================================
#add_rectangle_material((0,0), SIZE_X, SIZE_Y/4, 2, 0)
background_image = plt.imread('GPM1.png')[:SIZE_X, :SIZE_Y]
add_rectangle_material((0,277), 117, 100, 1.33, 0, "#191981", alpha=0.1)
#add_rectangle_material((300,200), 100, 40, 2, 0)


### =================================================================================================================

#GENERATE CONSTANTS
C_EField = (1 - (conductivity * DELTA_T) / (2 * epsilon)) /(1 + (conductivity * DELTA_T) / (2 * epsilon))
C_HField = 1 / (epsilon * (1 + (conductivity * DELTA_T) / (2 * epsilon)))

C_EzE = C_EField
C_EzH = C_HField


def init_plot():
	#ax.imshow(background_image)
	return heatmap,

ax.set_aspect(1)
for p in material_patches:
	ax.add_patch(p)

def update(frames):
	global Hx
	global Hy
	global Ez
	global time

	Hx = Hx - Hx_Ez.tensordot(Ez, axes=([1], [1])).T
	Hy = Hy + Hy_Ez.tensordot(Ez, axes=([1], [0]))
	Ez = C_EzE * Ez + C_EzH * (Ez_Hy.tensordot(Hy, axes=([1], [0])) - Ez_Hx.tensordot(Hx, axes=([1], [1])).T)
	#Ez += source(source_mesh_x, source_mesh_y, time) - source(source_mesh_x, source_mesh_y, time - DELTA_T)
	Ez[int(SIZE_X / 2), int(SIZE_Y / 2)] = 20*np.exp(-((time - 10)**2) / 50) * np.cos(0.5*time)
	#Ez[int(SIZE_X / 2), :] += 1*np.exp(-((time - 1)**2) / 5) * np.cos(0*time)
	#Ez[:, 100] += 0.1*np.exp(-((time - 100)**2) / 200)

	time += DELTA_T
	material_patches[0].center = (0, 0)

	heatmap.set_array(Ez.T)
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












