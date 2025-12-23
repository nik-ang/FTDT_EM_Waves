import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors
import scipy as sp
import FDTD_v2

# CREATE NEW PALETTE WITH TRANSPARENCY
ncolors = 256 
color_array = plt.get_cmap('seismic')(range(ncolors))
color_array[:, -1] = np.abs(np.linspace(1.0, -1.0, ncolors))
map_object = matplotlib.colors.LinearSegmentedColormap.from_list(name="seismic_alpha", colors = color_array)
plt.colormaps.register(cmap=map_object)


### MAIN ===================================================================================================
DELTA_T = 5E-17
DELTA_X = DELTA_T * np.sqrt(2) * sp.constants.c	
DELTA_Y = DELTA_X
simulator = FDTD_v2.FTDT_EM(500, 500, DELTA_X, DELTA_Y, DELTA_T, mur_abc = True)

epsilon_r = 5
sigma = 0


simulator.add_rectangle_material((250,0), 250, 200, 2, sigma, "#191981", alpha=0.1)
simulator.add_rectangle_material((300,300), 100, 100, 1, 0.2e-3 / DELTA_X, "#89911B", alpha=0.1)
#simulator.point_source(point=(400, 250), function = lambda t: 10*np.exp(-((t - 10)**2) / 50) * np.cos(0.5*t))
#simulator.point_source(point=(100, 400), function = lambda t: 10*np.exp(-((t - 10)**2) / 50) * np.cos(0.5*t))

frequency = 500E12
N = int(np.round(1 / (simulator.DELTA_T * frequency)))

# 2pi f t ==> 2pi f N Dt = M 2pi ==> N = M / (f Dt)
def source_func(t):
	return 50 * np.cos(2*np.pi * frequency * t) * (1 - np.exp(-(t / 2E-15)**2))

### Sanity check

def source_func_test(t):
	return 5 * np.cos(2*np.pi * frequency * t)

test_arr = np.arange(0, N*simulator.DELTA_T, simulator.DELTA_T)[:-1]
test_sum = np.sum(np.vectorize(source_func_test)(test_arr))
print(test_arr)
print(test_sum)
assert(np.isclose(0, test_sum))




#simulator.gaussian_beam(starting_point = (100, 100), direction = (1, 1), width = 40, function = source_func)
simulator.point_source((100, 400), function=source_func)


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
	for _ in range(2):
		simulator.update()
	#end = time.time()
	#print(end - start)
	
	heatmap.set_array(simulator.Ez.T)
	return heatmap,

ani = animation.FuncAnimation(fig, update, frames = np.arange(0, 2, 1), interval = 0.001, init_func = init_plot, blit = True)
plt.show()

print(f"Elapsed Time: {simulator.time}")

## MEASUREMENTS ============================================================================================

fig, (ax0, ax1) = plt.subplots(1, 2)
field_slice = simulator.Ez.T

fourier_2D = np.fft.fft2(field_slice)
fourier_2D = np.abs(np.fft.fftshift(fourier_2D))
ax0.pcolorfast(fourier_2D, cmap = plt.colormaps['turbo'])
ax1.pcolorfast(field_slice, cmap = plt.colormaps['seismic'])
ax0.set_aspect(1)
ax1.set_aspect(1)

plt.show()