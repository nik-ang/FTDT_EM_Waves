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

epsilon_r = 2.5
sigma = 0

simulator.add_rectangle_material((0,0), simulator.SIZE_X, simulator.SIZE_Y / 2, epsilon_r, sigma, "#191981", alpha=0.1)

frequency = 500E12
N = 2*int(np.round(1 / (simulator.DELTA_T * frequency)))

# 2pi f t ==> 2pi f N Dt = M 2pi ==> N = M / (f Dt)
def source_func(t):
	return 5 * np.cos(2*np.pi * frequency * t) * (1 - np.exp(-(t / 2E-15)**2))

### Sanity check

def source_func_test(t):
	return 5 * np.cos(2*np.pi * frequency * t)

test_arr = np.arange(0, N*simulator.DELTA_T, simulator.DELTA_T)[:-1]
test_sum = np.sum(np.vectorize(source_func_test)(test_arr))
print(test_sum)
assert(np.isclose(0, test_sum))


def arc_position(distance, angle):
	center = np.array([simulator.SIZE_X / 2, simulator.SIZE_Y / 2])
	pos = center + distance * np.array([-np.sin(angle), np.cos(angle)])
	direction = (pos - center)
	direction = direction / np.sqrt(np.dot(direction, direction))
	print(pos)
	print(direction)
	return (np.round(pos).astype(np.int32), direction)

def up_snapshot(sim):
	return sim.Ez[:, sim.SIZE_Y - 2]

def down_snapshot(sim):
	return sim.Ez[:, 1]

def up_left_snapshot(sim):
	return sim.Ez[1 ,int(sim.SIZE_Y / 2) : sim.SIZE_Y - 2]

def up_right_snapshot(sim):
	return sim.Ez[sim.SIZE_X - 2 ,int(sim.SIZE_Y / 2) : sim.SIZE_Y - 2]

def down_left_snapshot(sim):
	return sim.Ez[1, 1 : int(sim.SIZE_Y / 2)]

def down_right_snapshot(sim):
	return sim.Ez[sim.SIZE_X - 2, 1 : int(sim.SIZE_Y / 2)]



field_slice = np.zeros_like(simulator.Ez.T)
fourier_2D = np.zeros_like(field_slice)


# ========================================================



angles_arr = [] 
analytical_fresnel_T_arr = []
analytical_fresnel_R_arr = []
measured_fresnel_T_arr = []
measured_fresnel_R_arr = []
incoming_power = []

def propagate_waves(time):

	global field_slice

	while simulator.time < time * simulator.DELTA_T:

		simulator.update()
		field_slice = simulator.Ez.T


def calculate_fft2():
	global fourier_2D
	fourier_2D = np.fft.fft2(field_slice)
	fourier_2D = np.abs(np.fft.fftshift(fourier_2D))
	print("===========================================================================")
	print(f'Total Parseval Energy: {1 / (simulator.SIZE_X * simulator.SIZE_Y) * np.sum(np.power(fourier_2D, 2))}')
	print(f'Total Field Energy: {np.sum(np.power(field_slice, 2))}')
	
def plot():

	_, (ax0, ax1) = plt.subplots(1, 2)

	fft_slicer_x = (150, 350)
	fft_slicer_y = (150, 350)

	fft_diff_x = fft_slicer_x[1] - fft_slicer_x[0]
	fft_diff_y = fft_slicer_y[1] - fft_slicer_y[0]

	# Scale with the dispersion relation kc = 2pi f 
	fft_grid_x, fft_grid_y = np.meshgrid(
		np.arange(-int(fft_diff_x / 2), int(fft_diff_x / 2)) * 2 * np.pi * frequency / sp.constants.c, 
		np.arange(-int(fft_diff_y / 2), int(fft_diff_y / 2)) * 2 * np.pi * frequency / sp.constants.c)
	
	major_ticks = np.arange(-int(fft_diff_x / 2), int(fft_diff_x / 2), fft_diff_x / 8) * 2*np.pi*frequency / sp.constants.c
	

	fft_slice = fourier_2D[fft_slicer_x[0] : fft_slicer_x[1], fft_slicer_y[0] : fft_slicer_y[1]]
	fft_slice[:, 0 : int(fft_slice.shape[0] / 2)] = 0
	
	ax0.pcolormesh(fft_grid_x, fft_grid_y,
				fft_slice, cmap = plt.colormaps['turbo'], shading="auto")
	
	ax0.set_xticks(major_ticks)
	ax0.set_yticks(major_ticks)
	ax0.grid(which='major', alpha=0.5)
	ax0.set_xlabel("kx [1 / meter]")
	ax0.set_ylabel("ky [1 / meter]")
	#ax0.grid() 	
	ax0.set_aspect(1)

	field_grid_x, field_grid_y = np.meshgrid(
		np.arange(0, simulator.SIZE_X) * simulator.DELTA_X, 
		np.arange(0, simulator.SIZE_Y) * simulator.DELTA_Y)
	
	ax1.pcolormesh(field_grid_x, field_grid_y, field_slice, cmap = plt.colormaps['seismic'], shading="auto")
	ax1.set_aspect(1)
	ax1.set_xlabel("x [meter]")
	ax1.set_ylabel("y [meter]")
	plt.show()

## MEASUREMENTS ============================================================================================

angle_linspace = [np.pi / 4]

def clear_sim(sim):
	sim.Ex *= 0
	sim.Ey *= 0
	sim.Ez *= 0
	sim.Hx *= 0
	sim.Hy *= 0
	sim.Hz *= 0
	sim.time = 0
	sim.sources.clear()

for angle in angle_linspace:
	angles_arr.append(angle)
	pos_dir = arc_position(120, angle)

	simulator.gaussian_beam(starting_point = (pos_dir[0][0], pos_dir[0][1]), 
						direction = (pos_dir[1][0], pos_dir[1][1]), width = 40, function = source_func)
	

	propagate_waves(1000)
	calculate_fft2()
	plot()
	clear_sim(simulator)
	














