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
DELTA_T = 0.1
DELTA_X = DELTA_T * np.sqrt(2) * sp.constants.c
DELTA_Y = DELTA_X
simulator = FDTD_v2.FTDT_EM(500, 500, DELTA_X, DELTA_Y, DELTA_T, mur_abc = True)

epsilon_r = 5
sigma = 0

simulator.add_rectangle_material((0,0), simulator.SIZE_X, simulator.SIZE_Y / 2, epsilon_r, sigma, "#191981", alpha=0.1)
#simulator.point_source(point=(400, 250), function = lambda t: 10*np.exp(-((t - 10)**2) / 50) * np.cos(0.5*t))
#simulator.point_source(point=(100, 400), function = lambda t: 10*np.exp(-((t - 10)**2) / 50) * np.cos(0.5*t))

frequency = 1/6
N = int(np.round(1 / (simulator.DELTA_T * frequency)))

# 2pi f t ==> 2pi f N Dt = M 2pi ==> N = M / (f Dt)
def source_func(t):
	return 5 * np.cos(2*np.pi * frequency * t) * (1 - np.exp(-(t / 2)**2))

### Sanity check

def source_func_test(t):
	return 5 * np.cos(2*np.pi * frequency * t)

test_arr = np.arange(0, N*simulator.DELTA_T, simulator.DELTA_T)
test_sum = np.sum(np.vectorize(source_func_test)(test_arr))
print(test_sum)
assert(np.isclose(0, test_sum))


simulator.gaussian_beam(starting_point = (150, 350), direction = (1, -1), width = 40, function = source_func)
#simulator.phased_array(starting_point = (100, 400), direction = (1, -1), width = 40, function = source_func)

### PLOTY STUFF =============================================================================================
fig, ax = plt.subplots()
plt.style.use('dark_background')
heatmap = ax.pcolorfast(simulator.Ez.T, vmin=-1, vmax= 1, cmap = plt.colormaps['seismic_alpha'], rasterized=True, zorder = 1)

def init_plot():
	return heatmap,

ax.set_aspect(1)
for p in simulator.material_patches:
	ax.add_patch(p)


def incoming_x_snapshot():
	return simulator.Ez[1,int(simulator.SIZE_Y / 2) : simulator.SIZE_Y - 2]

def incoming_y_snapshot():
	return simulator.Ez[1 : int(simulator.SIZE_X / 2), simulator.SIZE_Y - 2]

def reflected_x_snapshot():
	return simulator.Ez[simulator.SIZE_X - 2,int(simulator.SIZE_Y /2) : simulator.SIZE_Y - 2]

def reflected_y_snapshot():
	return simulator.Ez[int(simulator.SIZE_X / 2) : simulator.SIZE_X - 2, simulator.SIZE_Y - 2]

def transmitted_x_snapshot():
	return simulator.Ez[:, 1]

def transmitted_y_snapshot():
	return simulator.Ez[simulator.SIZE_X - 2, 1 : int(simulator.SIZE_Y / 2)]

# Power measurement ======================================
Field_incoming_x = np.zeros_like(incoming_x_snapshot())
Field_incoming_y = np.zeros_like(incoming_y_snapshot())

Field_reflected_x = np.zeros_like(reflected_x_snapshot())
Field_reflected_y = np.zeros_like(reflected_y_snapshot())

Field_transmitted_x = np.zeros_like(transmitted_x_snapshot())
Field_transmitted_y = np.zeros_like(transmitted_y_snapshot())

Field_incoming_x_moving_avg_array = np.zeros((N,) + Field_incoming_x.shape)
Field_incoming_y_moving_avg_array = np.zeros((N,) + Field_incoming_y.shape)

Field_reflected_x_moving_avg_array = np.zeros((N,) + Field_reflected_x.shape)
Field_reflected_y_moving_avg_array = np.zeros((N,) + Field_reflected_y.shape)

Field_transmitted_x_moving_avg_array = np.zeros((N,) + Field_transmitted_x.shape)
Field_transmitted_y_moving_avg_array = np.zeros((N,) + Field_transmitted_y.shape)

print(Field_incoming_x_moving_avg_array.shape)

# ========================================================

def update(frames):
	global Field_incoming_x
	global Field_incoming_y
	global Field_reflected_x
	global Field_reflected_y
	global Field_transmitted_x
	global Field_transmitted_y
	global Field_incoming_x_moving_avg_array
	global Field_incoming_y_moving_avg_array
	global Field_reflected_x_moving_avg_array
	global Field_reflected_y_moving_avg_array
	global Field_transmitted_x_moving_avg_array
	global Field_transmitted_y_moving_avg_array


	#start = time.time()
	for _ in range(5):
		simulator.update()
	#end = time.time()
	#print(end - start)

	# Power measurement ======================================
	Field_incoming_x = np.maximum(Field_incoming_x, np.power(incoming_x_snapshot(),2))
	Field_incoming_y = np.maximum(Field_incoming_y, np.power(incoming_y_snapshot(),2))

	Field_reflected_x = np.maximum(Field_reflected_x, np.power(reflected_x_snapshot(),2))
	Field_reflected_y = np.maximum(Field_reflected_y, np.power(reflected_y_snapshot(), 2))

	Field_transmitted_x = np.maximum(Field_transmitted_x, np.sqrt(epsilon_r) * np.power(transmitted_x_snapshot(), 2))
	Field_transmitted_y = np.maximum(Field_transmitted_y, np.sqrt(epsilon_r) * np.power(transmitted_y_snapshot(),2))

	## MOVING AVERAGES

	# Incoming

	Field_incoming_x_moving_avg_array = np.roll(Field_incoming_x_moving_avg_array, -1, axis=0)
	Field_incoming_x_moving_avg_array[-1] = np.power(incoming_x_snapshot(), 2)

	Field_incoming_y_moving_avg_array = np.roll(Field_incoming_y_moving_avg_array, -1, axis=0)
	Field_incoming_y_moving_avg_array[-1] = np.power(incoming_y_snapshot(), 2)

	# Reflected

	Field_reflected_x_moving_avg_array = np.roll(Field_reflected_x_moving_avg_array, -1, axis = 0)
	Field_reflected_x_moving_avg_array[-1] = np.power(reflected_x_snapshot(), 2)

	Field_reflected_y_moving_avg_array = np.roll(Field_reflected_y_moving_avg_array, -1, axis = 0)
	Field_reflected_y_moving_avg_array[-1] = np.power(reflected_y_snapshot(), 2)

	# Transmitted

	Field_transmitted_x_moving_avg_array = np.roll(Field_transmitted_x_moving_avg_array, -1, axis=0)
	Field_transmitted_x_moving_avg_array[-1] = np.sqrt(epsilon_r) * np.power(transmitted_x_snapshot(),2)

	Field_transmitted_y_moving_avg_array = np.roll(Field_transmitted_y_moving_avg_array, -1, axis=0)
	Field_transmitted_y_moving_avg_array[-1] = np.sqrt(epsilon_r) * np.power(transmitted_y_snapshot(), 2)


	# ========================================================
	
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


print("===========================================================================")
print(f'Total Parseval Energy: {1 / (simulator.SIZE_X * simulator.SIZE_Y) * np.sum(np.power(fourier_2D, 2))}')
print(f'Total Field Energy: {np.sum(np.power(field_slice, 2))}')

incoming_angle = (np.pi/4)
transmission_angle = np.arcsin(np.sin(incoming_angle) * np.sqrt(1 / epsilon_r))
fresnel_r_te = (np.cos(incoming_angle) - np.sqrt(epsilon_r) * np.cos(transmission_angle)) / (np.cos(incoming_angle) + np.sqrt(epsilon_r) * np.cos(transmission_angle))
fresnel_t_te = (2 * np.cos(incoming_angle)) / (np.cos(incoming_angle) + np.sqrt(epsilon_r) * np.cos(transmission_angle))

print("===========================================================================")
print(f'Theoretical incoming angle: {incoming_angle * 180 / np.pi} degrees')
print(f'Theoretical transmission angle: {transmission_angle * 180 / np.pi} degrees')
print("===========================================================================")
print(f'Theoretical Fresnel Transmission Coefficient: {fresnel_t_te}')
print(f'Theoretical Fresnel Reflection Coefficient: {fresnel_r_te}')

fresnel_T_te = (np.sqrt(epsilon_r) * np.cos(transmission_angle)) / (1 * np.cos(incoming_angle)) * (fresnel_t_te**2)
fresnel_R_te = fresnel_r_te**2

print("===========================================================================")
print(f'Theoretical Fresnel Power Transmission Coefficient: {fresnel_T_te}')
print(f'Theoretical Fresnel Power Reflection Coefficient: {fresnel_R_te}')
print(f'Sum should be 1: {fresnel_R_te + fresnel_T_te}')

power_incoming = np.sum(Field_incoming_x) + np.sum(Field_incoming_y)
power_reflected = np.sum(Field_reflected_x) + np.sum(Field_reflected_y)
power_transmitted = np.sum(Field_transmitted_x) + np.sum(Field_transmitted_y)

print("===========================================================================")
print(f'Incoming power: {power_incoming}')
print(f'Transmitted power: {power_transmitted}')
print(f'Reflected power: {power_reflected}')
print(f'Power Sum: {power_transmitted + power_reflected}')
print(f'Power increase: {((power_reflected + power_transmitted) / power_incoming - 1)*100}%')

print(f'Measured Fresnel Power Transmission Coefficient: {power_transmitted / power_incoming}')
print(f'Measured Fresnel Power Reflection Coefficient: {power_reflected / power_incoming}')

## MOVING AVERAGES

power_incoming_x_moving_avg = np.average(Field_incoming_x_moving_avg_array, axis = 0)
power_incoming_y_moving_avg = np.average(Field_incoming_y_moving_avg_array, axis = 0)
power_incoming_moving_avg = np.sum(power_incoming_x_moving_avg) + np.sum(power_incoming_y_moving_avg)

power_reflected_x_moving_avg = np.average(Field_reflected_x_moving_avg_array, axis = 0)
power_reflected_y_moving_avg = np.average(Field_reflected_y_moving_avg_array, axis = 0)
power_reflected_moving_avg = np.sum(power_reflected_x_moving_avg) + np.sum(power_reflected_y_moving_avg)

power_transmitted_x_moving_avg = np.average(Field_transmitted_x_moving_avg_array, axis = 0)
power_transmitted_y_moving_avg = np.average(Field_transmitted_y_moving_avg_array, axis = 0)
power_transmitted_moving_avg = np.sum(power_transmitted_x_moving_avg) + np.sum(power_transmitted_y_moving_avg)


print("===========================================================================")
print(f'Incoming power moving average: {power_incoming_moving_avg}')
print(f'Reflected power moving average: {power_reflected_moving_avg}')
print(f'Transmitted power moving average: {power_transmitted_moving_avg}')

print(f'Power increase: {((power_reflected_moving_avg + power_transmitted_moving_avg) / power_incoming_moving_avg - 1)*100}%')
print(f'Measured Fresnel Power Transmission Coefficient: {power_transmitted_moving_avg / power_incoming_moving_avg}')
print(f'Measured Fresnel Power Reflection Coefficient: {power_reflected_moving_avg / power_incoming_moving_avg}')

plt.show()