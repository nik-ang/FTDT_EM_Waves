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
simulator_the_second = FDTD_v2.FTDT_EM(500, 500, DELTA_X, DELTA_Y, DELTA_T, mur_abc = True)

epsilon_r = 1.33
sigma = 0

simulator.add_rectangle_material((0,0), simulator.SIZE_X, simulator.SIZE_Y / 2, epsilon_r, sigma, "#191981", alpha=0.1)

frequency = 1/4
N = 2*int(np.round(1 / (simulator.DELTA_T * frequency)))

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


# Power measurement ======================================

# SIMULATOR 1

Field_transmitted_down = np.zeros_like(down_snapshot(simulator))
Field_transmitted_down_left = np.zeros_like(down_left_snapshot(simulator))
Field_transmitted_down_right = np.zeros_like(down_right_snapshot(simulator))

Field_transmitted_down_moving_avg_array = np.zeros((N,) + Field_transmitted_down.shape)
Field_transmitted_down_left_moving_avg_array = np.zeros((N,) + Field_transmitted_down_left.shape)
Field_transmitted_down_right_moving_avg_array = np.zeros((N,) + Field_transmitted_down_right.shape)

# SIMULATOR 2

Field_incoming_up = np.zeros_like(up_snapshot(simulator_the_second))
Field_incoming_down = np.zeros_like(down_snapshot(simulator_the_second))

Field_incoming_up_left = np.zeros_like(up_left_snapshot(simulator_the_second))
Field_incoming_up_right = np.zeros_like(up_right_snapshot(simulator_the_second))

Field_incoming_down_left = np.zeros_like(down_left_snapshot(simulator_the_second))
Field_incoming_down_right = np.zeros_like(down_right_snapshot(simulator_the_second))

Field_incoming_up_moving_avg_array = np.zeros((N,) + Field_incoming_up.shape)
Field_incoming_down_moving_avg_array = np.zeros((N,) + Field_incoming_down.shape)

Field_incoming_up_left_moving_avg_array = np.zeros((N,) + Field_incoming_up_left.shape)
Field_incoming_down_left_moving_avg_array = np.zeros((N,) + Field_incoming_down_left.shape)

Field_incoming_up_right_moving_avg_array = np.zeros((N,) + Field_incoming_up_right.shape)
Field_incoming_down_right_moving_avg_array = np.zeros((N,) + Field_incoming_down_right.shape)


field_slice = np.zeros_like(simulator_the_second.Ez.T)
fourier_2D = np.zeros_like(field_slice)


# ========================================================



angles_arr = [] 
analytical_fresnel_T_arr = []
analytical_fresnel_R_arr = []
measured_fresnel_T_arr = []
measured_fresnel_R_arr = []
incoming_power = []

def propagate_waves(time):

	global Field_incoming_up
	global Field_incoming_down

	global Field_incoming_up_left
	global Field_incoming_up_right

	global Field_incoming_down_left
	global Field_incoming_down_right

	global Field_incoming_up_moving_avg_array
	global Field_incoming_down_moving_avg_array

	global Field_incoming_up_left_moving_avg_array
	global Field_incoming_down_left_moving_avg_array

	global Field_incoming_up_right_moving_avg_array
	global Field_incoming_down_right_moving_avg_array

	global Field_transmitted_down
	global Field_transmitted_down_left
	global Field_transmitted_down_right

	global Field_transmitted_down_moving_avg_array
	global Field_transmitted_down_left_moving_avg_array
	global Field_transmitted_down_right_moving_avg_array

	global field_slice

	while simulator.time < time:

		simulator.update()
		simulator_the_second.update()

		## SIMULATION 1

		# Incoming

		Field_incoming_up_moving_avg_array = np.roll(Field_incoming_up_moving_avg_array, -1, axis=0)
		Field_incoming_up_moving_avg_array[-1] = np.power(up_snapshot(simulator_the_second), 2)

		Field_incoming_down_moving_avg_array = np.roll(Field_incoming_down_moving_avg_array, -1, axis=0)
		Field_incoming_down_moving_avg_array[-1] = np.power(down_snapshot(simulator_the_second), 2)

		Field_incoming_up_left_moving_avg_array = np.roll(Field_incoming_up_left_moving_avg_array, -1, axis=0)
		Field_incoming_up_left_moving_avg_array[-1] = np.power(up_left_snapshot(simulator_the_second), 2)

		Field_incoming_up_right_moving_avg_array = np.roll(Field_incoming_up_right_moving_avg_array, -1, axis=0)
		Field_incoming_up_right_moving_avg_array[-1] = np.power(up_right_snapshot(simulator_the_second), 2)

		Field_incoming_down_left_moving_avg_array = np.roll(Field_incoming_down_left_moving_avg_array, -1, axis=0)
		Field_incoming_down_left_moving_avg_array[-1] = np.power(down_left_snapshot(simulator_the_second), 2)

		Field_incoming_down_right_moving_avg_array = np.roll(Field_incoming_down_right_moving_avg_array, -1, axis=0)
		Field_incoming_down_right_moving_avg_array[-1] = np.power(down_right_snapshot(simulator_the_second), 2)

		# Transmitted

		Field_transmitted_down_moving_avg_array = np.roll(Field_transmitted_down_moving_avg_array, -1, axis=0)
		Field_transmitted_down_moving_avg_array[-1] = np.sqrt(epsilon_r) * np.power(down_snapshot(simulator),2)

		Field_transmitted_down_left_moving_avg_array = np.roll(Field_transmitted_down_left_moving_avg_array, -1, axis=0)
		Field_transmitted_down_left_moving_avg_array[-1] = np.sqrt(epsilon_r) * np.power(down_left_snapshot(simulator), 2)

		Field_transmitted_down_right_moving_avg_array = np.roll(Field_transmitted_down_right_moving_avg_array, -1, axis=0)
		Field_transmitted_down_right_moving_avg_array[-1] = np.sqrt(epsilon_r) * np.power(down_right_snapshot(simulator), 2)
		field_slice = simulator.Ez.T

def calculate_analytical_fresnel_coeff(angle):
	incoming_angle = angle
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

	analytical_fresnel_T_arr.append(fresnel_T_te)
	analytical_fresnel_R_arr.append(fresnel_R_te)

	print("===========================================================================")
	print(f'Theoretical Fresnel Power Transmission Coefficient: {fresnel_T_te}')
	print(f'Theoretical Fresnel Power Reflection Coefficient: {fresnel_R_te}')
	print(f'Sum should be 1: {fresnel_R_te + fresnel_T_te}')

def measure_power_flux():
	power_incoming_up_moving_avg = np.average(Field_incoming_up_moving_avg_array, axis = 0)
	power_incoming_down_moving_avg = np.average(Field_incoming_down_moving_avg_array, axis = 0)
	power_incoming_up_left_moving_avg = np.average(Field_incoming_up_left_moving_avg_array, axis = 0)
	power_incoming_up_right_moving_avg = np.average(Field_incoming_up_right_moving_avg_array, axis = 0)
	power_incoming_down_left_moving_avg = np.average(Field_incoming_down_left_moving_avg_array, axis = 0)
	power_incoming_down_right_moving_avg = np.average(Field_incoming_down_right_moving_avg_array, axis = 0)

	power_incoming_moving_avg = (np.sum(power_incoming_up_moving_avg) 
							  + np.sum(power_incoming_down_moving_avg) 
							  + np.sum(power_incoming_up_left_moving_avg) 
							  + np.sum(power_incoming_up_right_moving_avg) 
							  + np.sum(power_incoming_down_left_moving_avg) 
							  + np.sum(power_incoming_down_right_moving_avg))
	
	power_incoming_moving_avg *= 0.5

	power_transmitted_down_moving_avg = np.average(Field_transmitted_down_moving_avg_array, axis = 0)
	power_transmitted_down_left_moving_avg = np.average(Field_transmitted_down_left_moving_avg_array, axis = 0)
	power_transmitted_down_right_moving_avg = np.average(Field_transmitted_down_right_moving_avg_array, axis = 0)

	power_transmitted_moving_avg = (np.sum(power_transmitted_down_moving_avg) 
								 + np.sum(power_transmitted_down_left_moving_avg) 
								 + np.sum(power_transmitted_down_right_moving_avg))
	fresnel_T_te = power_transmitted_moving_avg / power_incoming_moving_avg
	fresnel_R_te = 1 - fresnel_T_te

	measured_fresnel_T_arr.append(fresnel_T_te)
	measured_fresnel_R_arr.append(fresnel_R_te)

	print("===========================================================================")
	print(f'Incoming power moving average: {power_incoming_moving_avg}')
	print(f'Transmitted power moving average: {power_transmitted_moving_avg}')

	print(f'Measured Fresnel Power Transmission Coefficient: {fresnel_T_te}')
	print(f'Measured Fresnel Power Reflection Coefficient: {fresnel_R_te}')


def calculate_fft2():
	fourier_2D = np.fft.fft2(field_slice)
	fourier_2D = np.abs(np.fft.fftshift(fourier_2D))
	print("===========================================================================")
	print(f'Total Parseval Energy: {1 / (simulator.SIZE_X * simulator.SIZE_Y) * np.sum(np.power(fourier_2D, 2))}')
	print(f'Total Field Energy: {np.sum(np.power(field_slice, 2))}')
	
def plot():
	fig, (ax0, ax1) = plt.subplots(1, 2)
	ax0.pcolorfast(fourier_2D, cmap = plt.colormaps['turbo'])
	ax1.pcolorfast(field_slice, cmap = plt.colormaps['seismic'])
	ax0.set_aspect(1)
	ax1.set_aspect(1)
	plt.show()

## MEASUREMENTS ============================================================================================

angle_linspace = np.linspace(0, np.pi/3, 10)
angle_linspace = np.flip(angle_linspace)

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
	
	simulator_the_second.gaussian_beam(starting_point = (pos_dir[0][0], pos_dir[0][1]), 
						direction = (pos_dir[1][0], pos_dir[1][1]), width = 40, function = source_func)

	propagate_waves(100)
	calculate_analytical_fresnel_coeff(angle)	
	measure_power_flux()
	calculate_fft2()
	#plot()
	clear_sim(simulator)
	clear_sim(simulator_the_second)
	

angles_arr = np.array(angles_arr)
analytical_fresnel_R_arr = np.array(analytical_fresnel_R_arr)
analytical_fresnel_T_arr = np.array(analytical_fresnel_T_arr)
measured_fresnel_T_arr = np.array(measured_fresnel_T_arr)


folder = 'results'
angles_arr.tofile(f'{folder}/angles_arr.csv', sep = ';')
analytical_fresnel_R_arr.tofile(f'{folder}/analytical_fresnel_R_arr.csv', sep = ';')
analytical_fresnel_T_arr.tofile(f'{folder}/analytical_fresnel_T_arr.csv', sep = ';')
measured_fresnel_T_arr.tofile(f'{folder}/measured_fresnel_T_arr.csv', sep = ';')













