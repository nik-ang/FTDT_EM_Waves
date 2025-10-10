import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors

folder = 'results'
angles_arr = np.genfromtxt(f'{folder}/angles_arr.csv', delimiter = ';')
analytical_fresnel_R_arr = np.genfromtxt(f'{folder}/analytical_fresnel_R_arr.csv', delimiter = ';')
analytical_fresnel_T_arr = np.genfromtxt(f'{folder}/analytical_fresnel_T_arr.csv', delimiter = ';')
measured_fresnel_T_arr = np.genfromtxt(f'{folder}/measured_fresnel_T_arr.csv', delimiter = ';')

fig, (ax0, ax1) = plt.subplots(1, 2)

ax0.set_ylim(ymin = 0.5, ymax = 1.1)
ax0.set_xlabel("Angle [radians]")
ax0.set_title("Fresnel T Coefficient")

ax1.set_ylim(ymin = 0, ymax = 0.5)
ax1.set_xlabel("Angle [radians]")
ax1.set_title("Fresnel R Coefficient")

ax0.plot(angles_arr, analytical_fresnel_T_arr)
ax0.plot(angles_arr, measured_fresnel_T_arr, "o--")
ax1.plot(angles_arr, analytical_fresnel_R_arr)

plt.show()

print(analytical_fresnel_R_arr + analytical_fresnel_T_arr)