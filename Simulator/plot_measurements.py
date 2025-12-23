import numpy as np
import matplotlib.pyplot as plt




folder = 'results'
angles_arr = np.genfromtxt(f'{folder}/angles_arr_1over6.csv', delimiter = ';')
analytical_fresnel_R_arr = np.genfromtxt(f'{folder}/analytical_fresnel_R_arr_1over6.csv', delimiter = ';')
analytical_fresnel_T_arr = np.genfromtxt(f'{folder}/analytical_fresnel_T_arr_1over6.csv', delimiter = ';')
measured_fresnel_T_arr_1over6 = np.genfromtxt(f'{folder}/measured_fresnel_T_arr_1over6.csv', delimiter = ';')
measured_fresnel_T_arr_1over4 = np.genfromtxt(f'{folder}/measured_fresnel_T_arr_1over4.csv', delimiter = ';')

fig, (ax0, ax1) = plt.subplots(1, 2)

ax0.set_ylim(ymin = 0.5, ymax = 1.1)
ax0.set_xlabel("Angle [radians]")
ax0.set_title("Fresnel T Coefficient")

ax1.set_ylim(ymin = 0, ymax = 0.5)
ax1.set_xlabel("Angle [radians]")
ax1.set_title("Fresnel R Coefficient")

ax0.plot(angles_arr, analytical_fresnel_T_arr, label="Analytical T Coefficient")
ax0.plot(angles_arr, measured_fresnel_T_arr_1over6, "o--", label="T Coefficient, f = 333.33 THz")
ax0.plot(angles_arr, measured_fresnel_T_arr_1over4, "o--", label="T Coefficient, f = 500 THz")
ax0.legend()

ax1.plot(angles_arr, analytical_fresnel_R_arr, label="Analytical R Coefficient")
ax1.plot(angles_arr, np.ones_like(measured_fresnel_T_arr_1over6) - measured_fresnel_T_arr_1over6, "o--", label="T Coefficient, f = 333.33 THz")
ax1.plot(angles_arr, np.ones_like(measured_fresnel_T_arr_1over4) - measured_fresnel_T_arr_1over4, "o--", label="T Coefficient, f = 500 THz")
ax1.legend()

plt.show()

print(analytical_fresnel_R_arr + analytical_fresnel_T_arr)