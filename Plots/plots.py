import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def f(x, y, h):
	return np.exp(-(x**2 + y**2) / h)

def main():

	x = np.arange(-5, 5, 0.1)
	y = np.arange(-5, 5, 0.1)
	h = 2

	X, Y = np.meshgrid(x, y)
	Z = f(X+1, Y+1, h) - f(X-1, Y-1, h)

	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	ax.plot_surface(X, Y, Z, vmin=Z.min() * 2, cmap='cool')
	ax.axis('off')

	plt.show()

main()