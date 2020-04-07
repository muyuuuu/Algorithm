import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.stats import norm

N = 1000
np.random.seed(1)
X = np.concatenate((np.random.normal(0, 1, int(0.3 * N)), np.random.normal(5, 1, int(0.7 * N))))[:, np.newaxis]

X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

colors = ['blue']
kernels = ['gaussian']
lw = 2

for color, kernel in zip(colors, kernels):
    kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(X)
    log_dens = kde.score_samples(X_plot)
    plt.plot(X_plot[:, 0], np.exp(log_dens), color=color, lw=lw,
            linestyle='-', label="kernel = '{0}'".format(kernel))
    arr = np.exp(log_dens)
    np.save('data.npy', arr)

plt.legend(loc='upper left')
plt.hist(X, 70, normed = 1, histtype='bar', facecolor='salmon', rwidth=0.9)  
# plt.set_xlim(-4, 9)
# plt.set_ylim(-0.02, 0.4)

plt.title("$N$ = 1000 points", fontsize=16)
plt.xlabel("Data", fontsize=14)
plt.ylabel("Density Function", fontsize=14)
plt.savefig('test.png', dpi=300)