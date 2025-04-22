import numpy as np
from random import random
import time

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import input_file as inp
import micromechanics_util as micro


pi = np.pi

### tests for material optimization routines, mostly by visualization.    

"""Test the micro_opt function with a brutforce approach. We just evaluate the value of the
function on all possible values and find the design variables that provide the maximum value of
the local strain density."""

rho = 0.65 #0.8*random()
Ex = np.array([random(),random(),random()]) # random values in [0,1]
Ex = -1 + (Ex*(1 + 1))# Random Macroscopic strains with values in [-1,1]
print(rho, Ex)

MAT = inp.Mat_par() #Material parameters
rho_HA = 1;  rho_col = 1.41/3; rho_w = 1/3

phi_HA_min = 0.3; phi_HA_max = 0.55
phi_col_min = 0.2; phi_col_max = 0.45*0.9
f_col_bar = np.linspace(phi_col_min, phi_col_max, 10)
theta_lac = np.linspace(0, pi, 181)
f_lac = 0.021
alpha = 1/0.9

fun_f_HA = lambda x: (rho - (1-f_lac)*alpha*x*(rho_col - rho_w) - f_lac*rho_w - (1 - f_lac)*rho_w)/((1 - f_lac)*(rho_HA - rho_w))

w = [[strain_energy_local_bone(Ex, m, fun_f_HA(m), n, MAT) for m in list(f_col_bar)]for n in list(theta_lac)]
w = np.array(w)
ind = np.unravel_index(np.argmax(w, axis=None), w.shape)

print(ind)

phi_col_max = f_col_bar[ind[1]]
phi_HA_max = fun_f_HA(phi_col_max)
theta_lac_max = theta_lac[ind[0]]

print ("optimal microstructure from brut force",theta_lac_max, phi_HA_max, phi_col_max, np.max(w))



data_x = [rho,Ex]
print(data_x)


[theta_lac_opt, phi_HA_opt, phi_col_opt] = micro_opt_bone(data_x)
w_opt = strain_energy_local_bone(Ex, phi_col_opt, phi_HA_opt, theta_lac_opt, MAT)
print("optimal microstructure from MicrOpt", theta_lac_opt, phi_HA_opt, phi_col_opt, w_opt)




w_val = [[strain_energy_local_bone(Ex, m, fun_f_HA(m), n, MAT) for m in list(f_col_bar)]for n in list(theta_lac)]
#print(w_val)
f_col_bar, phi_HA_bar = np.meshgrid(f_col_bar, f_HA_bar)
X = f_col_bar
Y = f_HA_bar
Z =  np.array(w_val)

fig = plt.figure(1)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
#cset = ax.contourf(X, Y, Z, zdir='z', offset=  0.4, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, Z, zdir='x', offset=  phi_A_min, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, Z, zdir='y', offset=  0.2, cmap=cm.coolwarm)

ax.set_xlabel('f_col_bar')
ax.set_ylabel('f_HA_bar')
ax.set_zlabel('w')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()




