# %load bone_micromechanics_util
import numpy as np
import input_file as inp
import tensor_util as tensor
import time
from scipy.integrate import quad
from scipy.optimize import fsolve
import scipy.linalg


def test_C_hom():
    """test the homogenization scheme for the paper example.
    with f_col = 0.41; f_HA = 0.29; f_HA_foam = 0.66; f_fib = 0.52; f_lac = 0.021
    The scheme has five scales with both MT scheme and self-consistent scheme."""

    MAT = inp.Mat_par()
    f_col = 0.41
    f_HA = 0.29
    f_HA_foam = 0.66
    f_fib = 0.52
    f_lac = 0.021

    d_s = 1.26

    f_col_bar = (f_fib*335.6)/(1.47*64*5*d_s)
    print('f_col_bar', f_col_bar)
    phi_HA_ef = (1-f_fib)/(1-f_col_bar)
    f_HA_bar = f_HA_foam*(1-f_fib)/phi_HA_ef
    print('f_HA_bar', f_HA_bar)
    f_HA_bar_2 = (f_HA*f_fib)/(1-phi_HA_ef)
    print('f_HA_bar_2', f_HA_bar_2)
    f_wetcol = 1 - f_HA
    f_col_bar_2 = f_col*f_wetcol
    print('f_col_bar_2', f_col_bar_2)
    theta_lac = 0
    C = bone_C_hom(f_col_bar, f_HA_bar, theta_lac, MAT)
    C = np.round(C,5)
    print ("C_test: ", C)
    C_ex = np.array([[ 18.3,  8.1, 0],
                     [ 8.1,  18.3, 0],
                     [ 0, 0, 10.1/2]])

    print ("Test for homogenization of bone:", np.allclose(C,C_ex,atol=0.1))



def test_del_C_by_rho():
    """test the sensitivity module for the bone example.
    with f_col = 0.41; f_HA = 0.29; f_HA_foam = 0.66; f_fib = 0.52; f_lac = 0.021
    The scheme has five scales with both MT scheme and self-consistent scheme."""

    MAT = inp.Mat_par()
    f_col = 0.41
    f_HA = 0.29
    f_HA_foam = 0.66
    f_fib = 0.52
    f_lac = 0.021

    d_s = 1.26
    theta_lac = 0
    f_col_bar = (f_fib*335.6)/(1.47*64*5*d_s)
    print('f_col_bar', f_col_bar)
    phi_HA_ef = (1-f_fib)/(1-f_col_bar)
    f_HA_bar = f_HA_foam*(1-f_fib)/phi_HA_ef
    print('f_HA_bar', f_HA_bar)
    f_HA_bar_2 = (f_HA*f_fib)/(1-phi_HA_ef)
    print('f_HA_bar_2', f_HA_bar_2)
    f_wetcol = 1 - f_HA
    f_col_bar_2 = f_col*f_wetcol
    print('f_col_bar_2', f_col_bar_2)    

    del_C_by_rho = del_C_by_rho_bone(f_col_bar, f_HA_bar, theta_lac, MAT)
    print('del_C_by_rho for bone:', del_C_by_rho)


"""Determination of tissue specific volume fractions"""
# rho_mu = 1.92 # wet/dry
# WF_mu_HA = 0.75
# WF_mu_org = 0.25
# d_s = 1.14
# f_mu_por = 0.05
# rho_w = 1
# rho_HA = 3
# rho_col = 1.41

# rho_ultra_wet = (rho_mu - rho_w*f_mu_por)/(1 - f_mu_por)
# print('rho_ultra_wet', rho_ultra_wet)
# rho_ultra_dry = rho_mu /(1 - f_mu_por)
# print('rho_ultra_dry', rho_ultra_dry)
# WF_mu_mupor = (rho_w*f_mu_por)/rho_mu
# print('WF_mu_mupor', WF_mu_mupor)
# WF_ultra_HA = WF_mu_HA/(1 - WF_mu_mupor)
# print('WF_ultra_HA', WF_ultra_HA)
# WF_ultra_org = WF_mu_org/(1 - WF_mu_mupor)
# print('WF_ultra_org', WF_ultra_org)

# f_HA_wet = (rho_ultra_wet/rho_HA)*WF_ultra_HA
# f_col_wet = (rho_ultra_wet/rho_col)*0.9*WF_ultra_org
# print('f_HA_wet', f_HA_wet)
# print('f_col_wet', f_col_wet)

# f_HA_dry = (rho_ultra_dry/rho_HA)*(WF_ultra_HA/(WF_ultra_HA+WF_ultra_org))
# f_col_dry = (rho_ultra_dry/rho_col)*(0.9*WF_ultra_org/(WF_ultra_HA+WF_ultra_org))
# print('f_HA_dry', f_HA_wet)
# print('f_col_dry', f_col_wet)

