""" Routine for calculation of element stiffness matrix   """
import numpy as np
import fem_util as fem
import micromechanics_util as micro
import bone_micro_test as microb
import input_file as inp


### test environment PyTest is used as testing framework. 
# to test all the functions in this file, run py.test .\test_element_util.py -s in the terminal. 
# -s flag to print the print statements

MAT = inp.Mat_par()

def test_bone():

    u_ele = np.zeros((8,1))
    coord = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    theta_lac_el = np.zeros((1,4))
    f_HA_el = 0.55*np.ones((1,4))
    f_col_el = 0.45*0.9*np.ones((1,4))
    
    micro_var_el = [theta_lac_el, f_HA_el, f_col_el]
    print(micro_var_el)
    #MAT = inp.Mat_par()
    ke, fint_el = ele_matrices_micro_bone(u_ele, coord, micro_var_el, MAT)

    print (np.round(ke,2))
    

def test_sensitivity_bone():

    u_ele = np.zeros((8,1))
    coord = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])

    theta_lac_el = np.zeros((1,4))
    f_HA_el = 0.55*np.ones((1,4))
    f_col_el = 0.45*0.9*np.ones((1,4))

    micro_var_el = [theta_lac_el, f_HA_el, f_col_el]
    print(micro_var_el)
    ke, fint_el = ele_matrices_micro_bone(u_ele, coord, micro_var_el, MAT)
    ke_rho = ele_senstivity_mat_bone(u_ele,coord,micro_var_el, MAT)

    print (np.round(ke_rho,2))




