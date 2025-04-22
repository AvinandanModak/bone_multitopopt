""" Routine for calculation of element stiffness matrix   """
import numpy as np
import src.fem_util as fem
import src.bone_micro_util as microb
import input_file as inp

# bone material function
def ele_matrices_micro_bone(u_ele,coord,micro_var_el):
    """ Function to calculate element stiffness matrix and internal force
    vector when constitutive material law is originating from micromechanics.
    Microscale can be described by many variables such as volume fraction of
    different constituents, inclusion shape, orientation etc. Place these
    variables inside the micro_var list. Use micromechanics principles to
    calculate elasticity tensor at each gauss point.

    The micromechanics scheme is illustrated in Gangwar et al., 2020 and the function
    is given in micromechanics_util.py
    
    INPUT: u_ele: nodal displacement solution vector for
                 the element (ndarray of shape (8,1) for Q4)
           coord: Coordinates of the nodes of the element ndarray of shape(4, 2)
           micro_var: List of microscale variables at each gauss point in an element
                     - 1st element of the list is volume fraction of Material A (phi_A)
                       at each Gauss point(ndarray shape (1, 4))
                     - 2nd element of the list is orientation of material stiffness tensor
                       of Material A (theta_A) in radians at each gauss point (ndarray shape (1,4))
                     - 3rd element of the list is elongation of Material A (zeta_A) at each
                       gauss point (ndarray shape (1, 4))
                     - 4th element of the list is the volume fraction of Material C at the
                       lowermost scale.

    Output: ke: element stiffness matrix
            fint_el: element internal force vector   """
    theta_lac_el = micro_var_el[0]
    f_HA_el = micro_var_el[1]
    f_col_el = micro_var_el[2]


    ke = np.zeros([8, 8]) # initialize the element stiffness matrix
    XW, XP = fem.gpoints2x2() # 2*2 gauss point co-ordinates
    ngpts = 4 # Number of gauss points
    gp_C = []

    for i in range(0, ngpts):

        xi  = XP[i, 0];eta = XP[i, 1]
        gw = XW[i]

        theta_lac = theta_lac_el[0][i]
        f_HA_bar = f_HA_el[0][i]
        f_col_bar = f_col_el[0][i]

        C = microb.bone_C_hom(f_col_bar, f_HA_bar, theta_lac) # homogenized stiffness tensor
        gp_C.append(C) # append Chom for 4 gps
        det_J, B = fem.B_matQ4(xi,eta,coord) # jacobian and B_mat       
        ke = ke + np.dot(np.dot(B.T,C), B)*gw*det_J 

    fint_el = np.dot(ke,u_ele) # internal force vector
    
    return ke, fint_el, gp_C

# bone material function
def ele_senstivity_mat_bone(u_ele,coord,micro_var_el):
    
    theta_lac_el = micro_var_el[0]
    f_HA_el = micro_var_el[1]
    f_col_el = micro_var_el[2]

    ke_rho = np.zeros([8, 8]) # initialize the delC_by_rho matrix for the element
    XW, XP = fem.gpoints2x2() # 2*2 gauss point co-ordinates
    ngpts = 4 # Number of gauss points

    for i in range(0, ngpts):
        
        xi  = XP[i, 0];eta = XP[i, 1]
        gw = XW[i]
        
        theta_lac = theta_lac_el[0][i]
        f_HA_bar = f_HA_el[0][i]
        f_col_bar = f_col_el[0][i]      

        delC_rho = microb.del_C_by_rho_bone(f_col_bar, f_HA_bar, theta_lac)
        det_J, B = fem.B_matQ4(xi,eta,coord) # jacobian and B_mat
        ke_rho = ke_rho + np.dot(np.dot(B.T,delC_rho), B)*gw*det_J

    return ke_rho



def eleStiffMat_SIMP(coord,C):   
    """ Stiffness matrix for isotropic SIMP type method verifications.
        Input: coord: Coordinates of the nodes of the element ndarray of shape(4, 2)
        C: isotropic elasticty matrix        
        Output: ke element stiffness matrix """
    
    ke = np.zeros([8, 8]) # initialize element stiffness matrix
    XW, XP = fem.gpoints2x2() # Gauss Points and weight

    ngpts = 4
    for i in range(0, ngpts):

        xi  = XP[i, 0];eta = XP[i, 1]
        gw = XW[i]
        det_J, B = fem.B_matQ4(xi,eta,coord)        
        ke = ke + np.dot(np.dot(B.T,C), B)*gw*det_J
        
    return ke





