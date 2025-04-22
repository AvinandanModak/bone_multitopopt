""" ------ Mesh routine -----

This routine creates a structured mesh with quadrilateral elements for different problems """

import numpy as np
import scipy.io as sio


### functions used in testing of finite element and optimization routines 

def test_Q4(x0, y0, x1, y1, nx, ny):
    """ A square plate subjected to uniform traction at the right end and symmtric boundary
    conditions at the bottom line and left line. """

    nodes, elem = RectMesh(x0, y0, x1, y1, nx, ny)
    DispBC = np.zeros((nx + ny + 1,3))

    # at x = 0, y = 0 ux = 0, uy = 0
    DispBC[0,0] = 0; DispBC[0,1] = 0.0; DispBC[0,2] = 0.0

    # uy = 0 at the bottom line (y = 0 line)
    disp_ind_x = np.linspace(1,nx,nx)

    DispBC[1:nx+1,0] = disp_ind_x[0:nx]
    DispBC[1:nx+1,1] = -1
    DispBC[1:nx+1,2] = 0.0

    # ux = 0 at the left line (x = 0 line)
    disp_ind_y = np.linspace(nx+1,(nx+1)*ny,ny)
    DispBC[nx+1:nx+ny+1,0] = disp_ind_y[0:ny]
    DispBC[nx+1:nx+ny+1,1] = 0.0
    DispBC[nx+1:nx+ny+1,2] = -1


    LoadBC = np.zeros((ny +1 ,3))
    load_ind = np.linspace(nx,(nx+1)*(ny+1) - 1 , ny +1)

    A_el = (y1 - y0)/ny # area of one element
    f = 1.0*A_el

    LoadBC[0,0] = load_ind[0]
    LoadBC[0,1] = f/2.
    LoadBC[0,2] = 0.0

    LoadBC[1:ny,0] = load_ind[1:ny]
    LoadBC[1:ny,1] = f
    LoadBC[1:ny,2] = 0.0

    LoadBC[ny,0] = load_ind[ny]
    LoadBC[ny,1] = f/2.
    LoadBC[ny,2] = 0.0

    return nodes, elem, LoadBC, DispBC

def test_square4():
    """ for the testing of finite element code.
    test problem: https://solidspy.readthedocs.io/en/latest/tutorials/square_example.html
    x0,y0 = 0.0,0.0
    x1,y1 = 2.0,2.0
    nx = ny = 2
    DirichletBCs = at (0,0): u_y = 0 ; (1,0): u_x = 0, u_y = 0 ; (2,0): u_y = 0
    LoadBCs = (0,2): Px = 0, Py = 1.0 ; (1,2): Px = 0, Py = 2.0 ; (2,2): Px = 0, Py = 1.0

    LoadBC = [nodenumber, Px, Py] """
    x0,y0 = 0.0, 0.0
    x1, y1 = 2.0, 2.0
    nx, ny = 2, 2
    nodes, elem = RectMesh(x0, y0, x1, y1, nx, ny)
                  
    LoadBC = np.array([[6, 0.0, 1.0],
                       [7, 0.0, 2.0],
                       [8, 0.0, 1.0]])

    DispBC = np.array([[0, -1, 0],
                       [1,  0, 0],
                       [2, -1, 0]]) # -1 flag is for free BC, 0 (or any other float) means the applied Displacement BC

    return nodes, elem, DispBC, LoadBC


## # functions that are used in the examples reported in the paper

def Mat_par_bone():
    """ Material input parameter for 3scale system reported in paper
    It has hierarchically organized three materials A,B, and C.
    This function allow us to write density, Young's Modulus and poisson's
    ratio for these materials."""

    rho_w = 1/3; k_w = 2.3;
    rho_col = 1.41/3; 
    rho_HA = 1.0; k_HA = 82.6; mu_HA = 44.9;
    phi_col_min = 0.2; phi_col_max = 0.45*0.9;
    phi_HA_min = 0.3; phi_HA_max = 0.55;
    return ((rho_w, k_w),(rho_col),(rho_HA, k_HA, mu_HA), (phi_col_min, phi_col_max), (phi_HA_min, phi_HA_max))


def Bone():

    mat_data1 = sio.loadmat("squared_mesh_data.mat")

    # Extract nodes and elements
    nodes = mat_data1['nodes']
    elem = mat_data1['elem']


    mat_data2 = sio.loadmat("Combined_BC.mat")


    LoadBC = mat_data2['LoadBC']
    DispBC = mat_data2['DispBC']

    return nodes, elem, LoadBC, DispBC
















 





    


    
