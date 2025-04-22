import numpy as np
from scipy.sparse import coo_matrix
import src.bone_element_util as EleR



def DispBC_operator(DispBC):
    """This function create a vector identifying degrees of freedom with zero
    displacement boundary conditions"""
    n_disp = np.asarray(DispBC).shape[0]
    disp_vec = []
    for i in range (0,n_disp):

        ind = DispBC[i,0]
        if (DispBC[i,1] != -1):
            disp_vec.append(ind*2)
        if (DispBC[i,2] != -1):
            disp_vec.append(ind*2 + 1)

    disp_vec = np.asarray(disp_vec, dtype = int)
    return disp_vec


### Functions that are used for testing (structural) topology optimization routines using SIMP method


def assem_load(nodes, elem, LoadBC, DispBC):
    """Assemble the force vector.
    Input: nodes, elem define the mesh (see input file routine)
           LoadBC: contains point load information with node identifier
           DispBC: contains Dirichlet BC information 
           Output: Fext force vector"""

    disp_vec = DispBC_operator(DispBC) 
    
    ndof = 2*nodes.shape[0]
    n_ele = elem.shape[0]
    n_loads = LoadBC.shape[0]

    Fext = np.zeros([ndof,1])

    for i in range(0,n_loads):
        ind = int(LoadBC[i,0])
        Fext[2*ind] = LoadBC[i,1]
        Fext[2*ind +1 ] = LoadBC[i,2]
    
    Fext[disp_vec] = 0.0

    return Fext


### Functions used in concurrent material-structure optimization in bone.
 
def assem_system_micro_bone(u_sol, nodes, elem, rho, micro_var, DispBC, Chom_full):
    """Assemble global stiffness matrix and global internal force vector 
    Input: u_sol: displacement solution vector
           nodes, elem define the mesh (see input file routine) 
           -------------------------------------------------------------------------------------------------
           nodes: Each row represents a node in the mesh, and the columns contain the following information: (A 2D NumPy array of size (Node numbers,3))
            Column 0: Node number (a unique identifier for each node).
            Column 1: x-coordinate of the node.
            Column 2: y-coordinate of the node.
           -------------------------------------------------------------------------------------------------
           elem: Each row represents an element in the mesh, and the columns contain the following information:  (A 2D NumPy array of size (Element numbers,5))
            Column 0: Element number (a unique identifier for each element).
            Columns 1-4: Node numbers corresponding to the vertices of the element in counterclockwise order:
            Column 1: Bottom-left node of the element.
            Column 2: Bottom-right node of the element.
            Column 3: Top-right node of the element.
            Column 4: Top-left node of the element.
           -------------------------------------------------------------------------------------------------
           [coord = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) , Coordinates of the nodes of the element ndarray of shape(4, 2)]
           -------------------------------------------------------------------------------------------------
           rho: density of each element 
           micro_var: micro characterization field (phi_A, theta_A, zeta_A, gamma_C)
           DispBC: contains Dirichlet BC information (generated in input_file.py)
    Output: Global stiffness matrix in sparse system 
            Fint: Global internal force vector """

    disp_vec = DispBC_operator(DispBC)
    ndof = 2*nodes.shape[0]
    n_ele = elem.shape[0]
    index = 0
    

    Fint = np.zeros([ndof,1]) # initialize the global internal force vector

    # global stiffness matrix sparse assembly information lists
    rows = []
    cols = []
    vals = []

    # unpacking of microvariables
    theta_lac = micro_var[0]
    f_HA_bar = micro_var[1]
    f_col_bar = micro_var[2]
    
    for el in range(0,n_ele):
        
        v = [2*elem[el,1], 2*elem[el,1] + 1, 2*elem[el,2], 2*elem[el,2] + 1,
             2*elem[el,3], 2*elem[el,3] + 1, 2*elem[el,4], 2*elem[el,4] + 1]
        u_ele = u_sol[v]
        
        coord =  nodes[elem[el][1:5]][:,1:3] #coordinates of the nodes of element number el

        # microscale variables in each element
        micro_var_el = [theta_lac[el].reshape((1,4)) ,f_HA_bar[el].reshape((1,4)), f_col_bar[el].reshape((1,4))]

        ke, fint_el, gp_C = EleR.ele_matrices_micro_bone(u_ele, coord, micro_var_el)
        #Chom assemble
        for elemC in gp_C:
            Chom_full[:, :, index] = elemC
            index += 1  # Increment the index

        #print(f"Iteration {el + 1} of {n_ele} is completed")
        #ke = rho[el]*ke
        Fint[v,0] = Fint[v,0] + fint_el # assemble internal force vector

        # assemble global stiffness in sparse setting
        for row in range(8):
            glob_row = v[row]
            for col in range(8):
                glob_col = v[col]
                rows.append(glob_row)
                cols.append(glob_col)
                if(not(glob_row in disp_vec) and not(glob_col in disp_vec)):
                    vals.append(ke[row,col])
                elif((glob_row in disp_vec) and (glob_col in disp_vec) and (glob_row == glob_col)): # application of zero DispBC
                    vals.append(1.0)
                else:
                    vals.append(0.0)


    Kglob = coo_matrix((vals,(rows,cols)),shape = (ndof,ndof)).tocsr()
    Fint[disp_vec] = 0.0 # apply zero displacement boundary conditions on Global Fint

    return Kglob, Fint, Chom_full



def assem_Fint_micro_bone(u_sol, nodes, elem, rho, micro_var, DispBC, Chom_full):
    """ Assemble global internal force vector
    Input: u_sol: displacement solution vector
           nodes, elem define the mesh (see input file routine)
           micro_var: micro characterization field (phi_A, theta_A, zeta_A, gamma_C)
           DispBC: contains Dirichlet BC information (generated in input_file.py)
    Output: Global stiffness matrix in sparse system 
            Fint: Global internal force vector """  

    disp_vec = DispBC_operator(DispBC)
    ndof = 2*nodes.shape[0]
    n_ele = elem.shape[0]
    index = 0

    Fint = np.zeros([ndof,1])

    # unpacking of microvariables
    theta_lac = micro_var[0]
    f_HA_bar = micro_var[1]
    f_col_bar = micro_var[2]

    for el in range(0,n_ele):
        
        v = [2*elem[el,1], 2*elem[el,1] + 1, 2*elem[el,2], 2*elem[el,2] + 1,
             2*elem[el,3], 2*elem[el,3] + 1, 2*elem[el,4], 2*elem[el,4] + 1]
        
        u_ele = u_sol[v]
        coord =  nodes[elem[el][1:5]][:,1:3] # coordinates of the nodes of element number el

        # micoscale variables in each element
        micro_var_el = [theta_lac[el].reshape((1,4)) ,f_HA_bar[el].reshape((1,4)), f_col_bar[el].reshape((1,4))] 

        ke, fint_el, gp_C = EleR.ele_matrices_micro_bone(u_ele, coord, micro_var_el)
        #print(f"Iteration {el + 1} of {n_ele} is completed")    
        Fint[v,0] = Fint[v,0] + fint_el

        #Chom assemble
        for elemC in gp_C:
            Chom_full[:, :, index] = elemC
            index += 1  # Increment the index

    Fint[disp_vec] = 0.0

    return Fint, Chom_full




