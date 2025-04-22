import numpy as np
from scipy.signal import convolve2d
import concurrent.futures

import src.fem_util as fem
import input_file as inp
import src.bone_element_util as EleR
import src.bone_micro_util as micro

## Filters and kernal that are used in structural topology optimization

def filter_indices_bone(nodes,elem,rmin):
    
    n_ele = elem.shape[0]

    # list with middle point of each element
    mid_el = [[np.mean(nodes[elem[i_ele,:],0]),np.mean(nodes[elem[i_ele,:],1])] for i_ele in range(n_ele)]

     # Create a mapping from node numbers to row indices in the nodes array
    #node_num_to_index = {node_num: i for i, node_num in enumerate(nodes[:, 0])}

    # # Compute the middle point of each element using the mapping
    # mid_el = []
    # for i_ele in range(n_ele):
    #     x_coords = []
    #     y_coords = []
    #     for j in range(1, 5):  # Columns 1 to 4 in elem correspond to node numbers
    #         node_index = node_num_to_index[elem[i_ele, j]]
    #         x_coords.append(nodes[node_index, 1])
    #         y_coords.append(nodes[node_index, 2])
    #     mid_el.append([np.mean(x_coords), np.mean(y_coords)])

    index = []; weight = []
    for i in range(n_ele):
        x_i = mid_el[i][0]
        y_i = mid_el[i][1]

        ind = []; w = []
        for j in range(n_ele):
            x_j = mid_el[j][0]
            y_j = mid_el[j][1]
            dst = np.sqrt((x_j - x_i)**2. + (y_j - y_i)**2.)
            if(rmin - dst > 0):
                ind.append(j)
                w.append(rmin - dst)

        index.append(ind)
        weight.append(w)

    return [index, weight]
    
        
def sensitivity_filter_bone(dc, index, weight):
    """Filtering of senstivities to tackle the checkboarding solutions
    Input: dc: ndarray of shape (n_ele,) of sensitivities in each element
           index:
           weight: 
    Output: senstivities after applying filter""" 
    dc = np.asarray(dc)
    n_ele = len(index)
    dc_new = np.zeros(n_ele,)
    for i in range(n_ele):
        ind = index[i]
        w = weight[i]
        dc_new[i] = sum(dc[ind]*w)/sum(w)
    
    return dc_new


## Functions that are used for testing (structural) topology optimization routines using SIMP method

def sensitivity_func(u_sol, nodes, elem, rho, C, p):
    """ Calculation of sensitivites in each element for SIMP optimization
    Input: nodes, elem: mesh description
           rho, C: material constant
           u_sol : displacement solution
            p    : SIMP penalty"""
    
    n_ele = elem.shape[0]
    dc = np.zeros(n_ele)

    for el in range(0,n_ele):

        v = [2*elem[el,1], 2*elem[el,1] + 1, 2*elem[el,2], 2*elem[el,2] + 1,
             2*elem[el,3], 2*elem[el,3] + 1, 2*elem[el,4], 2*elem[el,4] + 1]

        u_ele = u_sol[v]
        coord =  nodes[elem[el][1:5]][:,1:3]
        rho_el = rho[el]
        
        ke = EleR.eleStiffMat_SIMP(coord,C)    
        dc[el] = p*(rho_el**(p-1.))*np.dot(np.dot(u_ele.T,ke),u_ele)

    return dc

def update_rho_SIMP(rho, dc, vol_dom, vol_el, volfrac, move, dampCoeff):
    cutoff = 1e-5

    rho_min = 0.56
    rho_max = 0.75
    
    l1 = 0
    l2 = 1.0e10
    rho_k = rho
    
    itr = 0
    while((l2-l1)/(l1 + l2) > 1e-5):
        itr += 1
        lmid = (l1+l2)/2.0
        B = (dc/(vol_el*lmid))**dampCoeff
        rho_new = np.maximum(rho_min, np.maximum(rho_k - move,
                np.minimum(rho_max, np.minimum(rho_k + move, rho_k*B))))

        if ((np.sum(rho_new)*vol_el - volfrac*vol_dom) > 0.0):
            l1 = lmid
        else:
            l2 = lmid

    change = np.linalg.norm(rho_new - rho)/np.linalg.norm(rho)
    print("Langrange Multiplier update iterations", itr," Lagrange multiplier",
          lmid, " Change in density", change )

    return rho_new

### Functions used for bone material in concurrent material-structure optimization

def sesitivity_func_micro_el_bone(data_el):
  
    u_ele = data_el[0]
    coord = data_el[1]
    micro_var_el = data_el[2]

    ke_rho = EleR.ele_senstivity_mat_bone(u_ele,coord, micro_var_el)
    #ke_rho, fint = EleR.ele_matrices_micro(u_ele, coord, micro_var_el)
    dc_el = np.dot(np.dot(u_ele.T,ke_rho),u_ele)

    return dc_el
    
def sensitivity_func_micro_bone(u_sol, nodes, elem, rho, micro_var):

    n_ele = elem.shape[0]
    dc = np.zeros(n_ele)

    # unpacking of microvariables
    theta_lac = micro_var[0]
    f_HA_bar = micro_var[1]
    f_col_bar = micro_var[2]

    n_ele = elem.shape[0]
    u_el = [u_sol[[2*elem[el,1], 2*elem[el,1] + 1, 2*elem[el,2], 2*elem[el,2] + 1,
             2*elem[el,3], 2*elem[el,3] + 1, 2*elem[el,4], 2*elem[el,4] + 1]] for el in range(n_ele)]

    coord_el = [nodes[elem[el][1:5]][:,1:3] for el in range(n_ele)]

    micro_var_el = [[theta_lac[el].reshape((1,4)), f_HA_bar[el].reshape((1,4)), f_col_bar[el].reshape((1,4))] for el in range(n_ele)]

    data = [[u_el[el], coord_el[el], micro_var_el[el]] for el in range(n_ele)]

    with concurrent.futures.ProcessPoolExecutor() as executor: # For parallel computing 
        results = executor.map(sesitivity_func_micro_el_bone,data)

    return list(results)


def update_rho_bone(rho,dc,vol_dom,vol_el,volfrac,move,dampCoeff):
    cutoff = 1e-5
    rho_HA = 1.0;  rho_col = 1.41/3; rho_w = 1/3
    phi_lac = 0.021
    #
    rho_ec_min = (1 - 0.3 - (1/0.9) * 0.2) * rho_w + (1/0.9) * 0.2 * rho_col +  0.3 * rho_HA    
    rho_ec_max = (1 - 0.55 - (1/0.9) * (0.45 * 0.9)) * rho_w + (1/0.9) * (0.45 * 0.9) * rho_col + 0.55 * rho_HA
    #
    rho_min = max(0.56, rho_w*phi_lac + rho_ec_min*(1 - phi_lac))
    rho_max = min(0.75, rho_w*phi_lac + rho_ec_max*(1 - phi_lac))
    
    l1 = 0
    l2 = 1.0e10
    rho_k = rho
    
    itr = 0
    while((l2-l1)/(l1 + l2) > 1e-5):
        itr += 1
        lmid = (l1+l2)/2.0
        B = (dc/(vol_el*lmid))**dampCoeff
        rho_new = np.maximum(rho_min, np.maximum(rho_k - move,
                np.minimum(rho_max, np.minimum(rho_k + move, rho_k*B))))

        if ((np.sum(rho_new)*vol_el - volfrac*vol_dom) > 0.0):
            l1 = lmid
        else:
            l2 = lmid

    change_rho = np.linalg.norm(rho_new - rho)/np.linalg.norm(rho)
    print("Langrange Multiplier update iterations", itr," Lagrange multiplier",
          lmid, " Change in density", change_rho )

    return rho_new, change_rho

# rho_HA = 1.0;  rho_col = 1.41/3; rho_w = 1/3
# phi_lac = 0.021
# rho_ec_min = 1/3 #whole matrix is material B
# rho_ec_max = 1.0 # whole matrix is material C
# #
# rho_ec_min = (1 - 0.3 - (1/0.9) * 0.2) * rho_w + (1/0.9) * 0.2 * rho_col +  0.3 * rho_HA    
# rho_ec_max = (1 - 0.55 - (1/0.9) * (0.45 * 0.9)) * rho_w + (1/0.9) * (0.45 * 0.9) * rho_col + 0.55 * rho_HA
# #
# rho_min = max(0.001, rho_w*phi_lac + rho_ec_min*(1 - phi_lac))
# rho_max = min(0.75, rho_w*phi_lac + rho_ec_max*(1 - phi_lac))
# print(rho_min)
# print(rho_max)



