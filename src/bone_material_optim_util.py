import numpy as np
import concurrent.futures
import time
from scipy.optimize import minimize

import src.fem_util as fem
import input_file as inp
import src.bone_micro_util as microb

pi = np.pi

# Bone optimization functions

def strain_energy_local_bone(Ex, f_col_bar, f_HA_bar, theta_lac):

    C = microb.bone_C_hom(f_col_bar, f_HA_bar, theta_lac)
    w = np.matmul(Ex,np.matmul(C,Ex))

    return w

def optimal_orientation_bone(Ex):
    """ The inclusions at mesoscale make the macroscopic material orthotrpic. Hence, the optimal 
    orientation of inclsions is simply in the direction of principle strains axes (Jog et. al, 1994).
    INPUT: Macroscale strains Ex 
    OUTPUT: optimal orientation theta_A_opt"""

    e11 = (Ex[0] + Ex[1])/2.0 + np.sqrt( ((Ex[0] - Ex[1])/2.0)**2. + (Ex[2]/2)**2.) # principle strain 1
    e22 = (Ex[0] + Ex[1])/2.0 - np.sqrt( ((Ex[0] - Ex[1])/2.0)**2. + (Ex[2]/2)**2.) # principle strain 2
    theta_p1 = 0.5*np.arctan2(Ex[2]/2, (Ex[0] - ((Ex[0]+Ex[1])/2.0))) # Principle strain 1 direction from x axis
    theta_p2 = theta_p1 + pi/2 # Principle strain 2 direction from x axis
    # tht_max is optimal orientation
    if (abs(e11) >= abs(e22)):
        theta_lac_opt = theta_p1
    else:
        theta_lac_opt = theta_p2
    
    if (theta_lac_opt < 0 ):
        theta_lac_opt = pi + theta_lac_opt #keep theta value in the range of [0,pi]
    
    return theta_lac_opt



def micro_opt_bone(data_x):
    """Microscale optimization method for each gauss point.
    At each gauss point, the macroscale strains E(x) and density \rho(x) is known.
    We maximize the local strain energy at each gauss point and obtain optimal configuration
    of microscale parameters phi_A, theta_A, zeta_A, gamma_C. The inclusions at mesoscale
    make the macroscopic material orthotrpic. Hence, the optimal orientation of inclsions is
    simply in the direction of principle strains axes (Jog et. al, 1994). Given macroscopic
    density constraints and the microscopic volume fraction variables \phi_A and \gamma_C, we
    eliminate \gamma_C and replace it with \phi_A utilizing this constraint. It will also provide
    the bounds of \phi_A.
    INPUT: data_x: it contains rho and Ex at the gauss point. It is ndarray of shape (2,).
                   First value is density of material at the gauss point and second is the list of
                   all the component of Ex. 
    OUTPUT: List of optimal values of phi_A, theta_A, zeta_A and gamma_C.    """
    rho = data_x[0]
    Ex = np.array(data_x[1])
    
    # Constants and initial values
    rho_HA = 1
    rho_col = 1.41 / 3
    rho_w = 1 / 3
    phi_HA_min = 0.3
    phi_HA_max = 0.55
    phi_col_min = 0.2
    phi_col_max = 0.45 * 0.9
    f_lac = 0.021
    alpha = 1 / 0.9

    # Optimal orientation
    theta_lac_opt = optimal_orientation_bone(Ex)

    # Objective function to minimize (negative because we want to maximize strain energy)
    func = lambda x: -1*strain_energy_local_bone(Ex, x[0], x[1], theta_lac_opt)
    con = lambda x: rho_w * f_lac + ((1 - x[1] - alpha * x[0]) * rho_w + alpha * x[0] * rho_col + x[1] * rho_HA) * (1 - f_lac) - rho
    # Bounds on the design variables
    bnds = ((phi_col_min, phi_col_max), (phi_HA_min, phi_HA_max))

    # Initial guess
    phi_col_0 = (phi_col_min + phi_col_max) / 2
    phi_HA_0 = (phi_HA_min + phi_HA_max) / 2

    # Constraint setup
    cons = ({'type': 'eq', 'fun': con})

    # Optimization with SLSQP method
    res = minimize(func, (phi_col_0, phi_HA_0), method='SLSQP', bounds=bnds, constraints= cons, options={'ftol': 1e-9, 'disp': None})

    # Extract optimal values
    phi_col_opt, phi_HA_opt = res.x

    return [theta_lac_opt, phi_HA_opt, phi_col_opt]

  
def micro_update_bone(u_sol,nodes, elem, rho):
    """ This function update the microscopic variables in whole domain for given
    macroscopic density rho and macroscopic strain field.
    INPUT: u_sol: displacement solution vector
           nodes, elem: mesh information
           rho: macroscopic density distribution
    OUTPUT: microstructural field solution."""

    XW, XP = fem.gpoints2x2()

    n_ele = elem.shape[0]
    u_ele = [u_sol[[2*elem[el,1], 2*elem[el,1] + 1, 2*elem[el,2], 2*elem[el,2] + 1,
             2*elem[el,3], 2*elem[el,3] + 1, 2*elem[el,4], 2*elem[el,4] + 1]] for el in range(n_ele)]

    data = [[rho[y],list(fem.strain(XP[x],nodes[elem[y][1:5]][:,1:3],u_ele[y]))]
               for x in range(4)  for y in range(n_ele)]

    start = time.perf_counter()
    
    with concurrent.futures.ProcessPoolExecutor() as executor: # For parallel computing 
       results = executor.map(micro_opt_bone,data)

    stop = time.perf_counter()
    print ('time taken in parallel setting',stop - start)

    return list(results)



