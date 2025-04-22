#import standard modules
import numpy as np
import scipy.io as sio
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import time

from src import input_file as inp
from src import fem_util as fem
from src import bone_assembly_util as asem
from src import bone_micro_util as micro
from src import bone_material_optim_util as MicrOpt
from src import bone_struct_optim_util as MacrOpt
#from src import visualization as vis

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

pi = np.pi

# Macroscopic optimization parameters
niter = 30 # total number of iterations
rmin = 2 # filter radius
volfrac = 0.65 # target average material density
move = 0.02 # move parameter in Fixed point algo update
dampCoeff = 0.5 # Damp coeeficient in Fixed point algo update

nodes, elem, LoadBC, DispBC  = inp.Bone() # MODIFIED
n_ele = elem.shape[0]
print(n_ele)

vol_dom = 5860.3385 # estimated volume of the segmented bone domain
vol_el = vol_dom/(n_ele) # estimated volume of each element
vol = 0.75

rho = vol*np.ones(n_ele) # density in each element

[index, weight] = MacrOpt.filter_indices_bone(nodes, elem, rmin) #MODIFIED
dc_old = np.zeros(n_ele)

# Macro variables
i = 0
c = []# objective function value
change = 1.0
Fext = asem.assem_load(nodes, elem, LoadBC, DispBC)

# save microvariables
theta_lac_data = [] 
f_HA_data = []
f_col_data = []
rho_data = []
u_data = []
vol_data = []
Chom_full = np.zeros((3, 3, n_ele * 4))  # Initialize the 3D array

start = time.perf_counter()
if __name__ == '__main__':
    while i<= niter:

        vol = max(vol - 0.5*move, volfrac)
        vol_data.append(vol)

        # Micro variables
        theta_lac = np.zeros((n_ele,4))
        f_HA_bar = 0.55*np.ones((n_ele,4))  
        f_col_bar = 0.45*0.9*np.ones((n_ele,4))
        micro_var = [theta_lac, f_HA_bar, f_col_bar]
        
        ## FE2 routine with micro-strcuture optimization
        u = np.zeros((2*nodes.shape[0],))
        K0, Fint, Chom_full = asem.assem_system_micro_bone(u, nodes, elem, rho, micro_var, DispBC, Chom_full)  # Initialize the 3D array)
        du = spsolve(K0, Fext - Fint)
        u = u + du
        
        itr_max = 7
        tol = 1e-2
        itr = 0
        
        del_u = 1.
        while del_u > tol and  itr <= itr_max:               
            arr = MicrOpt.micro_update_bone(u,nodes, elem, rho)
            arr = np.asarray(arr)
            theta_lac = arr[:,0]; f_HA_bar = arr[:,1]; f_col_bar = arr[:,2]

            theta_lac = theta_lac.reshape((4,n_ele)); theta_lac = theta_lac.T
            f_HA_bar = f_HA_bar.reshape((4,n_ele)); f_HA_bar = f_HA_bar.T     
            f_col_bar = f_col_bar.reshape((4,n_ele)); f_col_bar = f_col_bar.T  
            micro_var = [theta_lac, f_HA_bar, f_col_bar]

            Fint, Chom_full = asem.assem_Fint_micro_bone(u, nodes, elem, rho, micro_var, DispBC, Chom_full)
            du = spsolve(K0, Fext - Fint)

            del_u = np.linalg.norm(du)/np.linalg.norm(u) 
            
            u = u + du 
            itr+=1
            #c.append(np.asscalar(np.dot(Fext.T,u.reshape(u.shape[0],1))))
            print ("Newton iteration number",itr,"Displacement norm ratio",del_u, " obj ",
                   np.dot(Fext.T,u.reshape(u.shape[0],1)).item())


        # append all the variables to save
        rho_data.append(rho)
        u_data.append(u)
        theta_lac_data.append(theta_lac)
        f_HA_data.append(f_HA_bar)
        f_col_data.append(f_col_bar)
            
        # value of objective function 
        c.append(np.dot(Fext.T,u.reshape(u.shape[0],1)).item())
       
        # Senstivity calculations
        dc = MacrOpt.sensitivity_func_micro_bone(u, nodes, elem, rho, micro_var)
        dc = MacrOpt.sensitivity_filter_bone(dc, index, weight) # MODIFIED
        
        #stablization 
        if(i>1):
            dc_old = dc
            dc = (dc + dc_old)/2.

        rho, change_rho = MacrOpt.update_rho_bone(rho, dc, vol_dom, vol_el, vol, move, dampCoeff)
    
        if(i>10):
            change = abs(sum(c[i-9:i-5]) - sum(c[i-4:i]))/sum(c[i-4:i])

        #print("It. {}: Obj.:".format(i), c[i], "ch.:", change)
        print("Optimization iteration", i+1 ,";completed with objective function value", c[i], ";average change in objective function", change, "; density change", change_rho)

        if(change_rho < 0.01 and change < 0.001):
            break            
        i+=1    

stop = time.perf_counter()
print (stop - start)
rho_data = np.asarray(rho_data)
u_data = np.asarray(u_data)
theta_lac_data = np.asarray(theta_lac_data)
f_HA_data = np.asarray(f_HA_data)
f_col_data = np.asarray(f_col_data)


sio.savemat('rho_data.mat',{'arr':rho_data})
sio.savemat('u_data.mat',{'arr':u_data})
sio.savemat('theta_lac_data.mat',{'arr':theta_lac_data})
sio.savemat('f_HA_data.mat',{'arr':f_HA_data})
sio.savemat('f_col_data.mat',{'arr':f_col_data})
sio.savemat('Chom_data.mat',{'arr':Chom_full})

