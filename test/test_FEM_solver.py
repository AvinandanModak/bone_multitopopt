import pytest
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve


from src import input_file as inp
from src import fem_util as fem
from src import assembly as asem

### test environment PyTest is used as testing framework. 
# to test all the functions in this file, run py.test .\test_FEM_solver.py  in the terminal. 

pi = np.pi

class Test_FEM_solver:
    ## tests for verifying the FEM solver

    def test_square4ele(self):
        # To test the FEM solver. Refer to
        # https://solidspy.readthedocs.io/en/latest/tutorials/square_example.html
        # for the description of the problem
        
        # lenhgth and height of domain
        nodes, elem, DispBC, LoadBC = inp.test_square4()

        n_ele = elem.shape[0]
        rho = np.ones(n_ele) # density in each element
        p = 1

        E = 1.0
        nu = 0.3

        C = fem.C_mat(E,nu)

        Fext = asem.assem_load(nodes, elem, LoadBC,DispBC)
        KG = asem.assem_stiff_SIMP(nodes, elem, rho, C, DispBC, p)

        u_sol = spsolve(KG, Fext)

        u_ex = np.array([0.6, 0.0 , 0.0, 0.0, -0.6, 0.0, 0.6, 2.0, 0.0, 2.0, -0.6, 2.0, 0.6, 4.0, 0.0, 4.0, -0.6,4.0])

        assert u_ex.all() == u_sol.all(), " FEM solver is not correct"




