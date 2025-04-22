import pytest
import numpy as np
from src import fem_util as fem

### test environment PyTest is used as testing framework. 
# to test all the functions in this file, run py.test .\test_fem_util.py in the terminal.

pi = np.pi

class Test_Q4_LagrangeBasisFunc:
    ### tests for function Q4_LagrangeBasisFunc(xi,eta)

    def test_1(self):       
        ##All the basis functions of Q4 element will  be 0.25 at the center (xi =0, eta = 0)

        xi = 0.0
        eta = 0.0

        N = fem.Q4_LagrangeBasisFunc(xi,eta)
        N_ex = np.array([[0.25, 0.0 , 0.25, 0.0 , 0.25, 0.0 , 0.25, 0.0 ],
                        [0.0 , 0.25, 0.0 , 0.25, 0.0 , 0.25, 0.0 , 0.25]])
        
        assert N_ex.all() == N.all(), "basis functions not implemented correctly"

    def test_2(self):
        ##Partition of unity test: anywhere in the domain sum of all basis function is one
        
        xi = 0.33
        eta = 0.78

        N = fem.Q4_LagrangeBasisFunc(xi,eta)
        sum_N = np.sum(N,1)
        sum_N_ex = np.array([1.0, 1.0])

        assert sum_N_ex.all() == sum_N.all(), "basis functions not implemented correctly"
     

class Test_jacobian:
    ### tests for function jacobian(dvx, dvy)

    def test_1(self):
        ## For dvx = 2 and dvy = 2 det_J = 1.0 and inv_J should be identity matrix

        dvx = 2.0
        dvy = 2.0

        det_J, inv_J = fem.jacobian(dvx, dvy)

        det_J_ex = 1.0
        inv_J_ex = np.array([[1.0, 0.0],
                            [0.0, 1.0]])

        assert det_J_ex == det_J, "jacobian is wrongly implemented"
        assert inv_J_ex.all() == inv_J.all(), "jacobian is wrongly implemented"
        

class Test_B_matQ4:
    ### tests for function B_matQ4(xi,eta,coord)

    def test_1(self):
        ## Taken from SolidPy code 
        xi = 1.0
        eta = 1.0
        coord = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])

        det_J, B_mat = fem.B_matQ4(xi,eta,coord)

        det_J_ex = 1.0

        B_ex = 0.5 * np.array([[0, 0, 0, 0, 1, 0, -1, 0],
                            [0, 0, 0, -1, 0, 1, 0, 0],
                            [0, 0, -1, 0, 1, 1, 0, -1]])
        
        assert B_ex.all() == B_mat.all(), "B matrix is wrongly implemented"                    


class Test_strain:
    ### test for the function strain(point,coord,u_ele)
    
    def test_1(self):
        ##     
        point = np.array([  0.,  0.])
        coord = np.array([[ 0.,  0.],
                        [ 2.,  0.],
                        [ 2.,  2.],
                        [ 0.,  2.]])

        u_ele = np.array([ 0.,  0.,  2.,  0.,  2.,  0.,  0.,  0.])
        E = fem.strain(point,coord,u_ele)
        E_ex = np.array([1.0, 0.0, 0.0])

        assert E_ex.all() == E.all(), "Function to calculate strain at a point is wrongly implemented "

    

    



    
