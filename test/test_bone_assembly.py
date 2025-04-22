import numpy as np
from src import input_file as inp
from src import fem_util as fem 

### test environment PyTest is used as testing framework. 
# to test all the functions in this file, run py.test .\test_assembly.py  in the terminal. 

## NOTE: assem_system_micro(u_sol, nodes, elem, rho, micro_var, MAT, DispBC) and 
# assem_Fint_micro(u_sol, nodes, elem, rho, micro_var, MAT, DispBC) are yet to be tested.

## tests for function test_Q4(x0,y0,x1,y1,nx,ny)

def test_1():
    # test the DispBC_operator routine for test_Q4 example with one element

    x0 = 0.0
    y0 = 0.0
    x1 = 2.0
    y1 = 2.0

    nx = 1
    ny = 1

    disp_vec_ex = np.array([0, 1, 3, 4])

    nodes, elem, LoadBC, DispBC = inp.test_Q4(x0,y0,x1,y1,nx,ny)
    disp_vec = DispBC_operator(DispBC)

    assert disp_vec_ex.all() == disp_vec.all(), " test failed "


test_1()


## tests for function assem_load(nodes, elem, LoadBC,DispBC)

def test_2():
    # test the Global Fext assembly routine for test_Q4 example with one element
    x0 = 0.0
    y0 = 0.0
    x1 = 2.0
    y1 = 2.0

    nx = 1
    ny = 1

    Fext_ex = np.array([[0.0],
                        [0.0],
                        [1.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [1.0],
                        [0.0]])

    nodes, elem, LoadBC, DispBC = inp.test_Q4(x0,y0,x1,y1,nx,ny)
    Fext = assem_load(nodes, elem, LoadBC,DispBC)

    assert Fext_ex.all() == Fext.all(), " assembly for force vector is wrongly implemented"

test_2()





