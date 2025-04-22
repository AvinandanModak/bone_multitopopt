import pytest
import numpy as np
from src import input_file as inp

### test environment PyTest is used as testing framework. 
# to test all the functions in this file, run py.test .\test_input_file.py in the terminal. 
 

class Test_RectMesh:
    ## tests for function RectMesh(x0, y0, x1, y1, nx, ny)

    def test_1(self):
        # test for mesh generation routine 
        x0 = 0.0
        y0 = 0.0
        x1 = 1.0
        y1 = 1.0

        nx = 2
        ny = 2

        coord, elem = inp.RectMesh(x0, y0, x1, y1, nx, ny)

        coord_ex = np.array([[0.0, 0., 0. ],[1.0, 0.5, 0. ],[2.0,1, 0.0],
                            [3.0, 0., 0.5],[4.0, 0.5, 0.5],[5,1.0, 0.5],
                            [6.0, 0., 1.0],[7.0, 0.5, 1.0],[8,1.0, 1.0]]) # manually calculated coordinates for l = 1.0, h = 1.0, m = n = 2
        elem_ex = np.array([[0, 0, 1, 4, 3], [1, 1, 2, 5, 4], [2, 3, 4, 7, 6], [3, 4, 5, 8, 7]]) # manually calculated connectivity matrix of elements for l = 1.0, h = 1.0, m = n = 2

        assert coord_ex.all() == coord.all(), "mesh generation routine is wrongly implemented "
        assert elem_ex.all() == elem.all(), "mesh generation routine is wrongly implemented "

class Test_cantilever:
    ## tests for function to initialize cantilever
    
    def test_1(self):

        x0 = 0.0
        y0 = 0.0
        x1 = 2.0
        y1 = 1.0

        nx = 40
        ny = 20
        h = 0.05

        nodes, elem, LoadBC, DispBC = inp.cantilever(x0,y0,x1,y1,nx,ny,h)

        LoadBC_ex = np.array([[409, 0.0, -0.025],
                            [450, 0.0, -0.05 ],
                            [491, 0.0, -0.025]])
        
        assert LoadBC_ex.all() == LoadBC.all(), " input file for cantilever problem is wrongly implemented"

class Test_half_MBB:

    def test_1(self):
        x0 = 0.0
        y0 = 0.0
        x1 = 1.0
        y1 = 1.0

        nx = 2
        ny = 2

        nodes, elem, LoadBC, DispBC = inp.half_MBB(x0,y0,x1,y1,nx,ny)


        nodes_ex = np.array([[0.0, 0., 0. ],[1.0, 0.5, 0. ],[2.0,1, 0.0],
                            [3.0, 0., 0.5],[4.0, 0.5, 0.5],[5,1.0, 0.5],
                            [6.0, 0., 1.0],[7.0, 0.5, 1.0],[8,1.0, 1.0]])

        elem_ex = np.array([[0, 0, 1, 4, 3], [1, 1, 2, 5, 4], [2, 3, 4, 7, 6], [3, 4, 5, 8, 7]])

        LoadBC_ex =np.array([[6, 0.0, -1.0]])

        DispBC_ex = np.array([[0, 0.0, -1 ],
                            [3, 0.0, -1 ],
                            [6, 0.0, -1 ],
                            [2, -1 , 0.0]])


        assert nodes_ex.all() == nodes.all(), "input file for half MBB problem is wrongly implemented "
        assert elem_ex.all() == elem.all(),  "input file for half MBB problem is wrongly implemented "
        assert LoadBC_ex.all() == LoadBC.all(), "input file for half MBB problem is wrongly implemented "
        assert DispBC_ex.all() == DispBC.all(), "input file for half MBB problem is wrongly implemented "

    

  
    

    
