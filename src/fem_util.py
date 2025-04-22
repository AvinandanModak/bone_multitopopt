import numpy as np

def gpoints2x2():
    """Gauss points for a 2 by 2 grid
    Returns
    -------
    xw : ndarray
      Weights for the Gauss-Legendre quadrature.
    xp : ndarray
      Points for the Gauss-Legendre quadrature.
    """
    xw = np.zeros([4])
    xp = np.zeros([4, 2])
    xw[:] = 1.0
    xp[0, 0] = -0.577350269189626
    xp[1, 0] = 0.577350269189626
    xp[2, 0] = -0.577350269189626
    xp[3, 0] = 0.577350269189626

    xp[0, 1] = 0.577350269189626
    xp[1, 1] = 0.577350269189626
    xp[2, 1] = -0.577350269189626
    xp[3, 1] = -0.577350269189626

    return xw, xp

def Q4_LagrangeBasisFunc(xi,eta):
    """this function will give the value of Lagrange basis function for 4 node Quadrilateral element at a given gauss point (xi, eta) in local co-ordinates.
    Input: xi and eta are local co-ordinates 
    Output: Numpy array of (2,8) size having vaule of all the basis functions at (xi,eta) """

    N = np.zeros((2, 8))

    H = 0.25*np.array([(1 - xi)*(1 - eta ), (1 + xi )*(1 - eta ), (1 + xi )*(1 + eta), (1 - xi)*(1 + eta)])
    N[0, ::2] = H
    N[1, 1::2] = H

    return N

def jacobian(dvx, dvy):
    """ function to evaluate jacobian of the transformation at the gauss point.
    NOTE: as the mesh in this problem is regular. every element is a square. Hence, jacobian and Inverse jacobian magtrices are constant.
    Hence, jacobian wont be evaluated at each gauss point to save the computational time
    Input: dvx: length of a element in x direction
           dvy: length of a element in y direction
    Output: Det_J: determinant of jacobian matrix
            Inv_J: inverse of jacobian matrix"""
    det_J = dvx*dvy/4.
    inv_J = (1.0/det_J)*np.array([[dvy/2.0, 0],[0, dvx/2.0]])
    return det_J, inv_J


def B_matQ4(xi,eta,coord):
    """Strain-displacement interpolator B for a 4-noded quad element
    Input: xi and eta are local co-ordinates
            coord: Coordinates of the nodes of the element (4, 2)
    Output: B = Bmatrix evaluated at xi,eta ndarray of shape (3,8)"""
    B_mat = np.zeros((3, 8))
    
    dNdxi = 0.25*np.array([[(eta - 1),(1 - eta),( 1 + eta),( -1 - eta)],
                           [(xi - 1),(-1 - xi),( 1 + xi),( 1 - xi)]]) # Value of derivative of Basis function in the local co-ordinates basis (xi, eta). (2,4) array first row dNdxi, second row dNdeta

    dvx = coord[2][0] - coord[0][0]
    dvy = coord[2][1] - coord[0][1]
    
    det_J, inv_J = jacobian(dvx, dvy)

    dNdX =  np.dot(inv_J, dNdxi)

    B_mat[0, ::2] = dNdX[0, :]
    B_mat[1, 1::2] = dNdX[1, :]
    B_mat[2, ::2] = dNdX[1, :]
    B_mat[2, 1::2] = dNdX[0, :]
    
    return det_J, B_mat

def strain(point,coord,u_ele):

    xi = point[0]
    eta = point[1]
    det_J, B = B_matQ4(xi,eta,coord)
    E =  np.dot(B,u_ele)

    return E

def C_mat(E, nu):
    """2D Elasticity consitutive matrix in plane stress
    for isotrpic material.
    Input:  material constants EA, ET, GA, nuA
    Output:  C : ndarray
    Constitutive tensor in Voigt notation."""

    C11 = E/(1 - nu**2.)
    C12 = nu*C11
    C22 = C11
    C66 = ((1 - nu)/2.)*C11

    C = np.array([[C11, C12, 0.0],
                  [C12, C22, 0.0],
                  [0.0, 0.0, C66]])

    return C


