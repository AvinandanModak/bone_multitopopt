import numpy as np


def I4():
    """Fourth order identity tensor
    I = del_{il}del_{jk} e_i e_j e_k e_l"""
    I = np.array([[1,0,0,0,0,0,0,0,0],
                  [0,1,0,0,0,0,0,0,0],
                  [0,0,1,0,0,0,0,0,0],
                  [0,0,0,0,1,0,0,0,0],
                  [0,0,0,1,0,0,0,0,0],
                  [0,0,0,0,0,0,1,0,0],
                  [0,0,0,0,0,1,0,0,0],
                  [0,0,0,0,0,0,0,0,1],
                  [0,0,0,0,0,0,0,1,0]])
    return I


def I4_sym():
    """Symmetric fourth-order tensor """

    Is = np.array([[1,0,0,0,0,0,0,0,0],
                   [0,1,0,0,0,0,0,0,0],
                   [0,0,1,0,0,0,0,0,0],
                   [0,0,0,0.5,0.5,0,0,0,0],
                   [0,0,0,0.5,0.5,0,0,0,0],
                   [0,0,0,0,0,0.5,0.5,0,0],
                   [0,0,0,0,0,0.5,0.5,0,0],
                   [0,0,0,0,0,0,0,0.5,0.5],
                   [0,0,0,0,0,0,0,0.5,0.5]])

    return Is


def a_dyad_b(a,b):
    """ a and b are vectors in catesian basis. dyadic product would be a
    second order tensor.
    INPUT: a and b are vectors in cartesian basis (ndarray of shape (3,) both).
    OUTPUT:dyadic product that is a second order tensor. (ndarray of shape (3,3))"""

    A = np.array([[a[0]*b[0], a[0]*b[1], a[0]*b[2]],
                  [a[1]*b[0], a[1]*b[1], a[1]*b[2]],
                  [a[2]*b[0], a[2]*b[1], a[2]*b[2]]])

    return A

def A2_dyad_B2(A,B):
    """dyadic of two second order tensor will result in a fourth order tensor.

    ------ Cijkl = A_ij*B_kl----------
    
    INPUT: A, B second order tensors in (3*3) matrix notations. (both ndarray of shape (3,3))
    OUTPUT: Fourth order tensor in matrix form (9*9). For the formation of matrix refer
    https://www.mate.tue.nl/~peters/4K400/VectTensColMat.pdf
    
    NOTE: A_ij and B_kl are considered relevant to our probelm that is 13,31,23,32 components are zero."""
    
    Cijkl = np.zeros((9,9))
    C1111 = A[0,0]*B[0,0];C1122 = A[0,0]*B[1,1];C1133 = A[0,0]*B[2,2];C1112 = A[0,0]*B[0,1];C1121 = A[0,0]*B[1,0]
    C2211 = A[1,1]*B[0,0];C2222 = A[1,1]*B[1,1];C2233 = A[1,1]*B[2,2];C2212 = A[1,1]*B[0,1];C2221 = A[1,1]*B[1,0]

    C3311 = A[2,2]*B[0,0];C3322 = A[2,2]*B[1,1];C3333 = A[2,2]*B[2,2];C3312 = A[2,2]*B[0,1];C3321 = A[2,2]*B[1,0]
    C1211 = A[0,1]*B[0,0];C1222 = A[0,1]*B[1,1];C1233 = A[0,1]*B[2,2];C1212 = A[0,1]*B[0,1];C1221 = A[0,1]*B[1,0]
    C2111 = A[1,0]*B[0,0];C2122 = A[1,0]*B[1,1];C2133 = A[1,0]*B[2,2];C2112 = A[1,0]*B[0,1];C2121 = A[1,0]*B[1,0]

    C_ijkl = np.array([[C1111,C1122,C1133,C1112,C1121,0,0,0,0],
                       [C2211,C2222,C2233,C2212,C2221,0,0,0,0],
                       [C3311,C3322,C3333,C3312,C3321,0,0,0,0],
                       [C1211,C1222,C1233,C1212,C1221,0,0,0,0],
                       [C2111,C2122,C2133,C2112,C2121,0,0,0,0],
                       [0    ,  0  ,  0  ,  0  ,  0  ,0,0,0,0],
                       [0    ,  0  ,  0  ,  0  ,  0  ,0,0,0,0],
                       [0    ,  0  ,  0  ,  0  ,  0  ,0,0,0,0],
                       [0    ,  0  ,  0  ,  0  ,  0  ,0,0,0,0]])

    return C_ijkl


def Transform(phi, theta, psi):

    """ Transformation matrix that transform a second order tensor (in
    indicial notations) from global co-ordinates to local co-ordinates.

    A' = T.A.T^-1. """

    
    l1 = np.cos(psi)*np.cos(theta)*np.cos(phi) - np.sin(psi)*np.sin(phi)
    l2 = -np.sin(psi)*np.cos(theta)*np.cos(phi) - np.cos(psi)*np.sin(phi)
    l3 = np.sin(theta)*np.cos(phi)

    m1 = np.cos(psi)*np.cos(theta)*np.sin(phi) + np.sin(psi)*np.cos(phi)
    m2 = -np.sin(psi)*np.cos(theta)*np.sin(phi) + np.cos(psi)*np.cos(phi)
    m3 = np.sin(theta)*np.sin(phi)

    n1 = -np.cos(psi)*np.sin(theta)
    n2 = np.sin(psi)*np.sin(theta)
    n3 = np.cos(theta)

    T = np.array([[ l1, m1, n1],
                  [ l2, m2, n2],
                  [ l3, m3, n3]])

    return T
                      
    
def A2_cont_B2(A,B):
    """Double contraction of two second order tensor would result
    in a scalar.

    ------ A:B = A_ij*B_ij (Einstein's summation in i and j) -------

    INPUT: A, B second order tensors in (3*3) matrix notations. (both ndarray of shape (3,3))
    OUTPUT: a scalar float value."""

    val = (A[0,0]*B[0,0] + A[0,1]*B[0,1] + A[0,2]*B[0,2] +
           A[1,0]*B[1,0] + A[1,1]*B[1,1] + A[1,2]*B[1,2] +
           A[2,0]*B[2,0] + A[2,1]*B[2,1] + A[2,2]*B[2,2] )

    return val

def A4_cont_B2(A,B):
    """Dounble contraction of a fourth order tensor with second order
    tensor will result in a second order tensor.

    -------- A:B = A_ijklB_kl e_i dyad e_j -----------

    INPUT: A: Fourth order tensor in matrix form (9*9).
           B: Second order tensor in matrix form (3*3)
    OUTPUT: Fourth order tensor in matrix form (9*9)

    NOTE: Both the tensor must be in a proper format. 4th order tensor should be full (12,21 etc both terms
    must be there. Not the reduced tensor for inverse transformations."""

    # flatten the B tensor in column for the calculations
    B_col = np.array([B[0,0], B[1,1], B[2,2], B[0,1], B[1,0],B[1,2],B[2,1],B[2,0],B[0,2]])

    C_col = np.matmul(A.T,B_col)
    C = np.array([[C_col[0],C_col[3],C_col[8]],
                  [C_col[4],C_col[1],C_col[5]],
                  [C_col[7],C_col[6],C_col[2]]])

    return C

def A4_cont_B4(A,B):
    """Double contraction of two 4th order tensor will result in a fourth order tensor.

    -------- C_ijrs = A_ijkl*B_klrs ------------

    INPUT:  A: Fourth order tensor in matrix form (9*9)
            B: Fourth order tensor in matrix form (9*9)
    OUTPUT: C: Fourth order tensor in matrix form (9*9)

    Ref: https://www.mate.tue.nl/~peters/4K400/VectTensColMat.pdf """

    C = np.matmul(A.T,B)

    return C


def voigt_to_tensor_Elasticity(C):
    """function to change a fourth order tensor (for example elasticity)
    from voigt notations to indicial notations. C is for plain strain case only (4*4)

    INPUT: C in voigt notations. (4*4) matrix with 12 components such that
        [sig11,sig22,sig33,sig12] = C:[eps11,eps22,eps33,2*eps12]. Sig and eps can be
        any second order tensor that are stress and strain type in voigt notations.

    OUTPUT: A fourth order tensor in matrix form (9*9)

    NOTE: This is a full tensor. (both 12,21 components are present)"""

    Cijkl = np.zeros((9,9))
    Cijkl[0,0] = C[0,0]# 1111 component
    Cijkl[0,1] = C[0,1];Cijkl[1,0] = C[1,0]# 1122 and 2211 components
    Cijkl[0,2] = C[0,2];Cijkl[2,0] = C[2,0]# 1133 and 3311
    Cijkl[0,3] = C[0,3];Cijkl[3,0] = C[3,0];Cijkl[0,4] = C[0,3];Cijkl[4,0] = C[3,0]# 1112, 1211,1121,2111 component

    Cijkl[1,1] = C[1,1]# 2222 component
    Cijkl[1,2] = C[1,2];Cijkl[2,1] = C[2,1]# 2233 and 3322
    Cijkl[1,3] = C[1,3];Cijkl[3,1] = C[3,1];Cijkl[1,4] = C[1,3];Cijkl[4,1] = C[3,1]# 2212, 2221,1222,2122 component

    Cijkl[2,2] = C[2,2] # 3333 component
    Cijkl[2,3] = C[2,3];Cijkl[3,2] = C[3,2];Cijkl[2,4] = C[2,3];Cijkl[4,2] = C[3,2]# 3312, 1233,3321,2133 component

    Cijkl[3,3] = C[3,3]; # 1212 component
    Cijkl[3,4] = C[3,3];Cijkl[4,3] = C[3,3];Cijkl[4,4] = C[3,3]# 1212, 1221, 2112,2121
    

    return Cijkl

def invertible_4order(Cijkl):
    """A function to change a elasticity (or equivalently compliance) type tensors to
    chanage into intertible tensors keeping the equivalent functionality.

    ---- SIG = C:E -----
    SIG = [sig11,sig22,sig33,sig12,sig21,sig23,sig32,sig31,sig13] and
    E = [eps11,eps22,eps33,eps12,eps21,eps23,eps32,eps31,eps13].

    ---- E = S:SIG------ Calculate S if C is given.

    C is in proper format (both major and minor symmetries for C and its matrix form is singular).
    Inverse of C is not straightforward. Therefore, convert C in a form so that the inverse is possible
    and both SIG: C:E and E = S:SIG operations are conserved.
                

    INPUT: Cijkl: A fourth order tensor in full format
    OUTPUT: Cijkl: Fourth order tensor that can be inverted."""

    Cijkl[3,0:3] = Cijkl[3,0:3] + Cijkl[4,0:3];Cijkl[4,0:3] = 0

    Cijkl[0:3,3] = Cijkl[0:3,3] + Cijkl[0:3,4];Cijkl[0:3,4] = 0

    Cijkl[3,3] = 2*Cijkl[3,3]; Cijkl[3,4] = 0
    Cijkl[4,4] = 2*Cijkl[4,4]; Cijkl[4,3] = 0

    return Cijkl

def revert_invertible_4order(Cijkl):
    """From invertible form to again the full tensor form for a fourth order tensor. """
    Cijkl[3,0:3] = 0.5*Cijkl[3,0:3]; Cijkl[4,0:3] = Cijkl[3,0:3]
    Cijkl[0:3,3] = 0.5*Cijkl[0:3,3]; Cijkl[0:3,4] = Cijkl[0:3,3]

    avg = (Cijkl[3,3] + Cijkl[4,4])/2

    Cijkl[3,3] = avg/2; Cijkl[3,4] = avg/2
    Cijkl[4,4] = avg/2; Cijkl[4,3] = avg/2

    return Cijkl

    
def inv_4order(Cijkl):
    """invert a 4th order tensor. In the 2D plain strain case, we took 13,31,23,32 components as zero.
    Therefore, the full matrix (9*9) is not invertible. We use only the 5*5 block to invert.
"""
    Sijkl = np.zeros((9,9))
    C_hat = Cijkl[0:5,0:5]
    
    S_hat = np.linalg.inv(C_hat)
    Sijkl[0:5,0:5] = S_hat

    return Sijkl

def tensor_to_voigt_Elasticity(Cijkl):
    """ For plain strain convert from  tensor(9*9) to voigt notations.
    NOTE: Cijkl is in invertible format. """
    C = np.zeros((4,4))

    C[0,0] = Cijkl[0,0] # 11 component
    C[0,1] = Cijkl[0,1]; C[1,0] = Cijkl[1,0] # 12,21 components
    C[0,2] = Cijkl[0,2]; C[2,0] = Cijkl[2,0] # 13,31 components
    C[0,3] = 0.5*Cijkl[0,3]; C[3,0] = 0.5*Cijkl[3,0] # 16,61 components

    C[1,1] = Cijkl[1,1] # 22 component
    C[1,2] = Cijkl[1,2]; C[2,1] = Cijkl[2,1] ## 23,32 components
    C[1,3] = 0.5*Cijkl[1,3]; C[3,1] = 0.5*Cijkl[3,1]; ## 26,62 compoents

    C[2,2] = Cijkl[2,2] # 33 component
    C[2,3] = 0.5*Cijkl[2,3]; C[3,2] = 0.5*Cijkl[3,2]; ## 36,63 compoents

    C[3,3] = 0.5*Cijkl[3,3]

    return C
    
    
def voigt_to_tensor_strain(E):
    """strain tensor in voigt notation,i.e. [eps11,eps22,eps33,2*eps12]
    convert into a indicial notations. ij, i=1:3 and j = 1:3
    INPUT: E as a vector of (4,) in voigt notations. 
    OUTPUT: 2nd order tensor in matrix form (3*3)"""
    
    Eij = np.zeros((3,3))
    Eij[0,0] = E[0]
    Eij[1,1] = E[1]
    Eij[2,2] = E[2]
    Eij[0,1] = 0.5*E[3]
    Eij[1,0] = 0.5*E[3]

    return Eij

def tensor_to_voigt_strain(Eij):
    """Reverse operation of above function. """

    E = np.zeros((4,))
    E[0] = Eij[0,0]
    E[1] = Eij[1,1]
    E[2] = Eij[2,2]
    E[3] = 2*Eij[0,1]

    return E

def voigt_to_tensor_stress(Sig):
    """stress tensor in voigt notation. [sig11,sig22,sig33,sig12]
    convert into a indicial notations. ij i=1:3, j=1:3

    INPUT: Sig as a vector of (4,) in voigt notations. 
    OUTPUT: 2nd order tensor in matrix form (3*3)"""

    Sig_ij = np.zeros((3,3))
    Sig_ij[0,0] = Sig[0]
    Sig_ij[1,1] = Sig[1]
    Sig_ij[2,2] = Sig[2]
    Sig_ij[0,1] = Sig[3]
    Sig_ij[1,0] = Sig[3]

    return Sig_ij

def tensor_to_voigt_stress(Sig_ij):
    """Reverse operation of above function. """

    Sig = np.zeros((4,))
    Sig[0] = Sig_ij[0,0]
    Sig[1] = Sig_ij[1,1]
    Sig[2] = Sig_ij[2,2]
    Sig[3] = Sig_ij[0,1]

    return Sig



### not important functions
def C_to_S(C):
    """A function to convert a fourth order elasticity tensor into a compliance tensor.
    We take our specific case for orthopropic material in plain strain. It implies that
    1111,1122,1133,2211,2222,2233,3311,3322,3333,1212,1221,2112,2121 are the only nonzero components.
    Rest are zero."""
    S = np.zeros((9,9))
    
    Cppqq = C[0:3,0:3]
    Sppqq = np.linalg.inv(Cppqq)
    Spqrs = np.array([[0.25*(1/C[3,3]),0.25*(1/C[3,4])],
                      [0.25*(1/C[4,3]),0.25*(1/C[4,4])]])
    S[0:3,0:3] = Sppqq
    S[3:5,3:5] = Spqrs

    return S

def voigt_to_tensor_compliance(S):

    Sijkl = np.zeros((9,9))
    Sijkl[0,0] = S[0,0]# 1111 component
    Sijkl[0,1] = S[0,1];Sijkl[1,0] = S[1,0]# 1122 and 2211 components
    Sijkl[0,2] = S[0,2];Sijkl[2,0] = S[2,0]# 1133 and 3311
    Sijkl[0,3] = 0.5*S[0,3];Sijkl[3,0] = 0.5*S[3,0];Sijkl[0,4] = 0.5*S[0,3];Sijkl[4,0] = 0.5*S[3,0]# 1112, 1211,1121,2111 component

    Sijkl[1,1] = S[1,1]# 2222 component
    Sijkl[1,2] = S[1,2];Sijkl[2,1] = S[2,1]# 2233 and 3322
    Sijkl[1,3] = 0.5*S[1,3];Sijkl[3,1] = 0.5*S[3,1];Sijkl[1,4] = 0.5*S[1,3];Sijkl[4,1] = 0.5*S[3,1]# 2212, 2221,1222,2122 component

    Sijkl[2,2] = S[2,2] # 3333 component
    Sijkl[2,3] = S[2,3];Sijkl[3,2] = S[3,2];Sijkl[2,4] = S[2,3];Sijkl[4,2] = S[3,2]# 3312, 1233,3321,2133 component

    Sijkl[3,3] = 0.25*S[3,3]; # 1212 component
    Sijkl[3,4] = 0.25*S[3,3];Sijkl[4,3] = 0.25*S[3,3];Sijkl[4,4] = 0.25*S[3,3]# 1221, 2112,2121

    return Sijkl

