# %load bone_micromechanics_util
import numpy as np
import input_file as inp
import src.tensor_util as tensor
import time
from scipy.integrate import quad
from scipy.optimize import fsolve
import scipy.linalg


"""NOTE: For all the micromechanical calculations, the format for stiffness tensor
is such that [\sig11,\sig22,\sig33,\sig12] = C: [eps11,eps22,eps33,eps12]. It implies that
shear component in C (C[3,3]) is 2*C1212. It allows us to perform all tensor
operations such as contraction, conveniently alike to matrix operations without
carrying Reuter's matrix. However, in Finite element analysis, strain energy in
voigt notation is 0.5*[eps11,eps22,2*eps12]'*C*[eps11,eps22,2*eps12] with C[2,2]
as C1212. Hence, use homogenized stiffnes in this format while using in a FE routine."""

pi = np.pi
I = np.identity(6)
J = np.array([[1.0,1.0,1.0,0,0,0],
              [1.0,1.0,1.0,0,0,0],
              [1.0,1.0,1.0,0,0,0],
              [ 0 , 0 , 0 ,0,0,0],
              [ 0 , 0 , 0 ,0,0,0],
              [ 0 , 0 , 0 ,0,0,0]])


J = (1.0/3.0)*J
K = I - J

def C_Ortho(E1,E2,E3,nu21,nu31,nu32,G23,G31,G12):
    """ Function gives elasticity tensor given all orthotropic material
    constants.
    The elasticity tensor is a relationship between
    [sig_11,sig_22,sig_33,sig_23,sig_13,sig_12] and
    [eps_11,eps_22,eps_33,eps_23,eps_13,eps_12]"""

    S  = np.array([[  1/E1  , -nu21/E2, -nu31/E3,   0      ,     0    ,     0    ],
                   [-nu21/E2,   1/E2  , -nu32/E3,   0      ,     0    ,     0    ],
                   [-nu31/E3, -nu32/E3,   1/E3  ,   0      ,     0    ,     0    ],
                   [   0    ,     0   ,   0     , 1/(2*G23),     0    ,     0    ],
                   [   0    ,     0   ,   0     ,   0      , 1/(2*G31),     0    ],
                   [   0    ,     0   ,   0     ,   0      ,     0    , 1/(2*G12)]])

    return np.linalg.inv(S)


def EngConstant(C):
    """Retrive Engineering constants from elasticity tensor """
    S = np.linalg.inv(C)
    E1   = 1/S[0,0]      ; E2   = 1/S[1,1]      ;  E3  = 1/S[2,2]
    G23  = 1/(2*S[3,3])  ; G13  = 1/(2*S[4,4])  ;  G12 = 1/(2*S[5,5])
    nu32 = -S[1,2]/S[2,2]; nu31 = -S[0,2]/S[2,2]; nu21 = -S[0,1]/S[0,0]

    
    # Abaqus Engineering constants take nu12, nu13, nu23
    nu12 = nu21
    nu13 = (E1/E3)*nu31
    nu23 = (E2/E3)*nu32
    
    ## Check material thermodynamic stablity requirments
    if (E1 > 0 and E2 > 0 and E3 > 0 and G12 > 0 and G13 > 0 and G23 > 0 and
        nu12 < np.sqrt(E1/E2) and nu13 < np.sqrt(E1/E3) and nu23 < np.sqrt(E2/E3) and
        ( 1 - nu21*nu12 - nu23*nu32 - nu31*nu13 - 2*nu21*nu32*nu13) > 0):

        values = np.array([E1,E2,E3,nu12,nu13,nu23,G12,G13,G23])

    else:
        print (" Engineering constants are not thermodyamically consistent")
        values = []

    return values

def P_sph_iso(C_inf):

    S_inf = np.linalg.inv(C_inf)
    E = 1/S_inf[0,0]
    nu = -S_inf[1,0]/S_inf[0,0]

    K_inf = E/(3*(1 - 2*nu)); Mu_inf = E/(2*(1 + nu))

    alpha = (3*K_inf/(3*K_inf + 4*Mu_inf))
    beta = 6*(K_inf + 2*Mu_inf)/(5*(3*K_inf + 4*Mu_inf))

    S_sph_iso = alpha*J + beta*K

    P = np.matmul(S_sph_iso,np.linalg.inv(C_inf))

    return P

def P_cyl_iso(C_inf):
    """ Function to calculate Hill Tensor P for cylinderical inclusions
    embedded into the isotropic matrix.
    The axial direction of the cylinder is 3."""

    S_inf = np.linalg.inv(C_inf)
    E = 1/S_inf[0,0]
    nu = -S_inf[1,0]/S_inf[0,0]

    S11 = (5-4*nu)/(8*(1-nu))
    S22 = S11
    S12 = (-1+4*nu)/(8*(1-nu))
    S21 = S12
    S13 = nu/(2*(1-nu))
    S23 = S13
    S44 = (1/4)
    S55 = (1/4)
    S66 = (3 - 4*nu)/(8*(1-nu))

    S_cyl_iso = np.array([[S11,S12,S13, 0 , 0 , 0],
                          [S21,S22,S23, 0 , 0 , 0],
                          [0 , 0 , 0 , 0 , 0 , 0 ],
                          [0 , 0 , 0 ,2*S44, 0 , 0],
                          [0 , 0 , 0 , 0 ,2*S55, 0],      
                          [0 , 0 , 0 , 0 , 0 ,2*S66]])

    P = np.matmul(S_cyl_iso, np.linalg.inv(C_inf))
    
    return P

def P_cyl_triso(C_inf):

    """ Function to calculate Hill Tensor P for cylinderical inclusions
    embedded into the transversely isotropic matrix. The axial direction
    of cylinder and direction of anisotropy of matrix coincide and is 3.
    1-2 is the direction of isotropy.
    C_inf: Elasticity tensor of matrix with 1-2 as isotropic plane and 3 is
            the direction of anisotropy.
    Output: Hill Tensor"""

    R = np.array([[1, 0, 0, 0  ,   0,   0],
                  [0, 1, 0, 0  ,   0,   0],
                  [0, 0, 1, 0  ,   0,   0],
                  [0, 0, 0, 0.5,   0,   0],
                  [0, 0, 0, 0  , 0.5,   0],
                  [0, 0, 0, 0  ,   0, 0.5]])

    C_inf = np.matmul(C_inf,R)
    D2 = C_inf[0,0] - C_inf[0,1]

    P11 = (1/8)*((5*C_inf[0,0] - 3*C_inf[0,1])/C_inf[0,0]/D2)
    P22 = P11
    P12 = (-1/8)*((C_inf[0,0] + C_inf[0,1])/C_inf[0,0])/D2
    P21 = P12
    P44 = 1/(8*C_inf[3,3])
    P44 = 2*P44
    P55 = P44
    P66 = (1/8)*((3*C_inf[0,0]-C_inf[0,1])/C_inf[0,0]/D2)
    P66 = 2*P66

    P = np.array([[ P11 , P12 , 0 , 0 , 0 , 0 ],
                  [ P21 , P22 , 0 , 0 , 0 , 0 ],
                  [  0  ,  0  , 0 , 0 , 0 , 0 ],
                  [  0  ,  0  , 0 ,P44, 0 , 0 ],
                  [  0  ,  0  , 0 , 0 ,P55, 0 ],
                  [  0  ,  0  , 0 , 0 , 0 ,P66]])

    return P

def P_spheroid_triso(C_inf,e):
    """Function to calculate Hill Tensor P when inclusion is embedded into
    the transversely isotropic matrix with C_inf as material elasticity tensor (see Appendix A.3, Hellmich, 2004)

    Input:  C_inf: material elasticity tensor with 1-2 as isotropic plane and 3 is the direction
            of anisotropy.
            e = elongation ratio of spheroid (ratio of the length of the axial to transverse axis of the
            spheroid
    Output: Hill tensor """

    R = np.array([[1, 0, 0, 0,   0,   0],
                  [0, 1, 0, 0,   0,   0],
                  [0, 0, 1, 0,   0,   0],
                  [0, 0, 0, 0.5, 0,   0],
                  [0, 0, 0, 0, 0.5,   0],
                  [0, 0, 0, 0,   0, 0.5]])

    C_inf = np.matmul(C_inf,R)

    p = [C_inf[2,2]*C_inf[3,3], -(C_inf[0,0]*C_inf[2,2] - 2*C_inf[0,2]*C_inf[3,3] - C_inf[0,2]*C_inf[0,2]), C_inf[0,0]*C_inf[3,3]]

    gamma = np.roots(p)
    gamma1 = gamma[0]
    gamma2 = gamma[1]
    def fun_I1(x, l, m, n):
        return (1/(C_inf[2,2]*C_inf[3,3]))*(e/(1 - (1-e**2.)*x**2.)**1.5)*( l + m*x**2. + n*x**4.)/((gamma1 + (1 - gamma1)*x**2.)*(gamma2 + (1-gamma2)*x**2.))

    def fun_I2(x, l, m):
        return (e/(1 - (1-e**2.)*x**2.)**1.5)*( l + m*x**2.)/(C_inf[5,5] + (C_inf[3,3] - C_inf[5,5])*x**2.)
    
    P11 = (3/16)*quad(fun_I1, -1,  1, args=(C_inf[3,3], C_inf[2,2] - 2*C_inf[3,3], C_inf[3,3] - C_inf[2,2]))[0] + (1/16)*quad(fun_I2,-1,1, args = (1,-1))[0]
    P12 = (1/16)*quad(fun_I1, -1,  1, args=(C_inf[3,3], C_inf[2,2] - 2*C_inf[3,3], C_inf[3,3] - C_inf[2,2]))[0] - (1/16)*quad(fun_I2,-1,1, args = (1,-1))[0]     
    P13 = (1/4)*quad( fun_I1, -1,  1, args = (0, - C_inf[0,2] - C_inf[3,3],C_inf[0,2] + C_inf[3,3]))[0]
    P33 = (1/2)*quad( fun_I1, -1,  1, args = (0,  C_inf[0,0], C_inf[3,3] - C_inf[0,0]))[0]
    P55 = (1/16)*quad(fun_I1, -1,  1, args=(C_inf[0,0], -2*(C_inf[0,0] + C_inf[0,2]), C_inf[0,0] + C_inf[2,2] + 2*C_inf[0,2]))[0] - (1/16)*quad(fun_I2,-1,1, args = (0,-1))[0]

    P21 = P12; P31 = P13; P22 = P11; P23 = P31; P32 = P23; P66 = (P11 - P12);P55 = 2*P55; P44 = P55

    P = np.array([[P11, P12, P13, 0,   0,    0],
                  [P21, P22, P23, 0,   0,    0],
                  [P31, P32, P33, 0,   0,    0],
                  [ 0 ,  0 ,  0 , P44, 0,    0],
                  [ 0 ,  0 ,  0 ,  0 , P55,  0],
                  [ 0 ,  0 ,  0 ,  0 , 0  , P66]])

    return P


def Transform(phi, theta, psi):
    """Section 6.2 Analysis and Design Principles of MEMS Devices
    Transformation matrix for co-ordinate transformation of a second order tensor written in
    stress like voigt notations. Sig_{local} = T*Sig_{global}

    [sig11,sig22,sig33,sig23,sig13,sig12]' = T*[sig11,sig22,sig33,sig23,sig13,sig12]

    
    phi, theta, psi are Euler's angles defined in the reference. T is eq 6.2.21"""

    l1 = np.cos(psi)*np.cos(theta)*np.cos(phi) - np.sin(psi)*np.sin(phi)
    l2 = -np.sin(psi)*np.cos(theta)*np.cos(phi) - np.cos(psi)*np.sin(phi)
    l3 = np.sin(theta)*np.cos(phi)

    m1 = np.cos(psi)*np.cos(theta)*np.sin(phi) + np.sin(psi)*np.cos(phi)
    m2 = -np.sin(psi)*np.cos(theta)*np.sin(phi) + np.cos(psi)*np.cos(phi)
    m3 = np.sin(theta)*np.sin(phi)

    n1 = -np.cos(psi)*np.sin(theta)
    n2 = np.sin(psi)*np.sin(theta)
    n3 = np.cos(theta)
    
    T = np.array([[l1**2.  , m1**2.  , n1**2.  ,  2*m1*n1    ,   2*n1*l1   , 2*l1*m1],
                  [l2**2.  , m2**2.  , n2**2.  ,  2*m2*n2    ,   2*n2*l2   , 2*l2*m2],
                  [l3**2.  , m3**2.  , n3**2.  ,  2*m3*n3    ,   2*n3*l3   , 2*l3*m3],
                  [l2*l3   , m2*m3   , n2*n3   , m2*n3+m3*n2 , n2*l3+n3*l2 , m2*l3+m3*l2],
                  [l3*l1   , m3*m1   , n3*n1   , m3*n1+m1*n3 , n3*l1+n1*l3 , m3*l1+m1*l3],
                  [l1*l2   , m1*m2   , n1*n2   , m1*n2+m2*n1 , n1*l2+n2*l1 , m1*l2+m2*l1]])

    return T


def C_MT(phi_ic, c_ic, P_ic,c_m):
    """ MORI-TANAKA Homogenization scheme.
    The RVE has inclusions with given elongation embedded into the
    matrix of material m.
    INPUT: phi_ic: inclusion volume fraction
           c_ic: inclusion elasticity tensor
           P_ic: Hill tensor for the inclusions 
           c_m: matrix elasticty tensor 
    OUTPUT: C_MT: Homegenized elasticity tensor of the RVE"""
    I = np.identity(6)
    A_ic = np.matmul(np.linalg.inv(I + np.matmul(P_ic,(c_ic - c_m))), np.linalg.inv((1-phi_ic)*I + phi_ic*(np.linalg.inv(I + np.matmul(P_ic,(c_ic - c_m))))))
    A_m = np.linalg.inv((1-phi_ic)*I + phi_ic*(np.linalg.inv(I + np.matmul(P_ic,(c_ic - c_m)))))

    C_MT = phi_ic*np.matmul(c_ic,A_ic) + (1-phi_ic)*np.matmul(c_m,A_m)    

    return C_MT


def bone_C_hom(f_col_bar, f_HA_bar, theta_lac, MAT):
    #start_tot = time.perf_counter()
    pi = np.pi
    I = np.identity(6)
    J = np.array([[1.0, 1.0, 1.0, 0, 0, 0],
                  [1.0, 1.0, 1.0, 0, 0, 0],
                  [1.0, 1.0, 1.0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]])

    J /= 3.0
    K = I - J

    # conversion of volume fractions for five length scales
    f_lac = 0.021
    d_s = 1.26
    f_fib = f_col_bar * (1.47 * 64 * 5 * d_s) / 335.6

    phi_HA_ef = (1 - f_fib) / (1 - f_col_bar)
    f_HA_foam = (phi_HA_ef * f_HA_bar) / (1 - f_fib)
    f_HA = ((1 - phi_HA_ef) * f_HA_bar) / f_fib

    f_wetcol = 1 - f_HA
    f_col = f_col_bar / f_wetcol

    # Material properties of collagen and ultra-structural water and HA minerals
    c_3333_col = 17.9
    c_1133_col = 7.1
    c_1111_col = 11.7
    c_1122_col = 5.1
    c_1313_col = 3.3

    C_col = np.array([[c_1111_col, c_1122_col, c_1133_col, 0, 0, 0],
                      [c_1122_col, c_1111_col, c_1133_col, 0, 0, 0],
                      [c_1133_col, c_1133_col, c_3333_col, 0, 0, 0],
                      [0, 0, 0, 2 * c_1313_col, 0, 0],
                      [0, 0, 0, 0, 2 * c_1313_col, 0],
                      [0, 0, 0, 0, 0, c_1111_col - c_1122_col]])

    # ultra-structural water
    k_w = 2.3  # bulk modulus
    C_im = 3 * k_w * J
    C_w = C_im  # assuming ultra-structural water properties for C_w

    # HA minerals
    k_HA = 82.6
    mu_HA = 44.9
    C_HA = 3 * k_HA * J + 2 * mu_HA * K

    # Wet Collagen
    phi_ic = 1 - f_col
    c_m = C_col
    start = time.perf_counter()
    P_ic = P_cyl_triso(c_m)
    c_ic = C_im

    C_wetcol = C_MT(phi_ic, c_ic, P_ic, c_m)
    #stop = time.perf_counter()
    #print("Time taken in calculating C_wetcol:", stop - start)

    # Mineralized collagen fibril
    f_wetcol = 1 - f_HA

    C_inf = C_wetcol
    e = 1  # Spherical inclusions of water and HA crystals.

    #start = time.perf_counter()
    for i in range(100):
        P_HA = P_spheroid_triso(C_inf, e)
        P_wetcol = P_cyl_triso(C_inf)

        inv_I_P_HA = np.linalg.inv(I + np.matmul(P_HA, (C_HA - C_inf)))
        inv_I_P_wetcol = np.linalg.inv(I + np.matmul(P_wetcol, (C_wetcol - C_inf)))

        f_HA_inv = f_HA * inv_I_P_HA
        f_wetcol_inv = f_wetcol * inv_I_P_wetcol

        A_HA = np.matmul(inv_I_P_HA, np.linalg.inv(f_HA_inv + f_wetcol_inv))
        A_wetcol = np.matmul(inv_I_P_wetcol, np.linalg.inv(f_HA_inv + f_wetcol_inv))

        C_fib = f_HA * np.matmul(C_HA, A_HA) + f_wetcol * np.matmul(C_wetcol, A_wetcol)  # C_SCSI

        error = np.linalg.norm(C_fib - C_inf)
        C_inf = C_fib
        if error < 1e-8:
            break
    #stop = time.perf_counter()
    #print("Time taken in calculating C_fib:", stop - start)

    # Extrafibrillar space
    f_w_foam = 1 - f_HA_foam
    C_inf = C_HA
    start = time.perf_counter()
    for i in range(100):
        P_w = P_sph_iso(C_inf)
        P_HA = P_sph_iso(C_inf)

        inv_I_P_w = np.linalg.inv(I + np.matmul(P_w, (C_w - C_inf)))
        inv_I_P_HA = np.linalg.inv(I + np.matmul(P_HA, (C_HA - C_inf)))

        f_w_foam_inv = f_w_foam * inv_I_P_w
        f_HA_foam_inv = f_HA_foam * inv_I_P_HA

        A_w = np.matmul(inv_I_P_w, np.linalg.inv(f_w_foam_inv + f_HA_foam_inv))
        A_HA = np.matmul(inv_I_P_HA, np.linalg.inv(f_w_foam_inv + f_HA_foam_inv))

        C_ef = f_w_foam * np.matmul(C_w, A_w) + f_HA_foam * np.matmul(C_HA, A_HA)  # C_SCSII

        error = np.linalg.norm(C_ef - C_inf)
        C_inf = C_ef
        if error < 1e-8:
            break
    #stop = time.perf_counter()
    #print("Time taken in calculating C_ef:", stop - start)

    # Ultrastructure
    phi_ic = f_fib
    c_m = C_ef
    #start = time.perf_counter()
    P_ic = P_cyl_triso(c_m)
    c_ic = C_fib

    C_ultra = C_MT(phi_ic, c_ic, P_ic, c_m)
    #stop = time.perf_counter()
    #print("Time taken in calculating C_ultra:", stop - start)

    # Extravascular
    phi_ic = f_lac
    C_lac = 3 * k_w * J
    c_m = C_ultra
    #start = time.perf_counter()
    P_ic = P_spheroid_triso(c_m, e)
    c_ic = C_lac
    C_exvas = C_MT(phi_ic, c_ic, P_ic, c_m)
    #stop = time.perf_counter()
    #print("Time taken in calculating C_exvas:", stop - start)

    v = [0, 1, 5]
    C = C_exvas[np.ix_(v, v)]
    C[:, 2] = 0.5 * C[:, 2]

    R = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 2.0]])  # Reuter's matrix
    # Theta_lac is the angle that the inclusion orientation makes from the X axes (varied between 0 to pi)
    c = np.cos(theta_lac)
    s = np.sin(theta_lac)
    T = np.array([[c ** 2, s ** 2, 2 * s * c],
                  [s ** 2, c ** 2, -2 * s * c],
                  [-s * c, s * c, c ** 2 - s ** 2]])

    T_ = np.matmul(R, np.matmul(T, np.linalg.inv(R)))
    C_hom = np.matmul(np.linalg.inv(T), np.matmul(C, T_))
    #stop_tot = time.perf_counter()
    #print("Time taken in calculating C_hom:", stop_tot - start_tot)
    return C_hom



def del_C_by_rho_bone(f_col_bar, f_HA_bar, theta_lac):
    """Evaluate the derivative of homogenized elasticity tensor with respect to rho.
    This function is used in the calculation of senstivities. """

    #normalized density
    rho_HA = 1;  rho_col = 1.41/3; rho_w = 1/3
    alpha = 1/0.9
    rho_ec = (1- alpha*f_col_bar - f_HA_bar)*rho_w + alpha*f_col_bar*rho_col + f_HA_bar*rho_HA
    f_lac = 0.021
    #del_phiA_by_rho = 1/(rho_A - rho_M)
    #del_gammaC_by_rho = 1/((rho_C - rho_B)*(1 - phi_A))

    del_f_col_by_rho = 1/(alpha*(rho_col - rho_w)*(1 - f_lac))
    del_f_HA_by_rho = 1/((rho_HA - rho_w)*(1 - f_lac))
    
    if (f_col_bar <= 0.4):
        #forward finite difference
        del_C_by_f_col = (bone_C_hom(f_col_bar + 0.005, f_HA_bar, theta_lac) -
                     bone_C_hom(f_col_bar, f_HA_bar, theta_lac))/0.005
    else:
        #backward finite difference at upper limit
        del_C_by_f_col = (bone_C_hom(f_col_bar, f_HA_bar, theta_lac) -
                     bone_C_hom(f_col_bar - 0.005,f_HA_bar, theta_lac))/0.005
        
    if (f_HA_bar <= 0.545):
        #forward finite difference
        del_C_by_f_HA = (bone_C_hom(f_col_bar, f_HA_bar + 0.005, theta_lac) -
                       bone_C_hom(f_col_bar, f_HA_bar, theta_lac))/0.005
    else:
        del_C_by_f_HA = (bone_C_hom(f_col_bar, f_HA_bar, theta_lac) -
                       bone_C_hom(f_col_bar, f_HA_bar - 0.005, theta_lac))/0.005
        
    del_C_by_rho = del_C_by_f_HA*del_f_HA_by_rho + del_C_by_f_col*del_f_col_by_rho

    return del_C_by_rho

