import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import fem_util as fem



def plot_u(x0, y0, x1, y1, nx, ny, u_sol):

    X, Y = np.meshgrid(np.linspace(x0,x1,nx+1), np.linspace(y0,y1,ny+1))
    x_shape = X.shape
    U = u_sol[0::2]
    U = U.reshape(x_shape)
    V = u_sol[1::2]
    V = V.reshape(x_shape)

    u_mag = np.sqrt(U**2. + V**2.)

    fig = plt.figure()
    Q = plt.quiver(X, Y, U, V, u_mag ,cmap='coolwarm')#, norm=colors.LogNorm(vmin=u_mag.min(),vmax=u_mag.max()))
    fig.colorbar(Q,extend='max')
    plt.gca().set_aspect('equal')
    plt.show()

    fig, (ax_l, ax_c, ax_r) = plt.subplots(nrows=3, ncols=1,
                                       sharex=True, figsize=(20, 20))

    ax_l.set_title('ux')
    im1 = ax_l.pcolor(X, Y, U, cmap='RdBu', vmin = np.min(U) , vmax = np.max(U))
    fig.colorbar(im1, ax=ax_l)
    
    ax_c.set_title('uy')
    im2 = ax_c.pcolor(X, Y, V, cmap='RdBu', vmin = np.min(V) , vmax = np.max(V))
    fig.colorbar(im2, ax=ax_c)
    
     
    ax_r.set_title('u mag')
    im3 = ax_r.pcolor(X, Y, u_mag, cmap='RdBu', vmin = 0 , vmax = np.abs(u_mag).max())
    fig.colorbar(im3, ax=ax_r)

    plt.xlim(x0, x1)
    plt.ylim(y0, y1)
    plt.gca().set_aspect('equal')
    plt.show()

def plot_theta(x0, y0, x1, y1, nx, ny, theta,rho):

    
    dx = (x1-x0)/nx
    dy = (y1-y0)/ny

    n_ele = theta.shape[0]
    
    X, Y = np.meshgrid(np.linspace(dx/4,x1-dx/4,2*nx), np.linspace(dy/4,y1-dy/4,2*ny))
    x_shape = X.shape

    tht = np.zeros(x_shape)
    U = np.zeros(x_shape)
    V = np.zeros(x_shape)
    
    for el in range(0,n_ele):
        theta_el = theta[el].reshape((2,2))
        theta_el[[0,1]] = theta_el[[1,0]]
        
        ind = int(el/nx)

        #tht[ind*2:ind*2+2,(el-ind*nx)*2:(el-ind*nx)*2+2] = theta_el
        U[ind*2:ind*2+2,(el-ind*nx)*2:(el-ind*nx)*2+2] = np.cos(theta_el)*rho[el]
        V[ind*2:ind*2+2,(el-ind*nx)*2:(el-ind*nx)*2+2] = np.sin(theta_el)*rho[el]

    plt.figure(1)
    plt.quiver(X, Y, U, V, units='xy' ,scale = 20, color='blue',width=0.007, headwidth=2., headlength=3.)
    plt.xlim(x0, x1)
    plt.ylim(y0, y1)
    plt.xticks(np.linspace(x0,x1,nx+1)) 
    plt.yticks(np.linspace(y0,y1,ny+1))
    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.show()

    #reorder the theta as original
    theta[:,[0,2]] = theta[:,[2,0]]
    theta[:,[1,3]] = theta[:,[3,1]]
    

def plot_rho(x0, y0, x1, y1, nx, ny, rho):

    dx = (x1-x0)/nx
    dy = (y1-y0)/ny


    X, Y = np.meshgrid(np.linspace(dx/2,x1-dx/2,nx), np.linspace(dy/2,y1-dy/2,ny))
    x_shape = X.shape

    rho = rho.reshape(x_shape)
    
    fig, ax = plt.subplots(nrows=1, ncols=1,sharex=True, figsize=(10, 10))

    ax.set_title('density')
    im1 = ax.pcolor(X, Y, rho, cmap='rainbow')#,vmin=0, vmax=1)
    fig.colorbar(im1, ax=ax)    

    plt.xlim(x0, x1)
    plt.ylim(y0, y1)
    plt.gca().set_aspect('equal')
    plt.show()

    plt.pause(0.2)
    plt.close()

def plot_strains(u_sol,rho,x0, y0, x1, y1, nx, ny,nodes, elem):

    dx = (x1-x0)/nx
    dy = (y1-y0)/ny
    
    X, Y = np.meshgrid(np.linspace(dx/2,x1-dx/2,nx), np.linspace(dy/2,y1-dy/2,ny))
    x_shape = X.shape

    n_ele = elem.shape[0]

    # displacement vector for each element
    u_ele = [u_sol[[2*elem[el,1], 2*elem[el,1] + 1, 2*elem[el,2], 2*elem[el,2] + 1, 2*elem[el,3], 2*elem[el,3] + 1, 2*elem[el,4], 2*elem[el,4] + 1]] for el in range(n_ele)]

    # to visualize only the solid part, penalize the dispacement with density
    u_ele = [u_ele[el]*rho[el] for el in range(n_ele)]
    
    strains = [list(fem.strain([0.0,0.0],nodes[elem[y][1:5]][:,1:3],u_ele[y]))  for y in range(n_ele)]
    strains =  np.asarray(strains)

    #print (strains)

    Exx = strains[:,0]; Exx = Exx.reshape(x_shape)
    Eyy = strains[:,1]; Eyy = Eyy.reshape(x_shape)
    Exy = strains[:,2]; Exy = Exy.reshape(x_shape)

    fig, (ax_l, ax_c, ax_r) = plt.subplots(nrows=1, ncols=3,
                                       sharey=True, figsize=(20, 20))

    ax_l.set_title('Exx')
    ax_l.set_aspect('equal')
    ax_l.set_xlim(x0,x1);ax_l.set_ylim(y0,y1)
    im1 = ax_l.pcolor(X, Y, Exx, cmap='RdBu', vmin = np.min(Exx) , vmax = np.max(Exx))
    fig.colorbar(im1, ax=ax_l,fraction=0.046, pad=0.04)

    
    
    ax_c.set_title('Eyy')
    ax_c.set_aspect('equal')
    ax_c.set_xlim(x0,x1);ax_c.set_ylim(y0,y1)
    im2 = ax_c.pcolor(X, Y, Eyy, cmap='RdBu', vmin = np.min(Eyy) , vmax = np.max(Eyy))
    fig.colorbar(im2, ax=ax_c, fraction=0.046, pad=0.04)
    
     
    ax_r.set_title('Exy')
    ax_r.set_aspect('equal')
    ax_r.set_xlim(x0,x1);ax_r.set_ylim(y0,y1)
    im3 = ax_r.pcolor(X, Y, Exy, cmap='RdBu', vmin = np.min(Exy) , vmax = np.max(Exy))
    fig.colorbar(im3, ax=ax_r, fraction=0.046, pad=0.04)

    plt.show()

def plot_volumefraction(x0, y0, x1, y1, nx, ny, phi_A):

    dx = (x1-x0)/nx
    dy = (y1-y0)/ny

    n_ele = phi_A.shape[0]

    X, Y = np.meshgrid(np.linspace(dx/4,x1-dx/4,2*nx), np.linspace(dy/4,y1-dy/4,2*ny))
    x_shape = X.shape

    phi = np.zeros(x_shape)

    for el in range(0,n_ele):
        phi_el = phi_A[el].reshape((2,2))
        phi_el[[0,1]] = phi_el[[1,0]]
        
        ind = int(el/nx)

        phi[ind*2:ind*2+2,(el-ind*nx)*2:(el-ind*nx)*2+2] = phi_el

    fig, ax = plt.subplots(nrows=1, ncols=1,sharex=True, figsize=(10, 10))

    ax.set_title('density')
    im1 = ax.pcolor(X, Y, phi, cmap='rainbow')#,vmin=0, vmax=1)
    fig.colorbar(im1, ax=ax)    

    plt.xlim(x0, x1)
    plt.ylim(y0, y1)
    plt.gca().set_aspect('equal')
    plt.show()


    
    
    
    
    
    

    
