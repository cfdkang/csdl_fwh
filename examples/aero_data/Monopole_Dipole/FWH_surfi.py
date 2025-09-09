import csdl_alpha as csdl
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define pi
pi = np.pi

def FWH_Surfi_def(r_wall, N_theta, N_phi, N_FWHsurfi):

    theta  = np.linspace( pi/2/N_theta, (pi)-pi/2/N_theta, N_theta); # cell center
    phi    = np.linspace( pi/N_phi, (2*pi)-pi/N_phi, N_phi);         # cell center

    dtheta = theta[1]- theta[0]
    dphi   = phi[1] - phi[0]

    X  = np.zeros([N_theta, N_phi])
    Y  = np.zeros([N_theta, N_phi])
    Z  = np.zeros([N_theta, N_phi])
    dS = np.zeros([N_theta, N_phi])

    for i in range(N_theta):
        for j in range(N_phi):
            X[i,j]  = r_wall *np.sin(theta[i]) *np.cos(phi[j])
            Y[i,j]  = r_wall *np.sin(theta[i]) *np.sin(phi[j])
            Z[i,j]  = r_wall *np.cos(theta[i])
            dS[i,j] = r_wall**2*np.sin(theta[i])*dtheta*dphi # Unit area


    # Unit normal vector
    nx = X/r_wall
    ny = Y/r_wall
    nz = Z/r_wall
    
    
    # # PLOT FW-H SURFACE & UNIT NORMAL VECTOR (Python)
    # fig = plt.figure()
    # ax  = fig.add_subplot(111,projection='3d')
    
    # ax.plot_surface(X,Y,Z)
    # ax.quiver(X,Y,Z,nx,ny,nz,color='k',length=0.2)
    # ax.set_xlabel('X'), ax.set_xlabel('Y'), ax.set_xlabel('Z') 
    # plt.show()
    

    # Reshape into single column for easy data treatment: [Xs, Ys, Zs, Rs, nxs, nys, nzs, dS]
    Xs  = np.reshape(X, [N_FWHsurfi,1])
    Ys  = np.reshape(Y, [N_FWHsurfi,1])
    Zs  = np.reshape(Z, [N_FWHsurfi,1])

    nxs = np.reshape(nx, [N_FWHsurfi,1])  #% Not used...
    nys = np.reshape(ny, [N_FWHsurfi,1])
    nzs = np.reshape(nz, [N_FWHsurfi,1])

    dS  = np.reshape(dS, [N_FWHsurfi,1])
    
    return Xs,Ys,Zs,nxs,nys,nzs,dS
    # ------
