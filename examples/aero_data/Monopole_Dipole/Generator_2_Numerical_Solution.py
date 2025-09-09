import csdl_alpha as csdl
import numpy as np

import matplotlib.pyplot as plt

from .Garrick_Triangle import Garrick_Triangle  # Garrick Triangle
from .Time_Derivative_1D  import Time_Derivative_1D   # Time derivative
from .rms              import rms
from .FWH_surfi        import FWH_Surfi_def     # FW-H Surface

# Define pi
pi = np.pi

# # ...
# file_path = './Source_Input.txt'
# # -------------------------
# # User defined parameters
# # -------------------------
# RAW = []
# with open(file_path,'r') as f:
#     for i, line in enumerate(f):        
#         if i < 4:
#             # (1) Divide string and value by space and then (2) take only value
#             RAW.append(line.split()[1]) 

# A         = float(RAW[0])  # Monopole or dipole source strength (amplitude)
# f         = float(RAW[1])  # Monopole or dipole pulsation [source] frequency [Hz]
# omega     = 2*pi*f         # Monopole or dipole pulsation [source] angular frequency [rad/s]

# R_rot     = float(RAW[2])  # Rotational radius
# f_rot     = float(RAW[3])  # Rotational frequency [Hz]
# omega_rot = 2*pi*f_rot     # Rotational angular frequency [rad/s]
# # ...

A         = 1          # Monopole or dipole source strength (amplitude)
f         = 5          # Monopole or dipole pulsation [source] frequency [Hz]
omega     = 2*pi*f     # Monopole or dipole pulsation [source] angular frequency [rad/s]

R_rot     = 0.7        # Rotational radius
f_rot     = 1          # Rotational frequency [Hz]
omega_rot = 2*pi*f_rot # Rotational angular frequency [rad/s]



# << OUTPUT >>
# [Ys_vec_1, Ys_vec_2, Ys_vec_3]: Source vector including the rotation effect
# [n_hat_1,  n_hat_2,  n_hat_3 ]: Unit normal vectors
# dS                            : Unit surface area 
# N_FWHsurfi                    : Numer of discretized surfaces

# -- ARRAY --
# [Ux, Uy, Uz]   : spatial derivative of fai + convective mean flow : U0,i + u'
# p_timederi     : time    derivative of fai
# pl_spatialderi : spaital derivative of fai  
# p              : p_timederi+pl_spatialderi  : p'
# [px, py, pz]   : p' delta_{ij} * n_{j}
# rho            : total density              : rho_0 + rho'

# << INPUT >>
# rho_0          : mean density
# U01            : streamwise  convective mean flow
# U02            : wall-normal convective mean flow
# M0             : freestream Mach number
# AoA            : angle of attack [deg]
# t              : source time [sec]
# dt             : source time step [sec]
# c              : speed of sound [m/s]
# A              : monopole or dipole source strength
# omega          : monopole or dipole source angular frequency
# Solution_Option: "Monopole" or "Dipole"

# XXXXXX THIS IS EASILY MISTAKBLE XXXXX
# Ux = Ux + U_mov_1(i,:) + U01;
# Uy = Uy + U_mov_2(i,:) + U02;
# Uz = Uz + U_mov_3(i,:);
# XXXXXX THIS IS EASILY MISTAKBLE XXXXX
    
def Generator_2_Numerical_Solution_def(rho_0, U01, U02, M0, AoA, t, dt, c, Solution_Option):
    
    # --------------- MOVING Source (function of source time ) ---------------
    
    # ------------
    # Moving Term
    # ------------
    Nt = len(t)
    y_mov_1 = R_rot*np.cos(omega_rot*t) # % 1 by Nt
    y_mov_2 = R_rot*np.sin(omega_rot*t)
    y_mov_3 = np.zeros(Nt)

    # ------------------------------------------------------
    # FW-H Surface Modeling (replaced with VLM solver later)
    # ------------------------------------------------------
    # 1) FW-H surface (Sperical system)
    r_wall     = 0.7
    N_theta    = 18
    N_phi      = 36

    # N_theta    = 15
    # N_phi      = 10

    N_FWHsurfi = N_theta*N_phi

    Xs,Ys,Zs,nxs,nys,nzs,dS = FWH_Surfi_def(r_wall, N_theta, N_phi, N_FWHsurfi) # Same as Matlab

    y_vec_1 = np.ones([N_FWHsurfi, Nt]) * Xs # % N_FWHsurfi by Nt
    y_vec_2 = np.ones([N_FWHsurfi, Nt]) * Ys
    y_vec_3 = np.ones([N_FWHsurfi, Nt]) * Zs

    # --------------
    # Unit normals
    # --------------
    y_vec_mag = np.sqrt(y_vec_1**2 + y_vec_2**2 + y_vec_3**2)
    n_hat_1 = y_vec_1 / y_vec_mag # % N_FWHsurfi by Nt
    n_hat_2 = y_vec_2 / y_vec_mag
    n_hat_3 = y_vec_3 / y_vec_mag

    # CHECK: magnitude of unity: plt.plot(np.sqrt(n_hat_1**2 + n_hat_2**2 + n_hat_3**2))

    # --------------- Define the FW-H Surface ---------------

    
    # ----------------------------------------------------------
    # OUTPUT1: Source vector (ADD THE MOVING TERM (BLADE KINEMATICS)
    # ----------------------------------------------------------
    Ys_vec_1 = y_vec_1 + y_mov_1
    Ys_vec_2 = y_vec_2 + y_mov_2
    Ys_vec_3 = y_vec_3 + y_mov_3
    
    # ---------------------------------------
    # OUTPUT2: Initialize the flow variables
    # ---------------------------------------
    ARRAY_Ux             = np.zeros([N_FWHsurfi, Nt])
    ARRAY_Uy             = np.zeros([N_FWHsurfi, Nt])
    ARRAY_Uz             = np.zeros([N_FWHsurfi, Nt])
    
    ARRAY_p_timederi     = np.zeros([N_FWHsurfi, Nt])
    ARRAY_pl_spatialderi = np.zeros([N_FWHsurfi, Nt])
    ARRAY_p              = np.zeros([N_FWHsurfi, Nt])
    
    ARRAY_px             = np.zeros([N_FWHsurfi, Nt])
    ARRAY_py             = np.zeros([N_FWHsurfi, Nt])
    ARRAY_pz             = np.zeros([N_FWHsurfi, Nt])
    
    ARRAY_rho            = np.zeros([N_FWHsurfi, Nt])

    
    for i in range(N_FWHsurfi):
        
        # Coordinate system [FWH_Surface]
        X1 = y_vec_1[i,:]
        Y1 = y_vec_2[i,:]
        Z1 = y_vec_3[i,:]
    
    
        if Solution_Option == 'MONOPOLE':
            
            # //////////////////// Numerical Monopole ////////////////////
            # -----------------------------------
            # Define velocity potential: monopole
            # -----------------------------------

            # ----------------- VELOCITY -----------------
            # velocity: get spatial derivative by very small perturbation
            perturb = 1E-08
            
            #  dfai/dx1 - streamwise
            Rs1_plus ,   R1_plus, _, _, _ = Garrick_Triangle(M0, AoA, X1+perturb, Y1, Z1 )
            Rs1_minus, R1_minus , _, _, _ = Garrick_Triangle(M0, AoA, X1-perturb, Y1, Z1 )

            fai1_plus   = A/(4*pi*Rs1_plus )*np.exp( 1j*omega*( t-R1_plus/c) )
            fai1_minus  = A/(4*pi*Rs1_minus)*np.exp( 1j*omega*( t-R1_minus/c) )

            Ux = (fai1_plus - fai1_minus) / (2*perturb)

            #  dfai/dx2 - wall-normal
            Rs2_plus ,   R2_plus, _, _, _ = Garrick_Triangle(M0, AoA, X1, Y1+perturb, Z1 )
            Rs2_minus, R2_minus , _, _, _ = Garrick_Triangle(M0, AoA, X1, Y1-perturb, Z1 )

            fai2_plus  = A/(4*pi*Rs2_plus )*np.exp( 1j*omega*( t-R2_plus/c) )
            fai2_minus = A/(4*pi*Rs2_minus)*np.exp( 1j*omega*( t-R2_minus/c) )

            Uy = (fai2_plus - fai2_minus) / (2*perturb)

            #  dfai/dx3 - spanwise
            Rs3_plus , R3_plus , _, _, _ = Garrick_Triangle(M0, AoA, X1, Y1, Z1+perturb )
            Rs3_minus, R3_minus, _, _, _ = Garrick_Triangle(M0, AoA, X1, Y1, Z1-perturb )

            fai3_plus  = A/(4*pi*Rs3_plus )*np.exp( 1j*omega*( t-R3_plus/c)  )
            fai3_minus = A/(4*pi*Rs3_minus)*np.exp( 1j*omega*( t-R3_minus/c) )

            Uz = (fai3_plus - fai3_minus) / (2*perturb)

            # fluctuating velocity components
            Ux=Ux.real
            Uy=Uy.real
            Uz=Uz.real
            # ----------------- VELOCITY -----------------


            # ----------------- PRESSURE & DENSITY -----------------

            # fluctuating pressure & fluctuating density
            Rs, R0, _, _, _ = Garrick_Triangle(M0, AoA, X1, Y1, Z1 )
            fai = A/(4*pi*Rs)*np.exp( 1j*omega*( t-R0/c) )
            # p_timederi     = -real( rho_0*(1i*omega).* fai  ); % Time-derivative

            # Same as the analytical way for the time-derivative term
            dRsdt = Time_Derivative_1D(Rs,dt)
            dR0dt = Time_Derivative_1D(R0,dt)
            p_timederi = -dRsdt/(Rs**2) * A/(4*pi*Rs) * np.exp(1j*omega*(t-R0/c)) + \
                                        A/(4*pi*Rs) *(1j*omega)*(1-dR0dt/c)* np.exp(1j*omega*(t-R0/c))

            p_timederi = p_timederi.real

            pl_spatialderi = - rho_0*(U01*Ux + U02*Uy)  # Spatial-derivative
            pl_spatialderi = pl_spatialderi.real

            p  = p_timederi + pl_spatialderi

            px = p * n_hat_1[i,:]
            py = p * n_hat_2[i,:]
            pz = p * n_hat_3[i,:]

            # deisntiy (reference density + fluctuating density): # CONCEPTUALLY IMPORTANT
            rho = rho_0 + p/c**2
            
            # ----------------- PRESSURE & DENSITY -----------------
            # //////////////////// Numerical Monopole ////////////////////


            
        elif Solution_Option == 'DIPOLE':
            # //////////////////// Numerical Dipole ////////////////////
            # -----------------------------------
            # Define velocity potential: monopole
            # -----------------------------------

            # ----------------- VELOCITY -----------------
            # velocity: get spatial derivative by very small perturbation
            perturb = 1E-03

            #  d^2 fai/ (dx2 dx1) - streamwise
            Rs1_11 , R1_11, _, _, _ = Garrick_Triangle(M0, AoA, X1+perturb, Y1+perturb, Z1 )
            Rs1_22 , R1_22, _, _, _ = Garrick_Triangle(M0, AoA, X1+perturb, Y1-perturb, Z1 )
            Rs1_33 , R1_33, _, _, _ = Garrick_Triangle(M0, AoA, X1-perturb, Y1+perturb, Z1 )
            Rs1_44 , R1_44, _, _, _ = Garrick_Triangle(M0, AoA, X1-perturb, Y1-perturb, Z1 )

            fai1_11   = A/(4*pi*Rs1_11 )*np.exp( 1j*omega*( t-R1_11/c) )
            fai1_22   = A/(4*pi*Rs1_22 )*np.exp( 1j*omega*( t-R1_22/c) )
            fai1_33   = A/(4*pi*Rs1_33 )*np.exp( 1j*omega*( t-R1_33/c) )
            fai1_44   = A/(4*pi*Rs1_44 )*np.exp( 1j*omega*( t-R1_44/c) )

            Ux = (fai1_11 + fai1_44 - fai1_22 - fai1_33) / (4*perturb*perturb)

            #  d^2 fai/ (dx2 dx2) - wall-normal
            Rs2_plus , R2_plus  , _, _, _  = Garrick_Triangle(M0, AoA, X1, Y1+perturb, Z1 )
            Rs2_mid  , R2_mid   , _, _, _  = Garrick_Triangle(M0, AoA, X1, Y1        , Z1 )
            Rs2_minus, R2_minus , _, _, _  = Garrick_Triangle(M0, AoA, X1, Y1-perturb, Z1 )

            fai2_plus  = A/(4*pi*Rs2_plus ) *np.exp( 1j*omega*( t-R2_plus /c) )
            fai2_mid   = A/(4*pi*Rs2_mid  ) *np.exp( 1j*omega*( t-R2_mid  /c) )
            fai2_minus = A/(4*pi*Rs2_minus) *np.exp( 1j*omega*( t-R2_minus/c) )

            Uy = (fai2_plus - 2*fai2_mid + fai2_minus) / (perturb*perturb)

            #  d^2 fai/ (dx2 dx3) - spanwise
            Rs3_11 , R3_11, _, _, _ = Garrick_Triangle(M0, AoA, X1, Y1+perturb, Z1+perturb )
            Rs3_22 , R3_22, _, _, _ = Garrick_Triangle(M0, AoA, X1, Y1+perturb, Z1-perturb )
            Rs3_33 , R3_33, _, _, _ = Garrick_Triangle(M0, AoA, X1, Y1-perturb, Z1+perturb )
            Rs3_44 , R3_44, _, _, _ = Garrick_Triangle(M0, AoA, X1, Y1-perturb, Z1-perturb )

            fai1_11  = A/(4*pi*Rs3_11 )*np.exp( 1j*omega*( t-R3_11/c) )
            fai1_22  = A/(4*pi*Rs3_22 )*np.exp( 1j*omega*( t-R3_22/c) )
            fai1_33  = A/(4*pi*Rs3_33 )*np.exp( 1j*omega*( t-R3_33/c) )
            fai1_44  = A/(4*pi*Rs3_44 )*np.exp( 1j*omega*( t-R3_44/c) )

            Uz = (fai1_11 + fai1_44 - fai1_22 - fai1_33) / (4*perturb*perturb)

            # fluctuating velocity components
            Ux = Ux.real
            Uy = Uy.real
            Uz = Uz.real
            
            # ----------------- VELOCITY -----------------


            # ----------------- PRESSURE & DENSITY -----------------

            # fluctuating pressure & fluctuating density
            Rs_plus , R0_plus , _, _, _ = Garrick_Triangle(M0, AoA, X1, Y1+perturb, Z1 )
            Rs_minus, R0_minus, _, _, _ = Garrick_Triangle(M0, AoA, X1, Y1-perturb, Z1 )

            dfaidt_plus  = A/(4*pi*Rs_plus  )*np.exp( 1j*omega*( t-R0_plus/c) )
            dfaidt_minus = A/(4*pi*Rs_minus )*np.exp( 1j*omega*( t-R0_minus/c) )

            p_timederi = Time_Derivative_1D(  (dfaidt_plus-dfaidt_minus)/(2*perturb) ,dt  )
            # p_timederi = gradient(  (dfaidt_plus-dfaidt_minus)/(2*perturb) ,t  );
            p_timederi = -rho_0*p_timederi

            pl_spatialderi = - rho_0*(U01*Ux + U02*Uy)  # Spatial-derivative

            p_timederi     = p_timederi.real
            pl_spatialderi = pl_spatialderi.real
            
            p  = p_timederi + pl_spatialderi

            px = p * n_hat_1[i,:]
            py = p * n_hat_2[i,:]
            pz = p * n_hat_3[i,:]

            # deisntiy (reference density + fluctuating density): % CONCEPTUALLY IMPORTANT
            rho = rho_0 + p/c**2
            
            

            # ----------------- PRESSURE & DENSITY -----------------
            # //////////////////// Numerical Dipole ////////////////////

        Ux = Ux + U01  # AVAILABLE FROM CFD (U = U0 + u')
        Uy = Uy + U02
            
        ARRAY_Ux[i,:]             = Ux
        ARRAY_Uy[i,:]             = Uy
        ARRAY_Uz[i,:]             = Uz
        
        ARRAY_p_timederi[i,:]     = p_timederi
        ARRAY_pl_spatialderi[i,:] = pl_spatialderi
        ARRAY_p[i,:]              = p
        
        ARRAY_px[i,:]             = px
        ARRAY_py[i,:]             = py
        ARRAY_pz[i,:]             = pz  
        
        ARRAY_rho[i,:]            = rho
    
    
    #return d_vec_1,d_vec_2,d_vec_3, Ux,Uy,Uz,p_timederi,pl_spatialderi,p,px,py,pz,rho,N_FWHsurfi
    return  Ys_vec_1,Ys_vec_2,Ys_vec_3,\
            n_hat_1,n_hat_2,n_hat_3,dS,\
            ARRAY_Ux,ARRAY_Uy,ARRAY_Uz,\
            ARRAY_p_timederi,ARRAY_pl_spatialderi,ARRAY_p,\
            ARRAY_px,ARRAY_py,ARRAY_pz,ARRAY_rho,N_FWHsurfi