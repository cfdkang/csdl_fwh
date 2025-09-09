# %%

import csdl_alpha as csdl
import numpy as np

from dataclasses import dataclass
from csdl_alpha.utils.typing import VariableLike, Variable
from typing import Union, Optional

import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .core.CSDL_Garrick_Triangle import CSDL_Garrick_Triangle   # Garrick Triangle: coordinate transform for convection flow
from .core.CSDL_Time_Derivative import CSDL_Time_Derivative     # Time derivative
from .core.CSDL_FFT_Module import CSDL_SPL_def                  # SPL computation
from .core.CSDL_Interp_vF2 import CSDL_Interp2                  # Linear Interpolation

# Start Solver
start_time = time.time()

# Define pi
pi = np.pi

@dataclass
class FWHVariableGroup(csdl.VariableGroup):
    c          : VariableLike
    M0         : VariableLike
    M_vec      : VariableLike
    U01        : VariableLike
    U02        : VariableLike
    AoA        : VariableLike
    rho_0      : VariableLike
    t          : VariableLike
    dt         : VariableLike
    Xobs       : VariableLike
    Yobs       : VariableLike
    Zobs       : VariableLike
    Ys_vec_1   : VariableLike
    Ys_vec_2   : VariableLike
    Ys_vec_3   : VariableLike
    n_hat_1    : VariableLike
    n_hat_2    : VariableLike
    n_hat_3    : VariableLike
    dS         : VariableLike
    N_FWHsurfi : VariableLike
    Nt         : VariableLike
    ARRAY_p    : VariableLike
    ARRAY_rho  : VariableLike
    ARRAY_Ux   : Optional[VariableLike] = None
    ARRAY_Uy   : Optional[VariableLike] = None
    ARRAY_Uz   : Optional[VariableLike] = None

def CSDL_FWH_def(FWHVariableGroup):
   
    c          = FWHVariableGroup.c 
    M0         = FWHVariableGroup.M0
    M_vec      = FWHVariableGroup.M_vec
    U01        = FWHVariableGroup.U01
    U02        = FWHVariableGroup.U02
    AoA        = FWHVariableGroup.AoA
    rho_0      = FWHVariableGroup.rho_0
    t          = FWHVariableGroup.t 
    dt         = FWHVariableGroup.dt
    Xobs       = FWHVariableGroup.Xobs
    Yobs       = FWHVariableGroup.Yobs
    Zobs       = FWHVariableGroup.Zobs
    Ys_vec_1   = FWHVariableGroup.Ys_vec_1
    Ys_vec_2   = FWHVariableGroup.Ys_vec_2
    Ys_vec_3   = FWHVariableGroup.Ys_vec_3
    n_hat_1    = FWHVariableGroup.n_hat_1
    n_hat_2    = FWHVariableGroup.n_hat_2
    n_hat_3    = FWHVariableGroup.n_hat_3 
    dS         = FWHVariableGroup.dS 
    N_FWHsurfi = FWHVariableGroup.N_FWHsurfi 
    Nt         = FWHVariableGroup.Nt
    ARRAY_p    = FWHVariableGroup.ARRAY_p
    ARRAY_rho  = FWHVariableGroup.ARRAY_rho
    ARRAY_Ux   = FWHVariableGroup.ARRAY_Ux
    ARRAY_Uy   = FWHVariableGroup.ARRAY_Uy
    ARRAY_Uz   = FWHVariableGroup.ARRAY_Uz
    
    
    # .......................................
    # CSDL Expand into (Nobs, NFWHSurfi, Nt)
    # .......................................
    Nobs = len(Xobs)
    target_shape = (Nobs, N_FWHsurfi, Nt)

    Exp_t    = csdl.expand(np.array([t for _ in range(N_FWHsurfi)]), target_shape,'ij->kij' ) # (NFWHSurfi, Nt) to (Nobs, NFWHSurfi, Nt)
    print('Exp_t.shape=',Exp_t.shape)
    
    if Nobs ==1:
        # NOTE: DO NOT ADD , 'i->ijk' for the scalar expansion
        Exp_Xobs = csdl.expand(Xobs, target_shape) # (Nobs) to (Nobs, NFWHSurfi, Nt)
        Exp_Yobs = csdl.expand(Yobs, target_shape)
        Exp_Zobs = csdl.expand(Zobs, target_shape)
    else:
        Exp_Xobs = csdl.expand(Xobs, target_shape,'i->ijk') # (Nobs) to (Nobs, NFWHSurfi, Nt)
        Exp_Yobs = csdl.expand(Yobs, target_shape,'i->ijk')
        Exp_Zobs = csdl.expand(Zobs, target_shape,'i->ijk')

    Exp_Ys_vec_1 = csdl.expand(Ys_vec_1, target_shape, 'jk->ijk') # (NFWHSurfi, Nt) to (Nobs, NFWHSurfi, Nt)
    Exp_Ys_vec_2 = csdl.expand(Ys_vec_2, target_shape, 'jk->ijk')
    Exp_Ys_vec_3 = csdl.expand(Ys_vec_3, target_shape, 'jk->ijk')
    
    Exp_n_hat_1  = csdl.expand(n_hat_1, target_shape, 'jk->ijk')  # (NFWHSurfi, Nt) to (Nobs, NFWHSurfi, Nt)
    Exp_n_hat_2  = csdl.expand(n_hat_2, target_shape, 'jk->ijk')
    Exp_n_hat_3  = csdl.expand(n_hat_3, target_shape, 'jk->ijk')
    
    Exp_dS        = csdl.expand(dS*np.ones((N_FWHsurfi,Nt)), target_shape, 'jk->ijk')
    
    Exp_ARRAY_p   = csdl.expand(ARRAY_p, target_shape, 'jk->ijk')
    Exp_ARRAY_rho = csdl.expand(ARRAY_rho, target_shape,'jk->ijk')
    Exp_ARRAY_Ux  = csdl.expand(ARRAY_Ux, target_shape, 'jk->ijk')
    Exp_ARRAY_Uy  = csdl.expand(ARRAY_Uy, target_shape, 'jk->ijk')
    Exp_ARRAY_Uz  = csdl.expand(ARRAY_Uz, target_shape, 'jk->ijk')
    
    
    # ===================================================
    # FW-H computation using the numerical solution (CFD)
    # ===================================================

    # //////////////////// Vector Calculator Module ////////////////////
    # ----------------------------------------------------------
    # Geometric distance (ADD THE MOVING TERM (BLADE KINEMATICS)
    # ----------------------------------------------------------
    d_vec_1 = Exp_Xobs - Exp_Ys_vec_1
    d_vec_2 = Exp_Yobs - Exp_Ys_vec_2
    d_vec_3 = Exp_Zobs - Exp_Ys_vec_3

    # ----------------------------------------------------------------
    # Acoustic distance using Garrick Triangle: UNIT RADIATION VECTOR
    # ----------------------------------------------------------------    
    _, r_mag, r_vec_1, r_vec_2, r_vec_3 = CSDL_Garrick_Triangle(M0, AoA, d_vec_1, d_vec_2, d_vec_3)
    r_hat_1  = r_vec_1 / r_mag #(R) % N_FWHsurfi by Nt
    r_hat_2  = r_vec_2 / r_mag 
    r_hat_3  = r_vec_3 / r_mag 
    # CHECK: magnitude of unity: plt.plot(r_hat_1**2 + r_hat_2**2 + r_hat_3**2)
    # //////////////////// Vector Calculator Module ////////////////////

        
    # ============================
    # FW-H Implementation
    # ============================
    p_interp_T = csdl.Variable(value = np.zeros([Nobs*N_FWHsurfi, Nt]) ) # 1ST AND 2END INTERPOLATED THICKNESS NOISE
    p_interp_L = csdl.Variable(value = np.zeros([Nobs*N_FWHsurfi, Nt]) ) # 1ST AND 2END INTERPOLATED LOADING NOISE
    
    p_FWH_T = csdl.Variable(value = np.zeros([Nobs, Nt]))
    p_FWH_L = csdl.Variable(value = np.zeros([Nobs, Nt]))
    p_FWH   = csdl.Variable(value = np.zeros([Nobs, Nt]))    

    Ux  = Exp_ARRAY_Ux     # N_FWHsurfi by Nt
    Uy  = Exp_ARRAY_Uy
    Uz  = Exp_ARRAY_Uz
    p   = Exp_ARRAY_p
    rho = Exp_ARRAY_rho
    
    n_hat_1 = Exp_n_hat_1
    n_hat_2 = Exp_n_hat_2
    n_hat_3 = Exp_n_hat_3
    
    dS  = Exp_dS
    
    px  = p * n_hat_1
    py  = p * n_hat_2
    pz  = p * n_hat_3

    # Debug:KDH
    # print('p_interp_P.shape=',p_interp_T.shape)
    # print('p_interp_L.shape=',p_interp_L.shape)    
    # print('Ux.shape',Ux.shape)
    # print('rho.shape',rho.shape)
    # print('n_hat_1.shape',n_hat_1.shape)
    # print('dS.shape',dS.shape)
    # print('px.shape',px.shape)

          
    # //////////////////// FW-H Main Module ////////////////////
    # =================================================
    # F1A Quantity : Calculate the FW-H stress tensors
    # =================================================
    
    dot_vn = Ux * n_hat_1 + \
             Uy * n_hat_2 + \
             Uz * n_hat_3
    
    # dot_vn_array[i,:] = dot_vn # DEBUG
    
    # Determine Qn Lr Lm Mr
    Qn = -rho_0 * ( U01*n_hat_1 + U02*n_hat_2 ) + rho*dot_vn
    
    Lr = ( px  +  rho *(Ux-U01) * dot_vn ) * r_hat_1 + \
         ( py  +  rho *(Uy-U02) * dot_vn ) * r_hat_2 + \
         ( pz  +  rho *(Uz    ) * dot_vn ) * r_hat_3

    Lm = ( px  +  rho *(Ux-U01) * dot_vn ) * M_vec[0] + \
         ( py  +  rho *(Uy-U02) * dot_vn ) * M_vec[1] + \
         ( pz  +  rho *(Uz    ) * dot_vn ) * M_vec[2] 

    Mr = M_vec[0] * r_hat_1 + \
         M_vec[1] * r_hat_2 + \
         M_vec[2] * r_hat_3 
         
    # Determine Q_dot_n & L_dot_r
    Qndot = CSDL_Time_Derivative( Qn , dt )
    Lrdot = CSDL_Time_Derivative( Lr , dt )
    
    # # Debug:KDH
    # print('Qn.shape=',Qn.shape)
    # print('Qndot.shape=',Qndot.shape)
    # print('Lrdot.shape=',Lrdot.shape)
    # print('dS.shape=',dS.shape)   
    
    
    # Thickness Noise & Loading Noise at Retarted time, 
    # although it was calcualted at source time (physically interpreted this way)
    P_T = 1/(4*pi)   * dS * Qndot             / r_mag    / (1-Mr)**2 + \
          1/(4*pi)   * dS * Qn*c*(Mr - M0**2) / r_mag**2 / (1-Mr)**3 

    P_L = 1/(4*pi*c) * dS * Lrdot             / r_mag    / (1-Mr)**2 + \
          1/(4*pi  ) * dS * (Lr-Lm)           / r_mag**2 / (1-Mr)**2 + \
          1/(4*pi  ) * dS * Lr*(Mr-M0**2)     / r_mag**2 / (1-Mr)**3

    # Debug:KDH
    # print('P_T.shape=',P_T.shape)
    # print('P_L.shape=',P_L.shape)
    
    # Find retarded time
    tau = Exp_t + r_mag/c    
    # print('tau.shape=',tau.shape)

    # THIS IS NEEDED
    R_source_const       = csdl.average(r_mag,axes=(2,)) # % Constant (Time-averaged) distance time, axis=2 row-wise # CSDL 3-D array
    R_source_const       = csdl.expand(R_source_const, target_shape, 'jk->jki')
    tau_const            = Exp_t + R_source_const/c # (Nobs, N_FWHsurfi, Nt)+ (Nobs, N_FWHsurfi, 1)  = (Nobs, N_FWHsurfi, Nt)
    # print('tau_const.shape=',tau_const.shape)
    # print('R_source_const.shape=',R_source_const.shape)
    

    # First interpolate for the Constant (Time-averaged) distance time
    # This is done due to the rotating effect, dt=\constant.

    # Using reshape [(Nobs,N_FWHsurfi,Nt) --> (Nobs*N_FWHsurfi,Nt)]
    p_interp_T = CSDL_Interp2(tau.reshape(Nobs*N_FWHsurfi,Nt), \
                              P_T.reshape(Nobs*N_FWHsurfi,Nt), \
                              tau_const.reshape(Nobs*N_FWHsurfi,Nt)) # >> INTERP
    p_interp_L = CSDL_Interp2(tau.reshape(Nobs*N_FWHsurfi,Nt), \
                              P_L.reshape(Nobs*N_FWHsurfi,Nt), \
                              tau_const.reshape(Nobs*N_FWHsurfi,Nt)) # >> INTERP
    print('1st_Iterpolation_complated')
    

    tau_min    = csdl.maximum(R_source_const, axes=(1,))/c
    tau_max    = csdl.minimum(R_source_const, axes=(1,))/c + dt*(Nt-1) # (Nobs,1)
    dt_interp  = ( tau_max - tau_min )/(Nt-1)
    tau_interp = tau_min+ np.array([np.arange (Nt) for _ in range(Nobs)]) * dt_interp # (0:Nt-1)
    
    
    tau_interp = csdl.expand(tau_interp[0,:], (N_FWHsurfi*Nobs,Nt),'i->ji') # (Nt) -> (N_FWHsurfi*Nobs, Nt)

    
    # Using reshape
    p_interp_T = CSDL_Interp2(tau_const.reshape(Nobs*N_FWHsurfi,Nt), p_interp_T, tau_interp) # >> INTERP
    p_interp_L = CSDL_Interp2(tau_const.reshape(Nobs*N_FWHsurfi,Nt), p_interp_L, tau_interp) # >> INTERP
    print('2nd_Iterpolation_complated')
    
    
    # ////////// Observer Time & 2nd Interpolation Module //////////
    # For loop for the obs array
    p_interp_T = p_interp_T.reshape(Nobs, N_FWHsurfi, Nt)
    p_interp_L = p_interp_L.reshape(Nobs, N_FWHsurfi, Nt)
    
    p_FWH_T = csdl.sum(p_interp_T, axes=(1,)) # Integrated w.r.t. Column (area)  0: [:,-,-] , 1: [-,:,-] , 2: [:,-,:]
    p_FWH_L = csdl.sum(p_interp_L, axes=(1,)) # Integrated w.r.t. Column (area)
    p_FWH   = p_FWH_T+p_FWH_L
    # ////////// Observer Time & 2nd Interpolation Module //////////   
    
    
    # ====================
    # CALCULATE SPL/OASPL
    # ====================
    
    # SPL at single microphone
    CSDL_freq, CSDL_f13, CSDL_Sqq, CSDL_SPL13, CSDL_OASPL = CSDL_SPL_def(tau_interp[0,:], p_FWH[0,:], p_FWH[0,:].shape[0])
    
    # SPL at multiple microphone
    CSDL_Sqq   = csdl.Variable(value=0, shape=(Nobs, CSDL_freq.shape[0]))
    CSDL_SPL13 = csdl.Variable(value=0, shape=(Nobs, CSDL_f13.shape[0]))
    CSDL_OASPL = csdl.Variable(value=0, shape=(Nobs,))
    
    for i in csdl.frange(Nobs):
        _, _, Sqq, SPL13, OASPL = CSDL_SPL_def(tau_interp[0,:], p_FWH[i,:], p_FWH[0,:].shape[0])
        CSDL_Sqq   = CSDL_Sqq.set(csdl.slice[i,:]  , Sqq  )    
        CSDL_SPL13 = CSDL_SPL13.set(csdl.slice[i,:], SPL13 )   
        CSDL_OASPL = CSDL_OASPL.set(csdl.slice[i]  , OASPL )   
    
    # Debug:KDH
    # print('CSDL_freq=',CSDL_freq.value)
    # print('CSDL_Sqq=',CSDL_Sqq.value)
    # print('CSDL_f13=',CSDL_f13.value)
    # print('CSDL_SPL13=',CSDL_SPL13.value)
    # print('CSDL_OASPL=',CSDL_OASPL.value)
    
    return tau_interp[0,:], p_FWH_T, p_FWH_L, p_FWH,   CSDL_freq, CSDL_f13, CSDL_Sqq, CSDL_SPL13, CSDL_OASPL
    

    # ...............
    # OUTPUT SHAPE
    # ...............
    # 1) tau_interp[0,:]  = (Nobs, Nt)
    # 2) p_FWH_T          = (Nobs, Nt)
    # 3) p_FWH_T          = (Nobs, Nt)
    # 4) p_FWH_T          = (Nobs, Nt)
    
    # 5) CSDL_freq        = (Nfreq)
    # 6) CSDL_f13         = (Nf13)
    
    # 7) CSDL_Sqq         = (Nobs, Nfreq)
    # 8) CSDL_SPL13       = (Nobs, Nf13)
    # 9) CSDL_OASPL       = (Nobs)    
