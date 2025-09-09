# %%
# ----------------------------
# CSDL_alpha and NumPy Modules
# ----------------------------
import csdl_alpha as csdl
import numpy as np

# ---------------
# Basis modules
# ---------------
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time   # Measureing Elapsed Time
import pickle # Data import format (.pickle)

# ----------------------------
# FW-H Main Solver (Module)
# ----------------------------

# # ---
# import sys, os
# CURRENT_DIR = os.path.dirname(__file__)                # examples/
# SRC_DIR     = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
# if SRC_DIR not in sys.path:
#     sys.path.insert(0, SRC_DIR)
# # ---
from pathlib import Path
EX_DIR = Path(__file__).parent

# from CSDL_FWH_vec import CSDL_FWH_def, FWHVariableGroup
from csdl_fwh import CSDL_FWH_def, FWHVariableGroup

# ----------
# Define pi
# ----------
pi = np.pi

# ----------------------
# Start Code
# ----------------------
start_time = time.time()

# ===========================================
# Three problems (Monopole/Dipole, VLM, KDH)
# ===========================================
Solution_Option = 'VLM'   # VLM

# -----------------------------------------
# User defined parameters (VLM INPUT)
# -----------------------------------------
if Solution_Option =='VLM':
    
    # file_path = './Main_Input_VLM.txt'
    # file_path = CURRENT_DIR+'/Main_Input_VLM.txt' 
    file_path = EX_DIR / "Main_Input_VLM.txt" 

    RAW = []
    with open(file_path,'r') as f:
        for i, line in enumerate(f):        
            if i < 7:
                # (1) Divide string and value by space and then (2) take only value
                RAW.append(line.split()[1]) 

    c         = float(RAW[0])  # Speed of sound [m/s]
    rho_0     = float(RAW[1])  # Amibient deisnty
    U0        = float(RAW[2])  # Freestream velocity [m/s]
    M0        = U0/c           # Mach Number
    AoA       = float(RAW[3])  # Angle of Attack [deg]
    R_obs     = float(RAW[4])  # Observer Radius [m]
    theta_obs = eval(RAW[5])   # Observer Azimuthal Angle Range [deg]
    theta_obs = theta_obs * pi/180 # deg -> rad

    Xo        = R_obs * np.cos(theta_obs)  # Observer Vector
    Yo        = R_obs * np.sin(theta_obs)
    Zo        = np.zeros(len(theta_obs))
    
    CFD_file_name = RAW[6]     # CFD file input
    
    # .................................................................
    # Import CFD data & define the FW-H source time and flow variables
    # .................................................................
    with open(CFD_file_name, 'rb') as f:  # 'rb': read binary
        FLW = pickle.load(f)
        
    # print(type(FLW))
    # >> dict

    # print(FLW.keys())
    # >> dict_keys(['mesh', 'panel_pressure', 'surface_vel'])

    data_mesh     = FLW['mesh']
    data_p        = FLW['panel_pressure']
    data_vel      = FLW['surface_vel']
    data_norm     = FLW['panel_normal'] # Updated at the second file
    data_coordi   = FLW['panel_center'] # Updated at the second file
    data_dS       = FLW['panel_area']   # Updated at the second file
    data_time     = FLW['time']         # Updated at the second file
    
    # Define "source" time array
    t  = data_time
    Nt = data_time.size
    dt = data_time[1]-data_time[0]
    



# Medium velocity and Mach number vectors
U0_vec = np.array([np.cos(AoA*pi/180), np.sin(AoA*pi/180), 0])*(M0*c)
M_vec  = -U0_vec/c  # FW-H variable

U01 = U0_vec[0]
U02 = U0_vec[1]


if (Solution_Option=='VLM'):
    # -------------------------------------------
    # Initialize the flow variables (Impermeable)
    # -------------------------------------------
    Nele,Nt,Nxi,Neta,_ = data_coordi.shape
    N_FWHsurfi         = Nxi*Neta

    Ys_vec_1             = np.zeros([N_FWHsurfi, Nt])
    Ys_vec_2             = np.zeros([N_FWHsurfi, Nt])
    Ys_vec_3             = np.zeros([N_FWHsurfi, Nt])

    n_hat_1              = np.zeros([N_FWHsurfi, Nt])
    n_hat_2              = np.zeros([N_FWHsurfi, Nt])
    n_hat_3              = np.zeros([N_FWHsurfi, Nt])

    # dS                   = np.zeros(N_FWHsurfi)
    dS                   = np.zeros([N_FWHsurfi,1])

    ARRAY_Ux             = np.ones([N_FWHsurfi, Nt]) *U01
    ARRAY_Uy             = np.ones([N_FWHsurfi, Nt]) *U02
    ARRAY_Uz             = np.zeros([N_FWHsurfi, Nt]) 

    ARRAY_p_timederi     = np.zeros([N_FWHsurfi, Nt])
    ARRAY_pl_spatialderi = np.zeros([N_FWHsurfi, Nt])
    ARRAY_p              = np.zeros([N_FWHsurfi, Nt])

    ARRAY_px             = np.zeros([N_FWHsurfi, Nt])
    ARRAY_py             = np.zeros([N_FWHsurfi, Nt])
    ARRAY_pz             = np.zeros([N_FWHsurfi, Nt])

    ARRAY_rho            = np.ones([N_FWHsurfi, Nt])*rho_0 # Incompressible flow

    # Using reshape, for loop can be avoided.
    # (i,j) -> (n,1)
    
    for tt in range(Nt):
        n=0
        for i in range(Nxi):
            for j in range(Neta):
                Ys_vec_1[n,tt] =  data_coordi[0,tt,i,j,0]
                Ys_vec_2[n,tt] =  data_coordi[0,tt,i,j,2]
                Ys_vec_3[n,tt] =  data_coordi[0,tt,i,j,1]
                
                n_hat_1[n,tt]  =  data_norm[0,tt,i,j,0]
                n_hat_2[n,tt]  =  data_norm[0,tt,i,j,2]
                n_hat_3[n,tt]  =  data_norm[0,tt,i,j,1]
                
                ARRAY_p[n,tt]  = data_p[0,tt,i,j]
                
                # ARRAY_Ux[n,tt] =  data_vel[0,tt,i,j,0]
                # ARRAY_Uy[n,tt] =  data_vel[0,tt,i,j,2]
                # ARRAY_Uz[n,tt] =  data_vel[0,tt,i,j,1]
                
                dS[n,0] = data_dS[0,0,i,j]                
                n=n+1


print(f"Data imported time: {time.time() - start_time:.4f} seconds")







# ///////////////////////////////////////////////
# CSDL START
# ///////////////////////////////////////////////
recorder = csdl.Recorder(inline=True)
# recorder = csdl.Recorder()
recorder.start()

# ====================================================================================
# FW-H computation using the numerical solution (CFD) : Ver.2 Vectorization (Adopted)
# ====================================================================================
FWH_vg = FWHVariableGroup(
c          = c, 
M0         = M0,
M_vec      = M_vec,
U01        = U01,
U02        = U02,
AoA        = AoA,
rho_0      = rho_0,
t          = t, 
dt         = dt,
Xobs       = Xo,       # Nobs
Yobs       = Yo,       # Nobs
Zobs       = Zo,       # Nobs
Ys_vec_1   = Ys_vec_1,
Ys_vec_2   = Ys_vec_2,
Ys_vec_3   = Ys_vec_3,
n_hat_1    = n_hat_1,
n_hat_2    = n_hat_2,
n_hat_3    = n_hat_3,
dS         = dS, 
N_FWHsurfi = N_FWHsurfi, 
Nt         = Nt,
ARRAY_p    = ARRAY_p,
ARRAY_rho  = ARRAY_rho,
ARRAY_Ux   = ARRAY_Ux,
ARRAY_Uy   = ARRAY_Uy,
ARRAY_Uz   = ARRAY_Uz,
)

tau_interp, p_FWH_T, p_FWH_L, p_FWH, CSDL_freq, CSDL_f13, CSDL_Sqq, CSDL_SPL13, CSDL_OASPL =\
CSDL_FWH_def(
    FWHVariableGroup = FWH_vg
)

p_FWH_RMS = csdl.sqrt(csdl.average(p_FWH*p_FWH, axes=(1,)))

print('p_FWH_RMS.shape=',p_FWH_RMS.shape)
print('OASPL.value=',CSDL_OASPL.value)


# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# =====< Summary: sequence of updating thickness/loading noise >=====

# 1. Calculate the FW-H equation (Farassat 1A) as P_T & P_L vs. (t + r_mag(i,:)/c)

# 2. First interpolate for the Constant (Time-averaged) distance time (Rotation
#    effect) as p_interp_T & p_interp_L (vs. t + R_source_const(i)/c)

# 3. Second interpolrate for the different source locations as p_interp_T & p_interp_L
#    (vs. tau_min+ (0:Nt-1)*dt_interp;)

# 4. Sum to yield p_FWH_T & p_FWH_L (vs. tau_interp)
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*



# ===========================================================================================
# Time domain (retarded time, acoustic pressure)
#                                               -> Frequency domain (frequency, SPL/OASPL)
# ===========================================================================================
elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")



# ////////////////////////////////////////////////////////////////////////////////////

# -------------------------
# Plot the FW-H results
# -------------------------
Nobs = len(Xo)


# VLM (Time)
Ref_tau_interp= np.array([0.0143143, 0.0639178, 0.1135213, 0.1631248, 0.2127283, 0.2623318, 0.3119353,
                        0.3615388, 0.4111423, 0.4607458, 0.5103493, 0.5599528, 0.6095563, 0.6591598,
                        0.7087633, 0.7583668, 0.8079703, 0.8575738, 0.9071773, 0.9567808, 1.0063843,
                        1.0559878, 1.1055913, 1.1551948, 1.2047983, 1.2544018, 1.3040053, 1.3536088,
                        1.4032123])

Ref_p_FWH= np.array([-0.32864071, -0.34975044, -0.42310286, -0.44751658, -0.49755523, -0.5243716,
                    -0.5592972,  -0.58289028, -0.60808809, -0.62733133, -0.64596851, -0.66117631,
                    -0.67520553, -0.68707279, -0.69776458, -0.70699194, -0.71521774, -0.72239978,
                    -0.72877658, -0.73438375, -0.7393601,  -0.74376009, -0.74766994, -0.75114168,
                    -0.75423536, -0.75699393, -0.75945934, -0.76166539, -0.76364109])

#%%
# -----------------------------------------------------
# Plot: Time-domain Pressure 
# -----------------------------------------------------
fig, ax = plt.subplots(figsize=(8,3), dpi=100)
ax.plot(tau_interp.value, p_FWH[0,:].value , linestyle='-', color='red', linewidth=2, label = 'FW-H (CSDL)')
ax.plot(Ref_tau_interp[:len(tau_interp.value)+10], Ref_p_FWH[:len(tau_interp.value)+10], '--', color='blue', linewidth=2, label = 'FW-H (SciPy)')

plt.xlabel(r"Observer time, $t^{'}$ (sec)",fontsize=14)
plt.ylabel(r"Acoustic pressure, $p^{'}$ (Pa)",fontsize=14)

plt.gca().tick_params(labelsize=14)
plt.legend(fontsize=14, bbox_to_anchor=(1, 1))
plt.grid()
plt.savefig('Result_CSDL_Time.png',dpi=300,bbox_inches='tight')
plt.savefig('Result_CSDL_Time.pdf',dpi=300,bbox_inches='tight')
plt.show()