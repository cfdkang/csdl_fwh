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
# from CSDL_FWH_vec import CSDL_FWH_def, FWHVariableGroup
from csdl_fwh import CSDL_FWH_def, FWHVariableGroup

# ------------------------------------------------------------------------
# Note this module is not CSDL. Just used for the comparisoin to CSDL-FFT
# ------------------------------------------------------------------------
# from FFT import FFT_def 
from csdl_fwh.core.FFT import FFT_def 

# --------------------------------------------------------------------
# Reference solutions (Analytic and SciPy-based numerical solutions)
# --------------------------------------------------------------------
from aero_data.Monopole_Dipole.Generator_1_Analytic_Solution import Generator_1_Analytic_Solution_def   # Analytic solution
from aero_data.Monopole_Dipole.Generator_2_Numerical_Solution import Generator_2_Numerical_Solution_def # Numerical solution

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
Solution_Option = 'MONOPOLE' # MONOPOLE or DIPOLE
# Solution_Option = 'DIPOLE'   # MONOPOLE or DIPOLE

# -----------------------------------------
# User defined parameters (MONOPOLE/DIPOLE)
# -----------------------------------------
if Solution_Option == 'MONOPOLE' or Solution_Option == 'DIPOLE':
    
    file_path = './Main_Input_monopole.txt'
    RAW = []
    with open(file_path,'r') as f:
        for i, line in enumerate(f):
            if i < 6:
                # (1) Divide string and value by space and then (2) take only value
                RAW.append(line.split()[1]) 

    c         = float(RAW[0])  # Speed of sound [m/s]
    rho_0     = float(RAW[1])  # Amibient deisnty
    M0        = float(RAW[2])  # Mach Number
    U0        = M0*c           # Freestream velocity [m/s]
    AoA       = float(RAW[3])  # Angle of Attack [deg]
    R_obs     = float(RAW[4])  # Observer Radius [m]
    theta_obs = eval(RAW[5])   # Observer Azimuthal Angle Range [deg]
    
    theta_obs = theta_obs * pi/180

    Xo        = R_obs * np.cos(theta_obs)  # Observer Vector
    Yo        = R_obs * np.sin(theta_obs)
    Zo        = np.zeros(len(theta_obs))

    # Define "source" time array
    
    # Moderate samples 
    # # Single Time domain
    # # Nt     = 100 
    # Nt     = 200 
    # t      = np.linspace(1/Nt,1,Nt)
    
    # Short samples (quick test or directivity)
    Nt     = 50
    t      = np.linspace(0.3/Nt,0.3,Nt)

    dt     = t[1]-t[0]   
    

# Medium velocity and Mach number vectors
U0_vec = np.array([np.cos(AoA*pi/180), np.sin(AoA*pi/180), 0])*(M0*c)
M_vec  = -U0_vec/c  # FW-H variable

U01 = U0_vec[0]
U02 = U0_vec[1]


if (Solution_Option=='MONOPOLE' or Solution_Option=='DIPOLE'):
    # ------------------------------------
    # MONOPOLE/DIPOLE analytical solution 
    # ------------------------------------
    p_anal_1, p_anal_2, p_anal, p_anal_RMS = \
    Generator_1_Analytic_Solution_def(Xo, Yo, Zo,\
    rho_0, U01, U02, M0, AoA, t, dt, c, Solution_Option)

    # -----------------------------------------------
    # MONOPOLE/DIPOLE numerical Solution (Permeable)
    # -----------------------------------------------
    Ys_vec_1, Ys_vec_2, Ys_vec_3,\
    n_hat_1, n_hat_2, n_hat_3, dS,\
    ARRAY_Ux,ARRAY_Uy,ARRAY_Uz,\
    ARRAY_p_timederi,ARRAY_pl_spatialderi,ARRAY_p,\
    ARRAY_px,ARRAY_py,ARRAY_pz,ARRAY_rho,N_FWHsurfi=\
    Generator_2_Numerical_Solution_def(rho_0, U01, U02, M0, AoA, t, dt, c, Solution_Option)

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

if Solution_Option=='MONOPOLE':
    # Monopole (Time) 200 (See Line.80)
    t_anal= np.array([0.005, 0.01,  0.015, 0.02, 0.025, 0.03,  0.035,  0.04,  0.045,  0.05,  0.055,  0.06,
            0.065, 0.07,  0.075, 0.08, 0.085, 0.09,  0.095,  0.1 ,  0.105,  0.11,  0.115,  0.12,
            0.125, 0.13,  0.135, 0.14, 0.145, 0.15,  0.155,  0.16,  0.165,  0.17,  0.175,  0.18,
            0.185, 0.19,  0.195, 0.2 , 0.205, 0.21,  0.215,  0.22,  0.225,  0.23,  0.235,  0.24,
            0.245, 0.25,  0.255, 0.26, 0.265, 0.27,  0.275,  0.28,  0.285,  0.29,  0.295,  0.3 ,
            0.305, 0.31,  0.315, 0.32, 0.325, 0.33,  0.335,  0.34,  0.345,  0.35,  0.355,  0.36,
            0.365, 0.37,  0.375, 0.38, 0.385, 0.39,  0.395,  0.4 ,  0.405,  0.41,  0.415,  0.42,
            0.425, 0.43,  0.435, 0.44, 0.445, 0.45,  0.455,  0.46,  0.465,  0.47,  0.475,  0.48,
            0.485, 0.49,  0.495, 0.5 , 0.505, 0.51,  0.515,  0.52,  0.525,  0.53,  0.535,  0.54,
            0.545, 0.55,  0.555, 0.56, 0.565, 0.57,  0.575,  0.58,  0.585,  0.59,  0.595,  0.6 ,
            0.605, 0.61,  0.615, 0.62, 0.625, 0.63,  0.635,  0.64,  0.645,  0.65,  0.655,  0.66,
            0.665, 0.67,  0.675, 0.68, 0.685, 0.69,  0.695,  0.7 ,  0.705,  0.71,  0.715,  0.72,
            0.725, 0.73,  0.735, 0.74, 0.745, 0.75,  0.755,  0.76,  0.765,  0.77,  0.775,  0.78,
            0.785, 0.79,  0.795, 0.8 , 0.805, 0.81,  0.815,  0.82,  0.825,  0.83,  0.835,  0.84,
            0.845, 0.85,  0.855, 0.86, 0.865, 0.87,  0.875,  0.88,  0.885,  0.89,  0.895,  0.9 ,
            0.905, 0.91,  0.915, 0.92, 0.925, 0.93,  0.935,  0.94,  0.945,  0.95,  0.955,  0.96,
            0.965, 0.97,  0.975, 0.98, 0.985, 0.99,  0.995,  1.   ])
        
    p_anal= np.array([ 8.03629865,  8.02706246,  7.81017759,  7.39464731,  6.79574   ,  6.03425406,
            5.13559208,  4.12868423,  3.04482753,  1.91649332,  0.77616326, -0.34475573,
            -1.4169004 , -2.41377064, -3.31237958, -4.09373138, -4.7431182 , -5.25025317,
            -5.60924918, -5.818465  , -5.88023818, -5.80052733, -5.58848949, -5.25600893,
            -4.81719836, -4.28789218, -3.68514098, -3.02672401, -2.33068723, -1.61491495,
            -0.89673859, -0.19259184,  0.48229268,  1.11413747,  1.69082395,  2.2020567 ,
            2.63947488,  2.99672137,  3.26946976,  3.45541101,  3.55420155,  3.56737774,
            3.49823894,  3.35170264,  3.13413539,  2.85316405,  2.51746993,  2.13657223,
            1.72060269,  1.28007747,  0.82566845,  0.36797751, -0.08267453, -0.51645125,
            -0.92419355, -1.2975833 , -1.62929173, -1.91310662, -2.14402835, -2.31834229,
            -2.43366528, -2.48896113, -2.4845303 , -2.42197264, -2.30412453, -2.13497045,
            -1.91953642, -1.6637591 , -1.37434507, -1.05860796, -0.72430521, -0.37946119,
            -0.03219011,  0.30947661,  0.63776962,  0.94533748,  1.22540641,  1.47191492,
            1.67963975,  1.84429698,  1.96262511,  2.03244507,  2.05269667,  2.0234506 ,
            1.94589729,  1.82230968,  1.65598747,  1.45117423,  1.21295928,  0.94715991,
            0.66018993,  0.35891191,  0.05048828, -0.25778318, -0.55862876, -0.84496289,
            -1.11004333, -1.34762338, -1.55208978, -1.71858993, -1.84314047, -1.92271896,
            -1.95533391, -1.94007532, -1.87713498, -1.7678134 , -1.61449228, -1.4205926 ,
            -1.19049858, -0.92947282, -0.6435405 , -0.33936056, -0.02408635,  0.29479836,
            0.60963984,  0.91278866,  1.19677169,  1.45446935,  1.67927319,  1.86525504,
            2.00730636,  2.1012712 ,  2.14405726,  2.13372693,  2.06956164,  1.95210763,
            1.78318291,  1.56587095,  1.30446882,  1.00442324,  0.67223161,  0.31531777,
            -0.05812034, -0.43927136, -0.81890469, -1.1875738 , -1.53582011, -1.85440556,
            -2.13452617, -2.36803373, -2.54764898, -2.66716078, -2.72160885, -2.70744424,
            -2.62266575, -2.46692463, -2.24159566, -1.9498104 , -1.59645979, -1.18814493,
            -0.73308838, -0.24101777,  0.27701168,  0.80882171,  1.34134712,  1.86092291,
            2.35357741,  2.80537086,  3.20273083,  3.53281157,  3.78385332,  3.94553239,
            4.00930157,  3.96870278,  3.81965505,  3.56069382,  3.19316327,  2.72134709,
            2.15253591,  1.49701826,  0.76798943, -0.01862568, -0.84443372, -1.68892801,
            -2.52992137, -3.34408444, -4.1075349 , -4.79649899, -5.38803296, -5.86076225,
            -6.19564388, -6.37670415, -6.39173984, -6.23294231, -5.89742132, -5.38759518,
            -4.71140907, -3.88237971, -2.91941644, -1.84643945, -0.69175454,  0.51273706,
            1.73258479,  2.93191685,  4.07473211,  5.12622704,  6.05409084,  6.8297729 ,
            7.42957624,  7.83556545])

    Ref_tau_interp= np.array([0.00547527, 0.01046215, 0.01544903, 0.02043591, 0.02542279, 0.03040967,
                0.03539655, 0.04038343, 0.0453703 , 0.05035718, 0.05534406, 0.06033094,
                0.06531782, 0.0703047 , 0.07529158, 0.08027846, 0.08526534, 0.09025222,
                0.09523909, 0.10022597, 0.10521285, 0.11019973, 0.11518661, 0.12017349,
                0.12516037, 0.13014725, 0.13513413, 0.140121  , 0.14510788, 0.15009476,
                0.15508164, 0.16006852, 0.1650554 , 0.17004228, 0.17502916, 0.18001604,
                0.18500292, 0.18998979, 0.19497667, 0.19996355, 0.20495043, 0.20993731,
                0.21492419, 0.21991107, 0.22489795, 0.22988483, 0.2348717 , 0.23985858,
                0.24484546, 0.24983234, 0.25481922, 0.2598061 , 0.26479298, 0.26977986,
                0.27476674, 0.27975362, 0.28474049, 0.28972737, 0.29471425, 0.29970113,
                0.30468801, 0.30967489, 0.31466177, 0.31964865, 0.32463553, 0.3296224 ,
                0.33460928, 0.33959616, 0.34458304, 0.34956992, 0.3545568 , 0.35954368,
                0.36453056, 0.36951744, 0.37450432, 0.37949119, 0.38447807, 0.38946495,
                0.39445183, 0.39943871, 0.40442559, 0.40941247, 0.41439935, 0.41938623,
                0.4243731 , 0.42935998, 0.43434686, 0.43933374, 0.44432062, 0.4493075 ,
                0.45429438, 0.45928126, 0.46426814, 0.46925502, 0.47424189, 0.47922877,
                0.48421565, 0.48920253, 0.49418941, 0.49917629, 0.50416317, 0.50915005,
                0.51413693, 0.5191238 , 0.52411068, 0.52909756, 0.53408444, 0.53907132,
                0.5440582 , 0.54904508, 0.55403196, 0.55901884, 0.56400572, 0.56899259,
                0.57397947, 0.57896635, 0.58395323, 0.58894011, 0.59392699, 0.59891387,
                0.60390075, 0.60888763, 0.6138745 , 0.61886138, 0.62384826, 0.62883514,
                0.63382202, 0.6388089 , 0.64379578, 0.64878266, 0.65376954, 0.65875642,
                0.66374329, 0.66873017, 0.67371705, 0.67870393, 0.68369081, 0.68867769,
                0.69366457, 0.69865145, 0.70363833, 0.7086252 , 0.71361208, 0.71859896,
                0.72358584, 0.72857272, 0.7335596 , 0.73854648, 0.74353336, 0.74852024,
                0.75350712, 0.75849399, 0.76348087, 0.76846775, 0.77345463, 0.77844151,
                0.78342839, 0.78841527, 0.79340215, 0.79838903, 0.8033759 , 0.80836278,
                0.81334966, 0.81833654, 0.82332342, 0.8283103 , 0.83329718, 0.83828406,
                0.84327094, 0.84825782, 0.85324469, 0.85823157, 0.86321845, 0.86820533,
                0.87319221, 0.87817909, 0.88316597, 0.88815285, 0.89313973, 0.8981266 ,
                0.90311348, 0.90810036, 0.91308724, 0.91807412, 0.923061  , 0.92804788,
                0.93303476, 0.93802164, 0.94300852, 0.94799539, 0.95298227, 0.95796915,
                0.96295603, 0.96794291, 0.97292979, 0.97791667, 0.98290355, 0.98789043,
                0.9928773 , 0.99786418])
                
    Ref_p_FWH= np.array([ 8.18386966e+00,  8.01176068e+00,  7.78121104e+00,  7.35699576e+00,
            6.75206991e+00,  5.98724168e+00,  5.08779002e+00,  4.08240485e+00,
            3.00204555e+00,  1.87877441e+00,  7.44619851e-01, -3.69482508e-01,
            -1.43462952e+00, -2.42474599e+00, -3.31721753e+00, -4.09335081e+00,
            -4.73866224e+00, -5.24300589e+00, -5.60055154e+00, -5.80963630e+00,
            -5.87250458e+00, -5.79496433e+00, -5.58597373e+00, -5.25718443e+00,
            -4.82245940e+00, -4.29737571e+00, -3.69873092e+00, -3.04406392e+00,
            -2.35121760e+00, -1.63791356e+00, -9.21359141e-01, -2.17905941e-01,
            4.57254402e-01,  1.09034635e+00,  1.66921480e+00,  2.18349415e+00,
            2.62472442e+00,  2.98643071e+00,  3.26414327e+00,  3.45539783e+00,
            3.55968809e+00,  3.57839882e+00,  3.51466746e+00,  3.37325917e+00,
            3.16039217e+00,  2.88358338e+00,  2.55140094e+00,  2.17327476e+00,
            1.75926576e+00,  1.31984486e+00,  8.65653532e-01,  4.07290272e-01,
            -4.49096032e-02, -4.81080468e-01, -8.92010060e-01, -1.26931269e+00,
            -1.60557621e+00, -1.89449519e+00, -2.13095951e+00, -2.31114258e+00,
            -2.43253804e+00, -2.49399218e+00, -2.49568860e+00, -2.43909542e+00,
            -2.32691841e+00, -2.16301759e+00, -1.95230002e+00, -1.70059411e+00,
            -1.41450802e+00, -1.10127528e+00, -7.68587392e-01, -4.24422321e-01,
            -7.68683688e-02,  2.66051194e-01,  5.96549163e-01,  9.07239413e-01,
            1.19129109e+00,  1.44256814e+00,  1.65575293e+00,  1.82645453e+00,
            1.95129051e+00,  2.02794944e+00,  2.05523039e+00,  2.03305734e+00,
            1.96247160e+00,  1.84559838e+00,  1.68558976e+00,  1.48654997e+00,
            1.25343597e+00,  9.91944877e-01,  7.08382567e-01,  4.09521771e-01,
            1.02450895e-01, -2.05585024e-01, -5.07344501e-01, -7.95750255e-01,
            -1.06404598e+00, -1.30594543e+00, -1.51577164e+00, -1.68858329e+00,
            -1.82028467e+00, -1.90771969e+00, -1.94874244e+00, -1.94226899e+00,
            -1.88830584e+00, -1.78795471e+00, -1.64339250e+00, -1.45783013e+00,
            -1.23544701e+00, -9.81305311e-01, -7.01242281e-01, -4.01747391e-01,
            -8.98216084e-02,  2.27174600e-01,  5.41684041e-01,  8.46120471e-01,
            1.13304000e+00,  1.39531085e+00,  1.62627976e+00,  1.81993148e+00,
            1.97103282e+00,  2.07526629e+00,  2.12934242e+00,  2.13109386e+00,
            2.07954801e+00,  1.97496855e+00,  1.81887511e+00,  1.61403516e+00,
            1.36443029e+00,  1.07518898e+00,  7.52497466e-01,  4.03481768e-01,
            3.60692624e-02, -3.41173144e-01, -7.19219622e-01, -1.08877599e+00,
            -1.44048689e+00, -1.76515384e+00, -2.05395532e+00, -2.29866007e+00,
            -2.49183701e+00, -2.62706245e+00, -2.69909059e+00, -2.70402349e+00,
            -2.63945515e+00, -2.50458184e+00, -2.30027672e+00, -2.02912605e+00,
            -1.69546283e+00, -1.30531923e+00, -8.66349016e-01, -3.87719872e-01,
            1.20027335e-01,  6.45181446e-01,  1.17508637e+00,  1.69640034e+00,
            2.19538511e+00,  2.65822749e+00,  3.07136902e+00,  3.42184732e+00,
            3.69765784e+00,  3.88809314e+00,  3.98409483e+00,  3.97856162e+00,
            3.86663593e+00,  3.64597537e+00,  3.31694521e+00,  2.88276729e+00,
            2.34961788e+00,  1.72663729e+00,  1.02588068e+00,  2.62161090e-01,
            -5.47186859e-01, -1.38261324e+00, -2.22275280e+00, -3.04490847e+00,
            -3.82561418e+00, -4.54126315e+00, -5.16879424e+00, -5.68640860e+00,
            -6.07431323e+00, -6.31546039e+00, -6.39624643e+00, -6.30716531e+00,
            -6.04335936e+00, -5.60506987e+00, -4.99790976e+00, -4.23299632e+00,
            -3.32685423e+00, -2.30113152e+00, -1.18209247e+00,  1.07266825e-04,
            1.21232854e+00,  2.41953810e+00,  3.58602060e+00,  4.67666494e+00,
            5.65827361e+00,  6.50082341e+00,  7.17862168e+00,  7.67103054e+00])
    
    # -----------------------------------------------------
    # Directivity outputs using Nt=50 samples (see Line 80)
    # -----------------------------------------------------
    Ref_p_anal_RMS= np.array([3.82933369, 4.45730824, 4.58324303, 3.91971136, 2.47640332, 2.31557541,
                            3.36332969, 3.27915688, 2.78676147, 2.38125449])
    Ref_p_FWH_RMS  = np.array([3.87493135, 4.49305306, 4.57458298, 3.85695847, 2.36501532, 2.24690357,
                               
                            3.2472076,  3.16514994, 2.70406234, 2.32043177])
    Ref_p_FWH_OASPL= np.array([105.48410145, 107.37186439, 107.90332107, 106.16928309, 100.67422827,
                            101.67022266, 103.73720797, 103.09302205, 101.84770242, 100.73868732])

# -----------------------------------------------------
# Plot: Time-domain Pressure 
# -----------------------------------------------------
fig, ax = plt.subplots(figsize=(8,3), dpi=100)
if Solution_Option=='MONOPOLE':
    # ax.plot(t_anal, p_anal , ':', color='black', linewidth=5, label = 'Analytic')
    ax.plot(t_anal[:len(tau_interp.value)+10], p_anal[:len(tau_interp.value)+10] , ':', color='black', linewidth=5, label = 'Analytic')
ax.plot(tau_interp.value, p_FWH[0,:].value , linestyle='-', color='red', linewidth=2, label = 'FW-H (CSDL)')
ax.plot(Ref_tau_interp[:len(tau_interp.value)+10], Ref_p_FWH[:len(tau_interp.value)+10], '--', color='blue', linewidth=2, label = 'FW-H (SciPy)')
# NOTE: +10 is intended to caliberate the comparion for the shorter time samples. If the time data is long enougth, no visible effect is shown.

plt.xlabel(r"Observer time, $t^{'}$ (sec)",fontsize=14)
plt.ylabel(r"Acoustic pressure, $p^{'}$ (Pa)",fontsize=14)

plt.gca().tick_params(labelsize=14)
plt.legend(fontsize=14, bbox_to_anchor=(1, 1))
plt.grid()
plt.savefig('Result_CSDL_Time.png',dpi=300,bbox_inches='tight')
plt.savefig('Result_CSDL_Time.pdf',dpi=300,bbox_inches='tight')
plt.show()


# -----------------------------------------------------
# Plot: Frequency-domain Pressure (Narrowband)
# -----------------------------------------------------
if Solution_Option=='MONOPOLE':
    fig, ax = plt.subplots(figsize=(8,3), dpi=100)
    ax.semilogx(CSDL_freq.value, CSDL_Sqq[0,:].value , linestyle='-', color='red'  , label = 'FFT/SPL (CSDL)')
    # if Nobs==1:
    #     ax.semilogx(CSDL_freq.value, CSDL_Sqq.value , linestyle='-', color='red'  , label = 'FFT/SPL (CSDL)')
    # elif Nobs!=1:
    #     ax.semilogx(CSDL_freq.value, CSDL_Sqq[0,:].value , linestyle='-', color='red'  , label = 'FFT/SPL (CSDL)')
        
    Ref_f,Ref_Sqq, f13_T, SPL13_T, foc, SPLoc, Ref_OASPL = FFT_def(tau_interp.value, p_FWH[0,:].value)
    ax.semilogx(Ref_f, Ref_Sqq , linestyle='--', color='blue' , label = 'FFT/SPL (SciPy)')
    print('OASPL_TEST=',Ref_OASPL)

    plt.text(1.5,-10,f"OASPL (CSDL)={CSDL_OASPL.value[0]:.2f} dB",fontsize=14)
    plt.text(1.5,-25,f"OASPL (SciPy)={Ref_OASPL:.2f} dB",fontsize=14)


    plt.xlabel('Frequency (Hz)',fontsize=14)
    plt.ylabel('SPL (dB)',fontsize=14)

    plt.gca().tick_params(labelsize=14)
    plt.legend(fontsize=14, bbox_to_anchor=(1, 1))
    plt.grid()
    plt.savefig('Result_CSDL_Freq.png',dpi=300,bbox_inches='tight')
    plt.savefig('Result_CSDL_Freq.pdf',dpi=300,bbox_inches='tight')
    plt.show()

if Nobs != 1:
    # -----------------------------------------------------
    # Plot: Directivity (RMS)
    # -----------------------------------------------------
    fig, ax = plt.subplots(subplot_kw={'projection':'polar'}, figsize=(5,5), dpi = 100)
    ax.plot(theta_obs,p_anal_RMS,':ok', label = 'Analytic')
    ax.plot(theta_obs,p_FWH_RMS.value,'-r', label = 'FW-H (CSDL)')
    ax.plot(theta_obs,Ref_p_FWH_RMS,'--b', label = 'FW-H (SciPy)')
    fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)
    
    plt.gca().tick_params(labelsize=14)
    plt.legend(fontsize=14, bbox_to_anchor=(0.85, 0.45))
    plt.grid()
    plt.savefig('Result_CSDL_Direc_RMS.png',dpi=300, bbox_inches='tight')
    plt.savefig('Result_CSDL_Direc_RMS.pdf',dpi=300, bbox_inches='tight')
    plt.show() 
    
    # -----------------------------------------------------
    # Plot: Directivity (OASPL [dB])
    # -----------------------------------------------------
    fig, ax = plt.subplots(subplot_kw={'projection':'polar'}, figsize=(5,5), dpi = 100)
    ax.plot(theta_obs,CSDL_OASPL.value,'-r', label = 'FFT/SPL (CSDL)')
    ax.plot(theta_obs,Ref_p_FWH_OASPL,'-ob', label = 'FFT/SPL (SciPy)')
    fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)
    
    plt.gca().tick_params(labelsize=14)
    plt.legend(fontsize=14, bbox_to_anchor=(0.9, 0.45))
    plt.grid()
    plt.savefig('Result_CSDL_Direc_OASPL.png',dpi=300, bbox_inches='tight')
    plt.savefig('Result_CSDL_Direc_OASPL.pdf',dpi=300, bbox_inches='tight')
    plt.show()