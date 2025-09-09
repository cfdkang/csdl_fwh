import csdl_alpha as csdl
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .Garrick_Triangle    import Garrick_Triangle  # Garrick Triangle
from .Time_Derivative_1D  import Time_Derivative_1D   # Time derivative
from .rms                 import rms

# Define pi
pi = np.pi

# # ...
# import os
# #file_path = './Source_Input.txt'
# file_path = os.path.join('./Source_Input.txt')
# # -------------------------
# # User defined parameters
# # -------------------------
# RAW = []
# with open(file_path,'r') as f:
# 	for i, line in enumerate(f):        
# 		if i < 4:
# 			# (1) Divide string and value by space and then (2) take only value
# 			RAW.append(line.split()[1]) 

# A         = float(RAW[0])  # Monopole or dipole source strength (amplitude)
# f         = float(RAW[1])  # Monopole or dipole pulsation [source] frequency [Hz]
# omega     = 2*pi*f         # Monopole or dipole pulsation [source] angular frequency [rad/s]

# R_rot     = float(RAW[2])  # Rotational radius
# f_rot     = float(RAW[3])  # Rotational frequency [Hz]
# omega_rot = 2*pi*f_rot     # Rotational angular frequency [rad/s]
# # ...


A         = 1          # Monopole or dipole source strength (amplitude)
f         = 5          # Monopole or dipole pulsation [source] frequency [Hz]
omega     = 2*pi*f     # Monopole or dipolePulsation [source] angular frequency [rad/s]

R_rot     = 0.7        # Rotational radius [m]
f_rot     = 1          # Rotational frequency [Hz]
omega_rot = 2*pi*f_rot # Rotational angular frequency [rad/s]



# << OUTPUT >>
# p_anal_1    : time derivative of fai
# p_anal_2    : spaital derivative of fai  
# pp_anal     : p_timederi+pl_spatialderi  : p'
# p_anal_RMS  : RMS pressure for different directional angles

# << INPUT >>
# [Xo,Yo,Zo]      : observer coordinate
# rho_0           : mean density
# U01             : streamwise  convective mean flow
# U02             : wall-normal convective mean flow
# M0              : freestream Mach number
# AoA             : angle of attack
# t               : source time [sec]
# dt              : source time step [sec]
# c               : speed of sound [m/s]
# Solution_Option : "Monopole" or "Dipole"

# XXXXXX THIS IS EASILY MISTAKBLE XXXXX
# Ux = Ux + U_mov_1(i,:) + U01
# Uy = Uy + U_mov_2(i,:) + U02
# Uz = Uz + U_mov_3(i,:)
# XXXXXX THIS IS EASILY MISTAKBLE XXXXX


def Generator_1_Analytic_Solution_def(Xo, Yo, Zo, \
                                      rho_0, U01, U02, M0, AoA, t, dt, c, Solution_Option):
    
    # --------------- MOVING Source (function of source time ) ---------------
	# ------------
	# Moving Term
	# ------------
	Nt = len(t)
	y_mov_1 = R_rot*np.cos(omega_rot*t) # % 1 by Nt
	y_mov_2 = R_rot*np.sin(omega_rot*t)
	y_mov_3 = np.zeros(Nt)
	# --------------- MOVING Source (function of source time ) ---------------
	

	p_anal_RMS = np.zeros(len(Xo))	
	

	if Solution_Option == 'MONOPOLE':
		for qq in range(len(Xo)):
			
			# Time-varying source-to-observer distance
			Anal_Vec_1 = Xo[qq] - y_mov_1
			Anal_Vec_2 = Yo[qq] - y_mov_2
			Anal_Vec_3 = Zo[qq] - y_mov_3

			# print(f"Anal_Vec_3.shape={Anal_Vec_3.shape}")

			# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
			# <<<<<<<<<<<<<<<<<<<<<< MONOPOLE >>>>>>>>>>>>>>>>>>>
			# ANALYTICAL TERM 1 : Pressure
			Rs, R0, _, _, _ = Garrick_Triangle(M0, AoA, Anal_Vec_1, Anal_Vec_2, Anal_Vec_3)

			# plt.plot(1/Rs) # plt.plot(Rs)
			# plt.plot(1/R0) # plt.plot(R0)
   
			# plt.plot(A/(4*pi*Rs)) # plt.plot(R0)

      
			dRsdt = Time_Derivative_1D(Rs,dt)
			dR0dt = Time_Derivative_1D(R0,dt)
			
			# plt.plot(dRsdt) # plt.plot(dRsdt)
			# plt.plot(dR0dt) # plt.plot(dR0dt)
			

			# .............
			# Original form
			# .............
			# dfaidt = (1j*omega) * A/(4*pi*Rs) * np.exp(1j*omega*(t-R0/c))
			

			# ......................................................................................
			# Below is derived when terms Rs and R are a function of time: 
			# Functional form: 
			#     fai   = 1/(g(t))*exp(iw*(t+f(t))
			# d(fai)/dt = -g'(t)/(g(t)^2)*exp(iw*(t+f(t))+ 1/(g(t))*(iw*(1+f'(t)))* exp(iw*(t+f(t))
			# ......................................................................................
			dfaidt = -dRsdt /(Rs**2) * A/(4*pi*Rs)*np.exp(1j*omega*(t-R0/c)) + \
								       A/(4*pi*Rs)*(1j*omega)*(1-dR0dt/c)* np.exp(1j*omega*(t-R0/c))
			
			# plt.plot(dR0dt.real)
			
			
			perturb = 1E-08 # % used for getting spatial derivative term

			# ANALYTICAL TERM 2-1 : Velocity (U01)
			Rs1_plus , R1_plus , _, _, _ = Garrick_Triangle(M0, AoA, Anal_Vec_1+perturb, Anal_Vec_2, Anal_Vec_3)
			Rs1_minus, R1_minus, _, _, _ = Garrick_Triangle(M0, AoA, Anal_Vec_1-perturb, Anal_Vec_2, Anal_Vec_3)

			fai1_plus  = A/(4*pi*Rs1_plus )*np.exp( 1j*omega*( t-R1_plus/c)  )
			fai1_minus = A/(4*pi*Rs1_minus)*np.exp( 1j*omega*( t-R1_minus/c) )

			dfaidx1 = (fai1_plus - fai1_minus) / (2*perturb)


			# ANALYTICAL TERM 1 2-2 : Velocity (U02)
			Rs2_plus , R2_plus , _, _, _ = Garrick_Triangle(M0, AoA, Anal_Vec_1, Anal_Vec_2+perturb, Anal_Vec_3)
			Rs2_minus, R2_minus, _, _, _ = Garrick_Triangle(M0, AoA, Anal_Vec_1, Anal_Vec_2-perturb, Anal_Vec_3)

			fai2_plus  = A/(4*pi*Rs2_plus )*np.exp( 1j*omega*( t-R2_plus/c) )
			fai2_minus = A/(4*pi*Rs2_minus)*np.exp( 1j*omega*( t-R2_minus/c) )

			dfaidx2 = (fai2_plus - fai2_minus) / (2*perturb)


			p_anal_1 = - rho_0* ( dfaidt )                    # Time-derivative
			p_anal_2 = - rho_0* ( U01*dfaidx1 + U02*dfaidx2 ) # Spatial-derivative

			p_anal_1 = p_anal_1.real
			p_anal_2 = p_anal_2.real

			p_anal   = p_anal_1 + p_anal_2  # Total
			
			# Store all the results in array
			p_anal_RMS[qq] = rms(p_anal_1 + p_anal_2)

			# <<<<<<<<<<<<<<<<<<<<<< MONOPOLE >>>>>>>>>>>>>>>>>>>
			# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.

	
	elif Solution_Option == 'DIPOLE':

		for qq in range(len(Xo)):
			# Time-varying source-to-observer distance
			Anal_Vec_1 = Xo[qq] - y_mov_1
			Anal_Vec_2 = Yo[qq] - y_mov_2
			Anal_Vec_3 = Zo[qq] - y_mov_3
			
			# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
			# <<<<<<<<<<<<<<<<<<<<<< DIPOLE >>>>>>>>>>>>>>>>>>>

			perturb = 1E-03 # used for getting spatial derivative term

			# ANALYTICAL TERM 1 : d/dy-based Pressure : d^2 p / (dt dy)
			Rs_plus , R0_plus , _, _, _ = Garrick_Triangle(M0, AoA, Anal_Vec_1, Anal_Vec_2+perturb, Anal_Vec_3)
			Rs_minus, R0_minus, _, _, _ = Garrick_Triangle(M0, AoA, Anal_Vec_1, Anal_Vec_2-perturb, Anal_Vec_3)

			dfaidt_plus  = A/(4*pi*Rs_plus  )*np.exp( 1j*omega*( t-R0_plus/c)  )
			dfaidt_minus = A/(4*pi*Rs_minus )*np.exp( 1j*omega*( t-R0_minus/c) )

			dfaidt = Time_Derivative_1D(  (dfaidt_plus-dfaidt_minus)/(2*perturb) ,dt  )
		    # dfaidt = gradient(  (dfaidt_plus-dfaidt_minus)/(2*perturb) ,t  )



			# ANALYTICAL TERM 2-1 : d/dy-based Velocity (U01) : d^2 U / (dx dy)
			Rs1_11 , R1_11 , _, _, _ = Garrick_Triangle(M0, AoA, Anal_Vec_1+perturb, Anal_Vec_2+perturb, Anal_Vec_3 )
			Rs1_22 , R1_22 , _, _, _ = Garrick_Triangle(M0, AoA, Anal_Vec_1+perturb, Anal_Vec_2-perturb, Anal_Vec_3 )
			Rs1_33 , R1_33 , _, _, _ = Garrick_Triangle(M0, AoA, Anal_Vec_1-perturb, Anal_Vec_2+perturb, Anal_Vec_3 )
			Rs1_44 , R1_44 , _, _, _ = Garrick_Triangle(M0, AoA, Anal_Vec_1-perturb, Anal_Vec_2-perturb, Anal_Vec_3 )

			fai1_11  = A/(4*pi*Rs1_11 )*np.exp( 1j*omega*( t-R1_11/c) )
			fai1_22  = A/(4*pi*Rs1_22 )*np.exp( 1j*omega*( t-R1_22/c) )
			fai1_33  = A/(4*pi*Rs1_33 )*np.exp( 1j*omega*( t-R1_33/c) )
			fai1_44  = A/(4*pi*Rs1_44 )*np.exp( 1j*omega*( t-R1_44/c) )

			dfaidx1 = (fai1_11 + fai1_44 - fai1_22 - fai1_33) / (4*perturb*perturb)


			# ANALYTICAL TERM 1 2-2 : d/dy-based Velocity (U02) : d^2 U / (dy dy)
			Rs2_plus , R2_plus , _, _, _ = Garrick_Triangle(M0, AoA, Anal_Vec_1, Anal_Vec_2+perturb, Anal_Vec_3 )
			Rs2_mid  , R2_mid  , _, _, _ = Garrick_Triangle(M0, AoA, Anal_Vec_1, Anal_Vec_2        , Anal_Vec_3 )
			Rs2_minus, R2_minus, _, _, _ = Garrick_Triangle(M0, AoA, Anal_Vec_1, Anal_Vec_2-perturb, Anal_Vec_3 )

			fai2_plus  = A/(4*pi*Rs2_plus )*np.exp( 1j*omega*( t-R2_plus/c)  )
			fai2_mid   = A/(4*pi*Rs2_mid  )*np.exp( 1j*omega*( t-R2_mid /c)  )
			fai2_minus = A/(4*pi*Rs2_minus)*np.exp( 1j*omega*( t-R2_minus/c) )

			dfaidx2 = (fai2_plus - 2*fai2_mid + fai2_minus) / (perturb*perturb)

			p_anal_1 = - rho_0* ( dfaidt)                     # Time-derivative
			p_anal_2 = - rho_0* ( U01*dfaidx1 + U02*dfaidx2)  # Spatial-derivative
   
			p_anal_1 = p_anal_1.real
			p_anal_2 = p_anal_2.real   
   
			p_anal   = p_anal_1 + p_anal_2 # Total
			
			# Store all the results in array
			p_anal_RMS[qq] = rms(p_anal_1 + p_anal_2)

			# <<<<<<<<<<<<<<<<<<<<<< DIPOLE >>>>>>>>>>>>>>>>>>>
			# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
	
	
	# # CHECK Analytical result
	# fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
	# plt.rc('font',family = 'Times New Roman')
	
	# ax.plot(t,p_anal_1, linestyle='-',color='red', label='p_anal_1')
	# ax.plot(t,p_anal_2, linestyle='-.',color='blue', label='p_anal_2')
	# ax.plot(t,p_anal  , linestyle='--',color='black', label='p_anal')

	# plt.legend()
	# plt.grid()
	# plt.show()
   
	return p_anal_1, p_anal_2, p_anal, p_anal_RMS

    # CHECK Analytical result: plot(t,p_anal_1); hold on; plot(t,p_anal_2); plot(t,p_anal,'--');




    