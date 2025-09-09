import csdl_alpha as csdl
import numpy as np

import matplotlib.pyplot as plt

# Define pi
pi = np.pi

def	CSDL_Garrick_Triangle(M0, AoA, vec_1, vec_2, vec_3):
	beta2 = 1-M0**2
	d1    = vec_1
	d2    = vec_2
	d3    = vec_3
	
	# /////////////////////
	# AoA Effect (Updated)
	# /////////////////////
	# 0. Set AoA as radian
	AoA = AoA * pi/180

	# 1. AoA Coordinate Transformation : vector_d -> vector_d prime (dp)
	d1p =  csdl.cos(AoA)*d1 + csdl.sin(AoA)*d2
	d2p = -csdl.sin(AoA)*d1 + csdl.cos(AoA)*d2
	d3p = d3

	# 2. Do Garrick Triangle : vector_d prime -> vector_R_prime
	Rs = csdl.sqrt(d1p**2 + beta2* (d2p**2 + d3p**2))
	R  = (-M0*d1p + Rs) / beta2

	R1p = (-M0*Rs + d1p) / beta2
	R2p = d2p
	R3p = d3p

	# 3. Transformation Back : vector_R_prime -> vector_R
	R1 =  csdl.cos(-AoA)*R1p + csdl.sin(-AoA)*R2p
	R2 = -csdl.sin(-AoA)*R1p + csdl.cos(-AoA)*R2p
	R3 = R3p

	R_vec_1 = R1
	R_vec_2 = R2
	R_vec_3 = R3

	return Rs, R, R_vec_1, R_vec_2, R_vec_3    