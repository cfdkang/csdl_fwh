import csdl_alpha as csdl
import numpy as np
import matplotlib.pyplot as plt

def CSDL_Time_Derivative(XX, dt):
    
	# Initializie YY whose array dimension is the same as XX.

	nobs ,nrow, ncol = XX.shape

	# YY=XX 
	YY = np.zeros([nobs, nrow, ncol]) # <-- Must be this way not YY=XX
	YY = csdl.Variable(value=YY)

	# Time derivative of loading pressure using second-order scheme
 
	YY = YY.set(csdl.slice[:,:,0]   , value = -( 3*XX[:,:,0]   - 4*XX[:,:,1]      +   XX[:,:,2]   ) / (2*dt))
	YY = YY.set(csdl.slice[:,:,-1]  , value =  (   XX[:,:,-3]  - 4*XX[:,:,-2]     + 3*XX[:,:,-1]  ) / (2*dt))
	YY = YY.set(csdl.slice[:,:,1:-1], value =  (   XX[:,:,2:]  -   XX[:,:,:-2]                    ) / (2*dt))

	return YY 
	