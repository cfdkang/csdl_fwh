import numpy as np
import matplotlib.pyplot as plt

def Time_Derivative_1D(XX, dt):
    
    # Initializie YY whose array dimension is the same as XX.
    
	# YY=XX 
	YY = np.zeros(len(XX)) # <-- Must be this way not YY=XX
    
	# Time derivative of loading pressure using second-order scheme
	
	YY[0]    = -( 3*XX[0]   - 4*XX[1]      + XX[2]     ) / (2*dt)
	YY[-1]   =  ( XX[-3]    - 4*XX[-2]     + 3*XX[-1]  ) / (2*dt)
	YY[1:-1] =  ( XX[2:]    - XX[:-2]                  ) / (2*dt)
 
	# YY(1)       = -( 3*XX(1)   - 4*XX(2)      + XX(3)     ) / (2*dt)
	# YY(end)     =  ( XX(end-2) - 4*XX(end-1)  + 3*XX(end) ) / (2*dt)
	# YY(2:end-1) =  ( XX(3:end) - XX(1:end-2)              ) / (2*dt)
 
	return YY 
	