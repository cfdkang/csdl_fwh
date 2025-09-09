import numpy as np
import csdl_alpha as csdl
from .CSDL_Switch2 import switch_indx2
import time
start_time = time.time()

# def linear_interp(x, y, xi, col, row):
def CSDL_Interp2(x, y, xi):
    
    row, col = x.shape  # row: N_FWHsurfi, col: Nt
    Nt = col
    interp_shape = (row,col)
    
    
    # interp_shape = (row,col)
    # interp_shape = (col,)
    yi        = csdl.Variable(value = 0, shape = interp_shape)
    dydx_sort = csdl.Variable(value = 0, shape = (row*col,))
    xi_sort   = csdl.Variable(value = 0, shape = (row*col,))
    y_sort    = csdl.Variable(value = 0, shape = (row*col,))
    
    dydx = csdl.Variable(value = 0, shape = interp_shape)
    for j in csdl.frange(row):
        dydx = dydx.set(csdl.slice[j,:-1], (y[j, 1:] - y[j, :-1])/(x[j, 1:] - x[j, :-1])  ) # USE

    
    # ------------------------------------------------    
    yi_fst = dydx[:,0] *( xi[:,0]  - x[:,0]  ) + y[:,0]
    yi_lst = dydx[:,-1]*( xi[:,-1] - x[:,-1] ) + y[:,-1]

    yi = yi.set(csdl.slice[:,0] , yi_fst)
    yi = yi.set(csdl.slice[:,-1], yi_lst)
    # ------------------------------------------------    
    
    print(f'>> switch_indx time evaluation...(AFTER): {time.time()-start_time:.4f} sec')
    
    # ---------------------
    # Most accurate so far
    # ---------------------
    interp_indx = switch_indx2(xi, x, Nt)
    indx        = interp_indx.flatten()
    

    dydx = dydx.flatten()
    xi   = xi.flatten()
    y    = y.flatten()
    print('flatten completed...')
    
    for i in csdl.frange(row*Nt):
        dydx_sort = dydx.set(csdl.slice[i], dydx[indx[i]] )
        xi_sort   = xi.set(csdl.slice[i], xi[indx[i]] )
        y_sort    = y.set(csdl.slice[i], y[indx[i]] )
    
    print('indx completed...')
    
    dydx_sort = csdl.reshape(dydx_sort, (row,Nt))
    xi_sort   = csdl.reshape(xi_sort, (row,Nt))
    y_sort    = csdl.reshape(y_sort, (row,Nt))
    print('reshape completed...')
    
        
    print(f'>> csdl.frange time evaluation...(AFTER): {time.time()-start_time:.4f} sec')
    
    x_diff = xi_sort - x
    yi     = dydx_sort*x_diff + y_sort
    
    print(f'>> Total linear_interp time evaluation...(AFTER): {time.time()-start_time:.4f} sec')
    
    return yi

