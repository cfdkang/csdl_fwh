import numpy as np
import csdl_alpha as csdl
# Initially written by Luca / date: 07.02.2024 
# Updated version for index finding / date: 03.05.2025


def switch_indx2(x, bounds_list, Nt, scale=1e6):
    
    # row = x.shape[0]
    row, col = x.shape

    funcs_list = csdl.Variable(value = 2*np.arange(Nt)-1)
    funcs_list = funcs_list.set(csdl.slice[0], value=0)
    
    # --------------------------------------------------------
    funcs_list = csdl.expand(funcs_list, (row,col),'i->ji')
    
    f_0   = csdl.Variable(value=0, shape=(row,col) )
    f_end = csdl.Variable(value=0, shape=(row,col) )
    x_0   = csdl.Variable(value=0, shape=(row,col) )
    x_end = csdl.Variable(value=0, shape=(row,col) )
    
    f_0   = f_0.set(csdl.slice[:,0]    , funcs_list[:,0]  )
    f_end = f_end.set(csdl.slice[:,-1] , funcs_list[:,-1] )
    x_0   = x_0.set(csdl.slice[:,0]    , bounds_list[:,0] )
    x_end = x_end.set(csdl.slice[:,-1] , bounds_list[:,-1])
    # --------------------------------------------------------

    y = f_0*(0.5*csdl.tanh(scale*(x_0-x)) + 0.5)
    
    # ---------------------------------------------
    f_i = csdl.Variable(value=0, shape=(row,col) )
    x_l = csdl.Variable(value=0, shape=(row,col) )
    x_h = csdl.Variable(value=0, shape=(row,col) )
    # ---------------------------------------------
    
    
    for i in csdl.frange(Nt-1):
        
        
        # Method.1
        # ----------------------------------------------------
        f_i = csdl.expand(funcs_list[:,i+1] , (row,col), 'i->ij')  
        x_l = csdl.expand(bounds_list[:,i]  , (row,col), 'i->ij')  
        x_h = csdl.expand(bounds_list[:,i+1], (row,col), 'i->ij')  
        # ----------------------------------------------------
        y = y + f_i * (0.5*(csdl.tanh(scale*(x-x_l)) - csdl.tanh(scale*(x-x_h))))
        # y = y + (f_i * (0.5*(csdl.tanh(scale*(x-x_l)) - csdl.tanh(scale*(x-x_h))))+1)/2-1
        
        # # -----------------initialize-------------------------
        # f_i = f_i.set(csdl.slice[:,i+1] , 0 )
        # x_l = x_l.set(csdl.slice[:,i]   , 0 )
        # x_h = x_h.set(csdl.slice[:,i+1] , 0 )
        # # ----------------------------------------------------
        
        
        # # Method.2
        # # ----------------------------------------------------
        # f_i = f_i.set(csdl.slice[:,i+1] , funcs_list[:,i+1]  )
        # x_l = x_l.set(csdl.slice[:,i]   , bounds_list[:,i]   )
        # x_h = x_h.set(csdl.slice[:,i+1] , bounds_list[:,i+1] )
        # # ----------------------------------------------------
        # y = y + f_i * (0.5*(csdl.tanh(scale*(x-x_l)) - csdl.tanh(scale*(x-x_h))))
        
        # # # -----------------initialize-------------------------
        # # f_i = f_i.set(csdl.slice[:,i+1] , 0 )
        # # x_l = x_l.set(csdl.slice[:,i]   , 0 )
        # # x_h = x_h.set(csdl.slice[:,i+1] , 0 )
        # # # ----------------------------------------------------
        

    y = y + f_end * (0.5*csdl.tanh(scale*(x-x_end)) + 0.5)
    right_indx = (y + 1)/2 -1
    
    # print('right_indx.value=',right_indx.value)
    return right_indx
