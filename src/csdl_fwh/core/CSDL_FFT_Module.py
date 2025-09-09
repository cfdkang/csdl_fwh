# %%
# =============================
# Fourier Transform Test Code
# =============================
import csdl_alpha as csdl
import numpy as np
import matplotlib.pylab as plt
import time as check_time

from .FFT import FFT_def

check_time0 = check_time.time()

# //////////////////////////////////////////
# Function. 1 of 3 Fourier Transform (FFT)
# //////////////////////////////////////////
def CSDL_FFT_def(X_real, X_imag=None):

    nDFT = X_real.shape[0]
    
    if X_imag is None:
        X_imag = csdl.Variable(value=np.zeros((nDFT,)))
    
    # >> Generate bit-reversed indices
    log2_nDFT    = int(np.log2(nDFT))
    idx_ori      = np.arange(nDFT)         # Original
    idx_reversed = np.zeros_like(idx_ori)  # Reversed

    # >> Perform bit-reversal
    for bit in range(log2_nDFT):
        idx_reversed |= ((idx_ori >> bit) & 1) << (log2_nDFT - bit - 1)
    
    # >> Rearrange the array based on bit-reversed indices
    idx_reversed = idx_reversed.tolist()
    X_real = X_real.set(csdl.slice[:], value=X_real[idx_reversed])
    X_imag = X_imag.set(csdl.slice[:], value=X_imag[idx_reversed])
    
    
    even_real = csdl.Variable( value=np.zeros(int(nDFT/2)) )
    even_imag = csdl.Variable( value=np.zeros(int(nDFT/2)) )
    odd_real = csdl.Variable( value=np.zeros(int(nDFT/2)) )
    odd_imag = csdl.Variable( value=np.zeros(int(nDFT/2)) )
    # ----- Set up the pre-determined parts -----
    step          = np.zeros( int(np.log2(nDFT)) )
    half_step     = np.zeros( int(np.log2(nDFT)) )
    twiddle_range = np.zeros( (int(np.log2(nDFT)), int(nDFT/2)) ) # 4 by 8 for 16 case
    idx_even_range= np.zeros( (int(np.log2(nDFT)), int(nDFT/2)) ) 
    idx_odd_range = np.zeros( (int(np.log2(nDFT)), int(nDFT/2)) )
    
    for s in range( int(np.log2(nDFT)) ):
        step[s]            = 2**(s+1)
        half_step[s]       = int(step[s]/2)
        twiddle_range[s,:] = np.concatenate( [np.arange(half_step[s])] * int((nDFT/2) / half_step[s]) )
        
        # Fetch even and odd terms
        idx_even_range[s,:]= ( np.arange(0, nDFT, step[s])[:, None] + np.arange(half_step[s]) ).reshape(-1)
        idx_odd_range[s,:] = ( idx_even_range[s,:] + half_step[s]                             )
    # ----- Set up the pre-determined parts -----
    step           = csdl.Variable(value=step)
    half_step      = csdl.Variable(value=half_step)
    twiddle_range  = csdl.Variable(value=twiddle_range)
    idx_even_range = csdl.Variable(value=idx_even_range)
    idx_odd_range  = csdl.Variable(value=idx_odd_range)
    
    
    # >> Iterative FFT computation
    for s in csdl.frange( int(np.log2(nDFT)) ):

        # >> Option.1
        for t in csdl.frange(int(nDFT/2)):
            # Compute twiddle factors for the current step
            cos_twiddle = csdl.cos( -2 * np.pi * twiddle_range[s,t] / step[s] )
            sin_twiddle = csdl.sin( -2 * np.pi * twiddle_range[s,t] / step[s] )
            
            # Fetch even and odd terms
            idx_even = idx_even_range[s,t]
            idx_odd  = idx_odd_range[s,t]
            
            # Set  X_real // X_imag as (old) variable
            even_real = X_real[idx_even]
            even_imag = X_imag[idx_even]
            odd_real = X_real[idx_odd]
            odd_imag = X_imag[idx_odd]
            
            # Update array in place (new) : [Re(even)+Im(even)] + [Re(Twiddle)+Im(Twiddle)]*[Re(odd)+Im(odd)]
            # NOTE Im*Im = -1
            X_real = X_real.set(csdl.slice[idx_even], value=even_real + (cos_twiddle * odd_real - sin_twiddle * odd_imag) )
            X_imag = X_imag.set(csdl.slice[idx_even], value=even_imag + (cos_twiddle * odd_imag + sin_twiddle * odd_real) )
            X_real = X_real.set(csdl.slice[idx_odd], value=even_real - (cos_twiddle * odd_real - sin_twiddle * odd_imag) )
            X_imag = X_imag.set(csdl.slice[idx_odd], value=even_imag - (cos_twiddle * odd_imag + sin_twiddle * odd_real) )

    return X_real, X_imag

check_time_F = check_time.time()
print(f'>>> 1.FFT Elapsed time = {check_time_F-check_time0:.4f} sec')



# /////////////////////////////////////////////////////////////////////
# Function. 2 of 3 Power Spectral Density (length must be a power of 2)
# /////////////////////////////////////////////////////////////////////
def CSDL_SPL_def(time, x, N):
    
    # INPUT  : time, pressure defined as 'x', and # of samples 
    # OUTPUT : CSDL_freq, CSDL_f13, CSDL_Sqq(=Narrowband SPL), CSDL_SPL13, and CSDL_OASPL

    # Power Spectral Parameters
    nDFT  = int(2**int(np.floor(np.log2(N))))
    nOvlp = int(nDFT/2)
    nBlk  = int( (N-nOvlp)//(nDFT-nOvlp) )

    n  = np.arange(nDFT)+1
    WW = 0.5*(1-np.cos(2*np.pi*n/(nDFT-1)))  # Hanning Windows
    
    Aw = nDFT/csdl.sum(WW**2)

    # Determine Frequency
    CSDL_freq = csdl.Variable(value=np.zeros(int(nDFT/2+1)))
    dt        = time[1] - time[0]
    df        = 1/(nDFT*dt)

    for s in csdl.frange(int(nDFT/2+1)):
        CSDL_freq = CSDL_freq.set( csdl.slice[s],value=s*df )  # s * df

    # PSD Computation
    offset  = np.zeros( (nBlk,) )
    timeIdx = np.zeros( (nBlk,nDFT) )
        
    for iBlk in range(nBlk):
        offset[iBlk]  = min((iBlk*(nDFT-nOvlp)+nDFT), N) - nDFT
        # print('offset[iBlk]=',offset[iBlk])
        timeIdx[iBlk,:] = np.arange(nDFT) + offset[iBlk]
        # print('timeIdx[iBlk,:]=',timeIdx[iBlk,:])
        
        
    offset  = csdl.Variable(value=offset)
    timeIdx = csdl.Variable(value=timeIdx)
    
    PSD_in = csdl.Variable(value=np.zeros((nDFT,)))

    sub = csdl.Variable(value=np.zeros(nDFT//2+1)) # Tempory variable in the For-loop
    Gxx = csdl.Variable(value=np.zeros(nDFT//2+1)) # Tempory variable in the For-loop

    for iBlk in csdl.frange(nBlk):
        
        for t in csdl.frange(nDFT):
            PSD_in = PSD_in.set( csdl.slice[t], value= x[timeIdx[iBlk,t]] )

        # Apply window and FFT
        Phat_real,Phat_imag = CSDL_FFT_def(PSD_in*WW*dt)
        
        # Compute PSD
        Sxx = (1/(nDFT*dt)) * (Phat_real**2 + Phat_imag**2)

        # Only take up to Nyquist frequency
        Gxx = Gxx.set(csdl.slice[0] ,value= Sxx[0])
        Gxx = Gxx.set(csdl.slice[1:],value= 2*Sxx[1:int(nDFT/2)+1])

        # Accumulate PSD
        sub = sub+Gxx * Aw
    
    
    # Average over blocks
    sub = sub/nBlk
    
    print('CSDL_p=',x.value)
    

    # ---
    PSD_out = csdl.sqrt(sub**2) # NOTE: Don't use csdl.absolute() since it has smootheing factor,
                                #       affecting floating numbers ; solution accuracy.
    #---

    # Convert to decibels
    P_ref        = 2e-5          # Reference Pressure (Pa)
    CSDL_Sqq     = 10*csdl.log(PSD_out*df/P_ref**2, base=10)
    CSDL_OASPL   = 10*csdl.log(csdl.sum(10**(CSDL_Sqq/10)), base=10)

    check_time_F = check_time.time()
    print(f'>>> 2.PSD Elapsed time = {check_time_F-check_time0:.4f} sec')
    
    print('CSDL_freq=',CSDL_freq.value)
    print('CSDL_Sqq=',CSDL_Sqq.value)

    # ///////////////////////////////////////////////////////////////
    # Function. 3 of 3 Calculate one-third octave band SPL (SPL 1/3)
    # ///////////////////////////////////////////////////////////////

    # INPUT  : CSDL_freq, CSDL_Sqq
    # OUTPUT : CSDL_f13, CSDL_SPL13

    # Define one-third octave band center frequencies
    one_third_oc_fc=[]
    preferred_one_third_octave_fc=[10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80]
    for s in range(6):
        ten_power = 10**s
        one_third_oc_fc.append([ten_power * t for t in preferred_one_third_octave_fc]) # List Update
    one_third_oc_fc = np.array(one_third_oc_fc).reshape(-1)

    CSDL_f13 = csdl.Variable(value=one_third_oc_fc)
    Nf13     = CSDL_f13.shape[0]

    # Define lower/upper limits for one-third octave bands
    f13lower = CSDL_f13 / (2**(1/6))
    f13upper = CSDL_f13 * (2**(1/6))

    # Initialize CSDL_SPL13 with -inf
    CSDL_SPL13 = csdl.Variable(value=np.ones(Nf13,)*float('-inf'))

    for s in csdl.frange(Nf13):
        # Extract indices of freq_1/3 band
        FT13 = 0.5 * (1+csdl.tanh(10**10*(CSDL_freq-f13lower[s]))) - \
               0.5 * (1+csdl.tanh(10**10*(CSDL_freq-f13upper[s])))
        
        # CSDL summation
        CSDL_SPL13 = CSDL_SPL13.set( csdl.slice[s], value=10*csdl.log(csdl.sum(10**(CSDL_Sqq/10)*FT13),base=10) )
        
    

    return CSDL_freq, CSDL_f13, CSDL_Sqq, CSDL_SPL13, CSDL_OASPL

check_time_F = check_time.time()
print(f'>>> 3.Band-pass Integration Elapsed time = {check_time_F-check_time0:.4f} sec')        
# =/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/

