import numpy as np
from scipy.signal import csd, windows
import math

def subroutine_SPL_Family(f, SPL_Narrow):
    """
    Convert narrowband SPL to one-third octave band and octave band SPLs.
    
    Parameters:
    - f         : array of narrowband frequency bins
    - SPL_Narrow: array of narrowband sound pressure levels in dB
    
    Returns:
    - f13       : array of one-third octave band center frequencies
    - SPL13     : array of one-third octave band SPLs in dB
    - foc       : array of octave band center frequencies
    - SPLoc     : array of octave band SPLs in dB
    - OASPL     : overall sound pressure level for octave bands in dB
    """
    # Define one-third octave band center frequencies
    preferred_one_third_octave_fc = [10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80]
    one_third_oc_fc = (preferred_one_third_octave_fc +
                       [x * 10     for x in preferred_one_third_octave_fc] +
                       [x * 100    for x in preferred_one_third_octave_fc] +
                       [x * 1000   for x in preferred_one_third_octave_fc] +
                       [x * 10000  for x in preferred_one_third_octave_fc] +
                       [x * 100000 for x in preferred_one_third_octave_fc])
    one_third_oc_fc = np.array(one_third_oc_fc) # Take list first and then convert to np.array
    
    # Define lower/upper limits for one-third octave bands
    f13lower = one_third_oc_fc / (2**(1/6))
    f13upper = one_third_oc_fc * (2**(1/6))
    
    # Find valid one-third octave bands within the frequency range
    one_third_oc_fc_end_point = np.where(one_third_oc_fc<f[-1])[0][-1] + 1
    # [0]: First tuple
    # [-1]: last index       
    
    # Calculate one-third octave SPL
    SPL_one_third_octave = []
    for i in range(one_third_oc_fc_end_point):
        indices = np.where((f >= f13lower[i]) & (f <= f13upper[i]))[0]
        if indices.size > 0:
            # Sum power in linear scale
            SPL_one_third_octave.append(10 * np.log10(np.sum(10**(SPL_Narrow[indices]/10))))
        else:
            SPL_one_third_octave.append(float('-inf'))  # Handle cases with no data points
    
    # Define octave band center frequencies
    oc_fc = one_third_oc_fc[2::3][:len(one_third_oc_fc) // 3] # it selects every 3rd element from the starting index (2).
    
    # Define lower/upper limits for octave bands
    foc_lower = oc_fc / (2**(1/2))
    foc_upper = oc_fc * (2**(1/2))
    
    # Find valid octave bands within the frequency range
    oc_fc_end_point = np.where(foc_upper < f[-1])[0][-1] + 1
    
    # Calculate octave SPL
    SPL_octave = []
    for i in range(oc_fc_end_point):
        indices = np.where((f >= foc_lower[i]) & (f <= foc_upper[i]))[0]
        if indices.size > 0:
            SPL_octave.append(10*np.log10(np.sum(10**(SPL_Narrow[indices]/10))))
        else:
            SPL_octave.append(float('-inf'))  # Handle cases with no data points
    
    # Calculate overall SPL for octave bands
    octave_OASPL_LP = 10**(np.array(SPL_octave)/10)
    OASPL_Octave    = 10 * np.log10(np.sum(octave_OASPL_LP))
    
    # Final outputs
    f13   = one_third_oc_fc[:one_third_oc_fc_end_point]
    SPL13 = np.array(SPL_one_third_octave)
    foc   = oc_fc[:oc_fc_end_point]
    SPLoc = np.array(SPL_octave)
    
    TEMP   = 10**(np.array(SPLoc)/10)
    OASPL  = 10 * np.log10(np.sum(TEMP))
    print('OASPL(Octave)=',OASPL)
    
    TEMP   = 10**(np.array(SPL13)/10)
    OASPL  = 10 * np.log10(np.sum(TEMP))
    print('OASPL(1/3)=',OASPL)
    
    TEMP   = 10**(np.array(SPL_Narrow)/10)
    OASPL  = 10 * np.log10(np.sum(TEMP))
    print('OASPL(Narrow)=',OASPL)    
    
    return f13, SPL13, foc, SPLoc, OASPL

def FFT_def(time, x):

    # =================================
    # Step.1 FFT -> OUTPUT: f, cpsd_Gxx
    # =================================

    # Input parameters
    dt = time[1] - time[0]
    Fs = 1 / dt     # Sampling rate [Hz]
    Nt = len(time)
    Ndata = int(2**int(np.floor(np.log2(Nt))))
    
    Noverlap = 0.5  # Overlap percentage

    # Output parameters
    Nshift = int(Ndata * Noverlap)
    Nseg = int(len(x) / Nshift) - int(1 / Noverlap - 1)

    print(f"Windowing Info: overlapping {Noverlap * 100:.2f} percent, # of Segments: {Nseg}")
    print(f"# of Data/Segment (Nwin): {Ndata}")
    
    # Compute the cross power spectral density (equivalent to cpsd in MATLAB)
    # f, cpsd_Gxx = csd(x, x, fs=Fs, window=windows.hann(Ndata), nperseg=Ndata, noverlap=Nshift, nfft=Ndata, scaling='density')
    f, cpsd_Gxx = csd(x, x, fs=Fs, window=windows.hann(Ndata), nperseg=Ndata, noverlap=Nshift, nfft=Ndata)
    
    # Delta frequency and reference pressure
    df    = Fs / Ndata
    P_ref = 2e-5  # Reference Pressure, [Pa]
    
    # Convert power spectral density to decibels
    cpsd_Gxx = 10 * np.log10(np.abs(cpsd_Gxx) * df / P_ref**2)


    # =====================================
    # Step.2 SPL_Narrow ->SPL_13 with OASPL
    # =====================================
    f13, SPL13, foc, SPLoc, OASPL = subroutine_SPL_Family(f, cpsd_Gxx)
    
    return f, cpsd_Gxx, f13, SPL13, foc, SPLoc, OASPL
