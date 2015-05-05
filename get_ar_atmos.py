import numpy as np
import numpy.random as ra
import scipy.fftpack as sf

def get_ar_atmos(phaseFT, powerlaw, alpha, nofroflo=False, first=False):

#;;;;;;;;;;;;;;;;;;;;;
#;; Srikar Srinath - 2014-05-07
#;; Returns one frame of an n-layer frozen flow atmosphere with boiling 
#;; sized for the telescope parameters embdedded in powerlaw and alpha
#;; Inputs:
#;;        phaseFT  - base phase screen to use as starting point
#;;                   bign x bign x n_layers array
#;;        powerlaw - powerlaw to scale noise to conform to system
#;;                   geometry
#;;        alpha    - autoregression scaling parameter
#;;                   complex array based on system geometry and
#;;                   wind r0, velocity, direction
#;;        myran    - random seed passed from calling program so
#;;                   noise generated is from same sequence
#;;        Flags:
#;;        nofroflo - flag that gives a "boiling only" atmosphere
#;;                   i.e. all frozen flow velocities set to 0
#;;        onelayer - flag to just use one  layer of atmsphere to speed
#;;                   things up
#;;        dept     - depiston and detilt the output and multiply by
#;;                   aperture
#;;        stop     - flag to activate stops at various points in the 
#;;                   code and allow variables to be examined
#;;        coyote   - use the more robust coyote random number
#;;                   generation for noise. In one session every 
#;;                   instance of the pogram will generate a new screen
#;;                   not setting the flag uses randomn which will
#;;                   give the same screen for each run with the same
#;;                   options - which has its value for comparisons
#;; Outputs:
#;;        newphase - array with bign x bign x n_layers
#
#;atm_lambda = 500.0            ;; for phase screen generator
#;wfs_lambda = 800.0            ;; central frequency of wfs in nm

    n_layers = alpha.shape[0]
    screenx = alpha.shape[1]
    screeny = alpha.shape[2]

    newphFT = []
    newphase = []

    for i in range(n_layers):
        noise = ra.normal(size=(screenx, screeny))
        noisescalefac = np.sqrt(1. - np.abs(alpha[i]**2))
        noiseFT = sf.fft2(noise)*powerlaw[i]
        if first:
            newphFT.append(noiseFT)
        else:
            newphFT.append(alpha[i]*phaseFT[i] + noiseFT*noisescalefac)
        newphase.append(sf.ifft2(newphFT[i]).real)

    return np.array(newphFT), np.array(newphase)
