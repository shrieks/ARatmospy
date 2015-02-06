#;; IDL written 19 Nov 2003 by Lisa Poyneer
#
#;; this is our primary phase screen generator code

import numpy as np
import scipy.fftpack as sf
import generate_grids as gg

def get_phase_streamlined(nsub, m, dx, r0, rseed):
    ra.seeed(rseed)
    n = m*nsub
    d = dx*m
    sampfac = 1.0

    screensize_meters = n*dx
    delf = 1./screensize_meters
    #generate_grids, /freqsh, newfx, newfy, n, scale=delf, double=doubleflag
    newfx, newfy = gg.generate_grids(n, scale=delf, freqsh=True)

    powerlaw = 2*np.pi/screensize_meters*np.sqrt(0.00058)*(r0**(-5.0/6.0))* \
               (newfx**2 + newfy**2)^(-11.0/12.0)*n*np.sqrt(np.sqrt(2.))

    powerlaw[0][0] = 0.0
    newfx = 0.0
    newfy = 0.0
    
    noise = ra.normal(size=(n, n))

    phaseFT = sf.fft2(noise)

    phaseFT = phaseFT*powerlaw
    powerlaw = 0.0
  
    phase = np.real(sf.fft2(phaseFT))

    return phase

if __name__ == '__main__':
    pass
#    phase = get_phase_streamlined(

