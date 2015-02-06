#;; creates the starting phase screen to be used for an autoregressive
#;; atmosphere model, the powerlaw that scales random noise generated
#;; to make it look like Kolmogorov turbulence and alpha, the
#;; autoregressive paramater to scale the current phase

import numpy as np
import generate_grids as gg

def create_multilayer_arbase(n, m, pscale, rate, paramcube, alpha_mag, nofroflo=False):
#;  atm_lambda = 500.0            ;; for phase screen generator
#;  wfs_lambda = 800.0            ;; central frequency of wfs in nm

    bign = n*m
    d = pscale * m

    n_layers = len(paramcube)

    cp_r0s = paramcube[:, 0]       # r0 in meters
    cp_vels = paramcube[:, 1]      # m/s,  change to [0,0,0] to get pure boiling

    if nofroflo:
        cp_vels *= 0
    cp_dirs   = paramcube[:, 2]*np.pi/180.   # in radians

    # decompose velocities
    cp_vels_x = cp_vels*np.cos(cp_dirs)
    cp_vels_y = cp_vels*np.sin(cp_dirs)
    
    screensize_meters = bign*pscale  # extent is given by aperture size and sampling
    deltaf = 1./screensize_meters     # spatial frequency delta
    fx, fy = gg.generate_grids(bign, scalefac=deltaf, freqshift=True)
  
    powerlaw = np.zeros((bign, bign, n_layers))
    alpha = np.zeros((bign, bign, n_layers), dtype=np.complex)
    #mlphaseFT = make_array(bign, bign, n_layers, double=dflag)

    powerlaw = []
    alpha = []
    for i in range(n_layers):
        factor1 = 2*np.pi/screensize_meters*np.sqrt(0.00058)*(cp_r0s[i]**(-5.0/6.0))
        factor2 = (fx*fx + fy*fy)**(-11.0/12.0)
        factor3 = bign*np.sqrt(np.sqrt(2.))
        powerlaw.append(factor1*factor2*factor3)
        powerlaw[-1][0][0] = 0.0

        # make array for the alpha parameter and populate it
        # phase of alpha = -2pi(k*vx + l*vy)*T/Nd where T is sampling interval
        # N is WFS grid, d is subap size in meters = pscale*m, k = 2pi*fx
        # fx, fy are k/Nd and l/Nd respectively
        alpha_phase = -2*np.pi*(fx*cp_vels_x[i] + fy*cp_vels_y[i])/rate
        try:
            alpha.append(alpha_mag[i]*(np.cos(alpha_phase) + 1j*np.sin(alpha_phase)))
        except TypeError:
            alpha.append(alpha_mag*(np.cos(alpha_phase) + 1j*np.sin(alpha_phase)))

    powerlaw = np.array(powerlaw)
    alpha = np.array(alpha)

    return powerlaw, alpha

if __name__ == '__main__':
#    np.seterr(all='raise')
    import pyfits
    n = 10
    m = 100
    pscale = 1
    rate = 1
    paramcube = np.array([(0.85, 23.2, 259, 7600),
                          (1.08, 5.7, 320, 16000)])
    alpha_mag = 0.99
    powerlaw, alpha = create_multilayer_arbase(n, m, pscale, rate, paramcube, alpha_mag)

    pl_output = pyfits.HDUList()
    pl_output.append(pyfits.PrimaryHDU(data=powerlaw.transpose()))
    pl_output.writeto('powerlaw.fits', clobber=True)

    alpha_output = pyfits.HDUList()
    alpha_output.append(pyfits.PrimaryHDU(data=alpha.real))
    alpha_output.writeto('alpha.fits', clobber=True)
