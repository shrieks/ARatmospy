import os, os.path
import datetime
import numpy as np
import numpy.random as ra
import scipy.fftpack as sf
import matplotlib.pyplot as mp
from astropy.io import fits
from .generate_grids import generate_grids
from .gen_avg_per_unb import gen_avg_per_unb

def make_ar_atmos(exptime, rate, n, m, alpha_params=None, telescope='GPI', nofroflo=False, depiston=True, 
                  detilt=False, dopsd=False, phmicrons=True, savefile=False, savelayers=False, outdir='.'):   
    """
    ####################
     Srikar Srinath - 2015-02-06
     Makes an multi-layer frozen flow atmosphere with boiling sized for the
     telescope parameters in the first few lines of the program
     Inputs:
           exptime   - (float) exposure time in seconds
           rate      - (float) optical system rate/cadence in Hertz (Gemini uses 1500 Hz)
           n         - number of subapertures across screen (not the
                       number of subapertures across the aperture, which
                       is less than this number)
           m         - number of pixels per subaperture (Gemini sim uses n=48, m=8)
     Optional:
           alpha_params - array with each row corresponding to parameters for a given layer
                         [alphaMag, r0 (m), vx (m/s), vy (m/s), altitude (m)]
                         alphaMag in range [0.990,0.999]
                         (1 - alphaMag) determines fraction of phase from
                         prior timestep that is "forgotten" e.g. 
                         alpha_mag = 0.9 implies 10% forgotten
     Flags:
            telescope - parameters for various telescopes. Currently popuated below. Should become
                        a passed array or structure
            nofroflo  - flag that gives a "boiling only" atmosphere if True
                        i.e. all frozen flow velocities set to 0
            dept      - depiston and detilt the output and impose the
                        aperture
            dopsd     - calculate and return the PSD of the phase array as well
            phmicrons - convert phase from radians to microns at 500 nm
            savefile  - save a fits file with the phase screen output
                        dopsd adds on a PSD HDU as well
            savelayers- save indivdual wind layers as separate phase screen arrays
            outdir    - change to save output files to a directory other than current
     Outputs:
            phase    - fits file with bign x bign x timesteps
    """

    if savefile:
        # filnename root for a multilayer simulation-worthy datacube
        timestamp = datetime.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
        arfileroot = ('ar_'+ timestamp + '_rate'+str(np.round(rate)) + '_exptime'+str(np.round(exptime))
                     + '_n'+str(np.round(n)))
        aroutfile = os.path.join(outdir, arfileroot)
    
    if savelayers:
        layerfileroot = arfileroot+'_layer'
        layeroutfile  = os.paht.join(outdir, layerfileroot)

    timesteps = np.int(np.floor(exptime * rate))   ## number of timesteps 
    
    if dopsd:
        if timesteps < 4096: 
            per_len = 1024
        else:
            per_len = 2048

    if telescope is 'GPI':
        bigD  = 7.77010              ## primary diameter - 7.7 for Gemini, 8.4 for LSST
        bigDs = 1.024                ## inner M2 is 1.024 m for Gemini, 3.0 m for LSST
        nacross = 43.2               ## Gemini 43.2, Shane 16.0
    elif telescope is 'LSST':
        bigD  = 8.4
        bigDs = 3.0
        nacross = n
    elif 'Shane' in telescope:
        bigD  = 3.048
        bigDs = 0.990
        nacross = 16.0              ## this will have to change when 32x (30 across) mode is operational
    else:                           # Keck II numbers by default
        bigD  = 9.96                # effective circular equivalent diameter
        bigDs = 0.87                # for f/15 secondary to Naysmith platform
        nacross = 100.0             # Updated WFS from 2007 is 100x100 lenslets
    
    ## derived quantities
    bign      = np.int(n*m)               ## width of phase screen for aperture

    ## for phase samples
    pscale    = bigD/nacross ## pixel size (m) of samples in pupil plane
    d         = pscale*m   ## subap diameter (m)

    ### make the aperture to impose later if desired
    ax, ay    = generate_grids(bign, scalefac=pscale)
    ar        = np.sqrt(ax**2 + ay**2) ## aperture radius
    ap_outer  = (ar <= bigD/2)
    ap_inner  = (ar <= bigDs/2)   
    aperture  = (ap_outer ^ ap_inner).astype(int)

    # create atmosphere parameter array
    if alpha_params is None:
        # use default Cerro Pachon parameters from Tokovinin 2002 paper
        # change to relevant Mauna kea params instead?
        #                      ( r0,     vel,    dir, alt] x n_layers
        #                      meters,   m/s,degrees (0-360), meters
        cp_params = np.array([
                              #(0.40	,    6.9	,284,  0		),
                              #(0.78	,    7.5	,267,  25		),
                              #(1.07	,    7.8	,244,  50		),
                              #(1.12	,    8.3	,267,  100		),
                              #(0.84	,    9.6	,237,  200		),
                              #(0.68	,    9.9	,232,  400		),
                              #(0.66	,    9.6	,286,  800		),
                              #(0.91	,    10.1	,293,  1600		),
                              #(0.40	,    7.2	,270,  3400		),
                              #(0.50	,    16.5	,269,  6000		),
                              (0.85	,    23.2	, 59,  7600		),
                              #(1.09	,    32.7	,259,  13300	),
                              (1.08	,    5.7	,320,  16000	)])

        #print cp_params
        n_layers  = np.int(cp_params.shape[0])
        alpha_mag = [0.95, 0.99]
        r0s       = cp_params[:,0]              ## r0 in meters
        vels      = cp_params[:,1]              ## m/s,  set to 0 to get pure boiling
        dirs      = cp_params[:,2] * np.pi/180. ## in radians
        
        ## decompose velocities into components
        vx    = vels * np.cos(dirs)
        vy    = vels * np.sin(dirs)
    else:
        n_layers  = np.int(alpha_params.shape[0])
        alpha_mag = alpha_params[:,0]
        r0s       = alpha_params[:,1]
        vx        = alpha_params[:,2]
        vy        = alpha_params[:,3]
        
    
    # generate spatial frequency grids
    screensize_meters = bign * pscale
    deltaf = 1./screensize_meters           ## spatial frequency delta
    fx, fy = generate_grids(bign, scalefac=deltaf, freqshift=True)

    phase = np.zeros((bign,bign,n_layers,timesteps),dtype=float)
    # the method only needs phaseFT from the previous timestep so this is unnecessary if memory
    # constraints exist - just save FT of the phase for the current timestep and update with the new
    # FT instead. This array is useful and saves time for PSD calculations
    # phFT  = np.zeros((bign,bign,n_layers,timesteps),dtype=complex)  ## array for FT of phase
    #phrms = np.zeros((timesteps, n_layers),dtype=float)            ## phase rms at each timestep
    #phvar = np.zeros((timesteps, n_layers),dtype=float)            ## phase variance at each timestep

    for i in np.arange(n_layers, dtype=int):
        # Set the noise scaling powerlaw - the powerlaw below is from Johansson & Gavel 1994 for a 
        # Kolmogorov screen
        powerlaw = (2*np.pi*np.sqrt(0.00058)*(r0s[i]**(-5.0/6.0))*
                    (fx**2 + fy**2)**(-11.0/12.0)*bign*np.sqrt(np.sqrt(2.))/screensize_meters)
        powerlaw[0,0] = 0.0  # takes care of divide by zero when fx, fy = 0
        ## make array for the alpha parameter and populate it
        alpha_phase = - 2 * np.pi * (fx*vx[i] + fy*vy[i]) / rate

        # alpha magnitude can be different for each layer (and each Fourier mode) - to be
        # implemented later
        #if type(alpha_mag) == list:
        #    alpha = alpha_mag[i] * (np.cos(alpha_phase) + 1j * np.sin(alpha_phase))
        #else:
        #    alpha = alpha_mag * (np.cos(alpha_phase) + 1j * np.sin(alpha_phase)) 
        alpha = alpha_mag[i] * (np.cos(alpha_phase) + 1j * np.sin(alpha_phase))

        noisescalefac = np.sqrt(1 - (np.abs(alpha))**2)
        print('Layer {} alpha created'.format(str(i)))
      
        for t in np.arange(timesteps, dtype=int):
            # generate noise to be added in, FT it and scale by powerlaw
            noise = np.random.randn(bign,bign)

            ## no added noise yet, start with a regular phase screen
            noiseFT = sf.fft2(noise) * powerlaw

            if t == 0:
                wfFT = noiseFT
                # phFT[:,:,i,t] = noiseFT
            else:      
            # autoregression AR(1)
            # the new wavefront = alpha * wfnow + noise
                wfFT = alpha * wfFT + noiseFT * noisescalefac
                # wfFT = alpha * phFT[:,:,i,t-1] + noiseFT * noisescalefac
                # phFT[:,:,i,t] = wfFT
            
            # the new phase is the real_part of the inverse FT of the above
            wf = sf.ifft2(wfFT).real
            #phrms[t,i] = rms(wf)
            #phvar[t,i] = np.var(wf)
 
            # impose aperture, depiston, detilt, if desired
            
            if detilt: 
                from .detilt import detilt
                wf = detilt(wf,aperture)
                
            if depiston:
                from .depiston import depiston                
                wf = depiston(wf,aperture)*aperture
                
            phase[:,:,i,t] = wf
        
        if savelayers:
            print('Writing layer {} file to {}.fits'.format(str(i), layeroutfile+str(i)))  
            phase[:,:,i,:].shape 
            hdu = fits.PrimaryHDU(phase[:,:,i,:].transpose())
            hdu.writeto(layeroutfile+str(i)+'.fits', overwrite=True)

        print('Done with Layer {}'.format(str(i)))

    # collapse to one summed screen
    phaseout = np.sum(phase, axis=2)  # sum along layer axis, in radians of phase
    
    if phmicrons:
        phaseout /= 4.0*np.pi   # convert to microns at 500 nm
    
    if dopsd:
        # Works for m=1 so far, add code to extract apertured part of screen
        freq_dom_scaling = np.sqrt(np.float(bign)**2/aperture.sum())
        scale_for_nm = 1.0e3
        if phmicrons:
            phaseout *= scale_for_nm  # convert phase to nm
        else:
            phaseout *= scale_for_nm/(4.0*np.pi)
            
        if not depiston:
            from .depiston import depiston                
            phaseout = depiston(phaseout,aperture)*aperture
            
        phFT  = np.zeros((bign,bign,timesteps),dtype=complex)
        print('Generating Fourier modes')                        
        for t in np.arange(timesteps, dtype=int): 
            phFT[:,:,t] = sf.fft2(phaseout[:,:,t]) * freq_dom_scaling / bign**2
            
        psd  = np.zeros((bign, bign, per_len),dtype=float)    
        print('Generating PSD')
        for k in np.arange(bign, dtype=int):
            for l in np.arange(bign, dtype=int): 
                psd[k,l,:] = gen_avg_per_unb(phFT[k,l,:], per_len, meanrem=True, hanning=True, halfover=True)
                
        kx, ky = generate_grids(bign, scalefac=2*np.pi/(bign*pscale), freqshift=True)  
        kr = np.sqrt(kx**2 + ky**2)
        f = np.arange(per_len)
        omega = np.roll((np.arange(per_len, dtype=int) - per_len/2)*2*np.pi/per_len, per_len/2)
        hz = omega * rate / (2.0 * np.pi)
        varpsd = np.sum(psd, axis=2)
        
        eff_r0 = (r0s**(-5./3.)).sum()**(-3./5.)
        mp.clf()
        mp.yscale('log')
        mp.xscale('log')
        mp.plot(kr, varpsd, 'b.')
        mp.plot(kr, 0.490*(eff_r0)**(-5./3.)*kr**(-11./3.)*scale_for_nm, 'r-')
        mp.ylim(1e-8,1e2)
        mp.xlim(1,200)
        mp.grid(True)
        mp.show()
        
    
    if savefile:
        print('Writing output to {}.fits'.format(arfileroot))
        hdu = fits.PrimaryHDU(phaseout.transpose())  # for the benefit of IDL programs?
        hdu.writeto(aroutfile+'.fits', overwrite=True)
    elif dopsd:
        return phaseout, psd
    else:
        return phaseout

    print('Done generating {}-layer AR atmopshere file with params n={}, m={}, rate={}, time={}s'
           .format(n_layers, n, m, rate, exptime))
