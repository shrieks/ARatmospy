import numpy   as np
import pyfits  as pf
import os.path as op
import matplotlib.pyplot as mp

import generate_grids as gg
import gen_avg_per_unb as gapu
import cdr_create_parameters as ccp

def check_ar_atmos(filename, perlen, alpha_mag, rate, n,
                   compare=False, dopsd=False, newfmt=False):
    # Build in capability for handling FITS files or HDF5 files
    # Add in memmap option once arrays cross 1K
    # add in parallel processing option too
    #rootdir = '/Users/srikar/Data/ARMovies/'
    rootdir = '.'

    # Gemini parameters. Change to read from FITS header or HDF5 metadata
    bigD  = 7.770                   # primary diameter
    bigDs = 1.024                   # inner M2 is 1.024 m
    m     = 8.0                     # number of samples between actuators - usually 8.0
	# derived quantities
    bign      = n*m                 # width of phase screen for aperture
    nacross   = 43.0                # number of subaps across the pupil - latest GPI design is 43
	# for phase samples
    pscale    = bigD/((nacross)*m)  # pixel size (m) of samples in pupil plane
    d         = pscale*m            # subap diameter (m)

    # make the aperture
    ax, ay    = gg.generate_grids(bign, scalefac=pscale)
    ar        = np.sqrt(ax**2 + ay**2) # aperture radius
    ap_outer  = (ar <= bigD/2)
    ap_inner  = (ar <= bigDs/2)
    aperture  = ap_outer - ap_inner
    # mp.imshow(aperture)
    # mp.show()

    freq_dom_scaling = np.sqrt(bign**2/aperture.sum())

    cp_params = ccp.cdr_create_parameters(0)
    r0s       = cp_params[:,0]
    eff_r0    = (r0s**(-5./3.)).sum()**(-3./5.)
    #eff_r0 = n*d
    
    # open the file
    hdulist = pf.open(rootdir+filename, memmap=True, ignore_missing_end=True)
    # newer FITS files (later than 2014-08-01) have header information filled in
    if newfmt:
        alpha_mag = hdulist[0].header['alphamag']
        #rate      = hdulist[0].header['rate']
        n         = hdulist[0].header['n_subaps']
        n_layers   = hdulist[0].header['n_layers']
        r0s        = np.zeros(n_layers)
        vels       = np.zeros(n_layers)
        dirs       = np.zeros(n_layers)
        for i in range(n_layers):
            r0s[i]  = hdulist[0].header['r0'+str(i)]
            vels[i] = hdulist[0].header['vel'+str(i)]
            dirs[i] = hdulist[0].header['dir'+str(i)]

        if n_layers == 1:
            eff_r0 = r0s[0]
        else:
            eff_r0 = (r0s**(-5./3.)).sum()**(-3./5.)

    phdim = hdulist[0].data.shape # output is in 
    phx   = phdim[1]
    phy   = phdim[2]
    timesteps = phdim[0]

    phFT = np.zeros((timesteps,phx,phy), dtype=complex)

    #for t in np.arange(timesteps):
    # by default, the transform is computed over the last two axes
    # of the input array, i.e., a 2-dimensional FFT
    # phFT = np.fft.fft2(hdulist[0].data) / (phx*phy) #* freq_dom_scaling 
    for t in np.arange(timesteps):
        phFT[t,:,:] = np.fft.fft2(hdulist[0].data[t,:,:]) / (phx*phy)
    print 'Done with FT'
    
    if dopsd:
        print 'Doing PSD'
        mft = np.sum(phFT, axis=0)
        
        kx, ky = gg.generate_grids(phx, scalefac=2*np.pi/(bign*pscale), freqshift=True)
        kr = np.sqrt(kx**2 + ky**2)
        f = np.arange(perlen)
        omega = 2*np.pi*f/rate
        #shift array
        hz = np.roll(f-per_len/2, np.int(per_len/2))/per_len*rate
                
        this_psd = np.zeros((perlen, phx, phy),dtype=float)
        for k in np.arange(phx):
            for l in np.arange(phy):
                this_psd[:,k,l] = gapu.gen_avg_per_unb(phFT[:,k,l], perlen, meanrem=True)

        varpsd = np.sum(this_psd, axis=0)
        # Plot spatial PSD
        #print eff_r0
        mp.clf()
        mp.yscale('log')
        mp.xscale('log')
        mp.plot(kr, varpsd, 'b.')
        mp.plot(kr, 0.490*(eff_r0)**(-5./3.)*kr**(-11./3.), 'r-')
        mp.ylim(1e-8,1e2)
        mp.xlim(1,200)
        mp.grid(True)
        mp.show()

        k = 4
        l = 4
        mp.clf()
        mp.yscale('log')
        mp.xlim(-200,200)
        mp.plot(hz, this_psd[:,k,l])
        # 0.490*eff_r0**(-5./3.)*kr[4,4]**(-11./3.)/np.abs(1-alpha_mag*np.exp(-1j*2*np.pi*hz/rate))**2
        mp.grid(True)
        mp.show()
    return
    
