import numpy as np
import pyfits as pf
import os

import generate_grids as gg
import gen_avg_per_unb as gapu

path = '/Users/srikar/Data/ARMovies/'
#path = '/Users/srikar/ssrinath@ucsc.edu/research/wind/GPIsimulator/Residuals'
#fitsfiles = [os.path.join(root, name)
#             for root, dirs, files in os.walk(path)
#             for name in files
#             if name.endswith((".fits"))]

#fitsfiles = [path+'ARMovies/gpi_arf_rate1000_exptime22.1_amag0.99.fits', 
#             path+'ARMovies/gpi_arf_rate1000_exptime22.1_amag0.999.fits']

fitsfiles = [path+'gpi_build_mag8_rate2048_exptime4.00000_atm0_plain_aplcgrey_hband+input_phase.fits']

perlen    = 1024.0
rate      = 1000.0
n         = 48.0

bigD  = 7.770                   # primary diamete
bigDs = 1.024                   # inner M2 is 1.024 m
m     = 1.0                     # number of samples between actuators - usually 8.0
# derived quantities
bign    = n*m                 # width of phase screen for aperture
nacross = 43.0                # number of subaps across the pupil - latest GPI design is 43
# for phase samples
pscale  = bigD/((nacross)*m)  # pixel size (m) of samples in pupil plane
d       = pscale*m            # subap diameter (m)

# make the aperture
ax, ay    = gg.generate_grids(bign, scalefac=pscale)
ar        = np.sqrt(ax**2 + ay**2) # aperture radius
ap_outer  = (ar <= bigD/2)
ap_inner  = (ar <= bigDs/2)
aperture  = ap_outer - ap_inner

freq_dom_scaling = np.sqrt(bign**2/aperture.sum())    

#for root, dirs, files in os.walk(path):
#    for name in files:
#        if name.startswith(("ugp_")) and name.endswith((".fits")):
for fname in fitsfiles:
    print 'Analyzing ', fname
    fn,fe    = os.path.splitext(fname)
    FTfile   = fn+'-FT.dat'
    PSDfile  = fn+'-psd.dat'

    if os.path.isfile(PSDfile):
        continue
    else:    
        hdulist  = pf.open(fname,ignore_missing_end=True)
    
        phdim = hdulist[0].data.shape
        phx   = phdim[1]
        phy   = phdim[2]
        timesteps = phdim[0]
        print 'Computing FT'
        phFT  = np.memmap(FTfile, dtype=complex, mode='w+', shape=(timesteps,phx,phy))
        for t in np.arange(timesteps):
            wf = hdulist[0].data[t,:,:]
            phFT[t,:,:] = np.fft.fft2(wf) *freq_dom_scaling / (phx*phy)
        print 'Doing PSD'
        dtpsd = np.memmap(PSDfile, dtype='float64', mode='w+', shape=(perlen,phx,phy))
        for k in np.arange(phx):
            if k % 100 == 0:
                print k, ' ', 
            for l in np.arange(phy):
                dtpsd[:,k,l] = gapu.gen_avg_per_unb(phFT[:,k,l], perlen, meanrem=True)
        print '-----------------'

            