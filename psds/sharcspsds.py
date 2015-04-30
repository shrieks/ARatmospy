import numpy as np
import pyfits as pf
import os

import generate_grids as gg
import gen_avg_per_unb as gapu

path = '/Users/srikar/Data/Telemetry/'
fitsfiles = [os.path.join(root, name)
             for root, dirs, files in os.walk(path)
             for name in files
             if name.startswith(("Phase_")) and name.endswith((".fits"))]

#for fname in fitsfiles:
#    hdulist = pf.open(fname,ignore_missing_end=True)
#    print fname
#    if np.isnan(np.sum(hdulist[0].data)): print 'Damn, a Nan'

perlen    = 1024.0
rate      = 1000.0
n         = 18.0

bigD  = 3.048                   # primary diameter
bigDs = 0.990                   # M2 is 33" dia, light baffle is 39" di
m     = 1.0                     # number of samples between actuators - usually 8.0
nacross = 14.0                  # number of subaps across the pupil
# derived quantities
bign    = n*m                 # width of phase screen for aperture
# for phase samples
pscale  = bigD/((nacross)*m)  # pixel size (m) of samples in pupil plane
d       = pscale*m            # subap diameter (m)

# make the aperture
ax, ay    = gg.generate_grids(bign, scalefac=pscale)
ar        = np.sqrt(ax**2 + ay**2) # aperture radius
ap_outer  = (ar <= bigD/2)
ap_inner  = (ar <= bigDs/2)
ap_inner[7,7] = 0
ap_inner[7,10] = 0
ap_inner[10,7] = 0
ap_inner[10,10] = 0
aperture  = ap_outer - ap_inner

freq_dom_scaling = np.sqrt(bign**2/aperture.sum())    

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
