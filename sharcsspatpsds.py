import pyfits as pf
import os

import numpy as np
import matplotlib.pyplot as mp
#import os.path as op

#import check_ar_atmos as caa
import generate_grids as gg
import gen_avg_per_unb as gapu

path    = '/Users/srikar/Data/Telemetry/sharcs'
date    = '20141006'
fext    = '-psd.dat'
perlen  = 1024.0

tsteps    = 4096.0
rate      = 1000.0
n         = 18.0
m         = 1.0
bigD      = 3.038               # primary diameter
bigDs     = 0.990
nacross   = 14.0                # number of subaps across the pupil

bign      = n*m                 # width of phase screen for aperture
phx       = bign
phy       = bign
pscale    = bigD/((nacross)*m)  # pixel size (m) of samples in pupil plane
d         = pscale*m  


per_len = perlen
f       = np.arange(per_len)

hz      = np.roll(f-per_len/2, np.int(per_len/2))/per_len*rate
shz     = np.sort(hz)
omega   = 2*np.pi*shz/rate
ahz     = np.argsort(hz)
kx, ky  = gg.generate_grids(phx, scalefac=2*np.pi/(bign*pscale), freqshift=True)
kr      = np.sqrt(kx**2 + ky**2)
r2m     = (0.6/(2.0*np.pi))**2  # radians to microns at 0.6 micron wavelength - WFS lambda

psdfiles = [os.path.join(root, name)
            for root, dirs, files in os.walk(path)
             for name in files
             if name.startswith(("Phase_")) and name.endswith(("-psd.dat"))]

mode = raw_input('Enter to continue')             
for fidx, fname in enumerate(psdfiles):
    if mode is 'q':
        break
    
    print 'Analyzing ', fname
    psd = np.memmap(fname, dtype='float64', mode='r', shape=(perlen,bign,bign))
    fnum = fname.split('_')[1].split('-')[0]

    mp.ion()

    mp.clf()
    mp.yscale('log')
    mp.xscale('log')
    mp.xlim(1,phx)
    mp.ylim(1e-8, 1.0)
    mp.grid(True)
    mp.title('Spatial PSDs for %s file %s'%(date, str(fnum).zfill(4)) )
    mp.xlabel('Log(Spatial frequency)')
    mp.ylabel('Log(Power)')

    mp.plot(kr, np.sum(psd,axis=0)*r2m, 'k.')
    
    mode = raw_input('Enter to continue, q to quit: ')
