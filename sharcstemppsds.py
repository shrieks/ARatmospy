import os
import pyfits as pf
import numpy  as np
import matplotlib.pyplot as mp

import generate_grids  as gg
import gen_avg_per_unb as gapu

perlen = 1024.0
f      = np.arange(perlen)
zoom   = False   
xlog   = False

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

mp.clf()
mp.ion()

fmode = 1            
#for fidx, fname in enumerate(psdfiles):
#    if fmode is 'q':
#        break
while fmode is not 'q':
    fmode = raw_input('Enter number to jump to file, q to quit: ') 
    if fmode == 'q': 
        break
    try:
        int(fmode)
    except ValueError:
        print 'Not a valid number'
        continue
    else:
        fnum = fmode.zfill(4)        
  
    try:
        fname = [f for f in psdfiles if fnum in f][0] 
    except IndexError:
        print 'File number out of range'
        continue
        
    print 'Analyzing ', fname
    psd = np.memmap(fname, dtype='float64', mode='r', shape=(perlen,bign,bign))
    #fnum = fname.split('_')[1].split('-')[0]

    mode = '2,1'
    while mode is not 'q': 
        mode = mode.split(',')
        k = int(mode[0])
        l = int(mode[1])
        mp.figure(1)
        mp.clf()
        mp.yscale('log')
        #mp.xscale('symlog')
        if zoom:
            mp.xlim(-150,150)
            legloc = 'lower center'
        elif xlog:
            mp.xscale('log')
            mp.xlim(1, np.max(hz))
            legloc = 'lower left'
        else:
            mp.xlim(np.min(hz),np.max(hz))
            legloc = 'upper left'
        
        mp.xlabel('Temporal frequency [Hz]')
        mp.ylabel('Power')
        mp.title('ShARCS Temporal PSD: %s, file %s, mode k=%s, l=%s' %(date, fnum, mode[0],mode[1]))
        psdplot, = mp.plot(shz, psd[ahz,k,l], 'r-')
        #mp.plot(shz, 0.490*eff_r0**(-5./3.)*kr[k,l]**(-11./3.)*1e-5/np.abs(1 - 0.99*np.exp(-1j*omega))**2, 'r--')
    
        #mp.legend([psdplot],['ShARCS (rate=1000, perlen=1024)'], loc=legloc)
        mp.grid(True)
        #mp.show()
        mode = raw_input('Enter mode - k,l: ')

    #fmode = raw_input('Enter to go to next file, number to jump to file, q to quit: ')