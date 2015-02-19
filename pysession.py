#%load_ext autoreload
#%autoreload 2
import numpy as np
import pyfftw as pfw
import pyfits as pf
import matplotlib.pyplot as mp
import os.path as op

import check_ar_atmos as caa
import generate_grids as gg
import gen_avg_per_unb as gapu
import depiston as dp
import detilt as dt

perlen    = 1024.0
alpha_mag = 0.99
rate      = 1000.0
n         = 48.0

fds = True  # turn on or off frequency domain scaling in FT calc
rootdir = '/Users/srikar/Data/ARMovies/'
#rootdir   = '/Users/srikar/Data/ARMovies/'
#filename  = 'ar-atmos1l-0.99-20140730_122722-psd.dat'
filename = 'ar-atmos1l-0.99-48-20140820_180148.fits'

fn, fe    = op.splitext(filename)
FTfile    = rootdir+fn+'-FT.dat'
apFTfile  = rootdir+fn+'-apFT.dat'
dtFTfile  = rootdir+fn+'-dtFT.dat'
if op.isfile(dtFTfile):
    FTmmode = 'r'   # set to 'r' if FT, psd files exist and need not be created
else:
    FTmmode = 'w+' 
    
PSDfile   = rootdir+fn+'-psd.dat'
apPSDfile = rootdir+fn+'-appsd.dat'
dtPSDfile = rootdir+fn+'-dtpsd.dat'
if op.isfile(apPSDfile):
    PSDmmode = 'r'   
else:
    PSDmmode = 'w+' 


bigD  = 7.770                   # primary diameter
bigDs = 1.024                   # inner M2 is 1.024 m
m     = 1.0                     # number of samples between actuators - usually 8.0
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

freq_dom_scaling = np.sqrt(bign**2/aperture.sum())
eff_r0 = n*d

hdulist = pf.open(rootdir+filename,ignore_missing_end=True)

phdim = hdulist[0].data.shape # output is in 
phx   = phdim[1]
phy   = phdim[2]
timesteps = phdim[0]

#phFT = np.zeros((timesteps,phx,phy), dtype=complex)
if FTmmode == 'w+':
    print "Creating FT mmap - this will take a while"
else:
    print "Reading FT dat files"
phFT   = np.memmap(FTfile, dtype=complex, mode=FTmmode, shape=(timesteps,phx,phy))
phapFT = np.memmap(apFTfile, dtype=complex, mode=FTmmode, shape=(timesteps,phx,phy))
phdtFT = np.memmap(dtFTfile, dtype=complex, mode=FTmmode, shape=(timesteps,phx,phy))
# by default, the transform is computed over the last two axes
# of the input array, i.e., a 2-dimensional FFT
#phFT = np.fft.fft2(hdulist[0].data) / (phx*phy)
if FTmmode == 'w+':
    print "Computing FT"
    for t in np.arange(timesteps):
        wf = hdulist[0].data[t,:,:]
        if fds:
	        phFT[t,:,:] = np.fft.fft2(wf) *freq_dom_scaling / (phx*phy)
        else:
            phFT[t,:,:] = np.fft.fft2(wf) / (phx*phy)

    print "Computing Apertured FT"
    for t in np.arange(timesteps):
        wf = hdulist[0].data[t,:,:]
        if fds:
            phapFT[t,:,:] = np.fft.fft2(wf*aperture) *freq_dom_scaling / (phx*phy) 
        else:
            phapFT[t,:,:] = np.fft.fft2(wf*aperture) / (phx*phy)

    print "Computing Apertured/Depistoned/Detilted FT"
    for t in np.arange(timesteps):
        wf = hdulist[0].data[t,:,:]
        if fds:
	        phdtFT[t,:,:] = (np.fft.fft2(dt.detilt(dp.depiston(wf*aperture,aperture),aperture)) *
                             freq_dom_scaling / (phx*phy))
        else:
            phdtFT[t,:,:] = np.fft.fft2(dt.detilt(dp.depiston(wf*aperture,aperture),aperture)) / (phx*phy)
##	      phFT[t,:,:] = pfw.interfaces.numpy_fft.fft(hdulist[0].data[t,:,:]) / (phx*phy)



per_len = perlen
f       = np.arange(per_len)
omega   = 2*np.pi*f/rate
hz      = np.roll(f-per_len/2, np.int(per_len/2))/per_len*rate
kx, ky  = gg.generate_grids(phx, scalefac=2*np.pi/(bign*pscale), freqshift=True)
kr      = np.sqrt(kx**2 + ky**2)

#mft = np.sum(phFT, axis=0)/timesteps
#mp.clf()
#mp.yscale('log')
#mp.xscale('log')
#mp.xlim(1,200)
#mp.ylim(1e-8, 1e2)
#mp.plot(kr, mft, 'b.')
#mp.plot(kr, 0.490*(eff_r0)**(-5./3.)*kr**(-11./3.),'r-')
#mp.show()

#this_psd = np.zeros((perlen, phx, phy),dtype=float)
this_psd = np.memmap(PSDfile, dtype='float64', mode=PSDmmode, shape=(perlen,phx,phy))
appsd    = np.memmap(apPSDfile, dtype='float64', mode=PSDmmode, shape=(perlen,phx,phy))
dtpsd    = np.memmap(dtPSDfile, dtype='float64', mode=PSDmmode, shape=(perlen,phx,phy))
if PSDmmode == 'w+':
    print "Doing PSD"
    for k in np.arange(phx):
        k20 = False
        if k%20 == 0:
            print
            k20 = True 
            print k, ":"
            
        for l in np.arange(phy):
                if k20 and l%20 == 0:
                    print l,
                this_psd[:,k,l] = gapu.gen_avg_per_unb(phFT[:,k,l], perlen, meanrem=True)
    print
    print "Doing Apertured PSD"
    for k in np.arange(phx):
        k20 = False
        if k%20 == 0:
            print
            k20 = True 
            print k, ":"
            
        for l in np.arange(phy):
                if k20 and l%20 == 0:
                    print l,
                appsd[:,k,l]    = gapu.gen_avg_per_unb(phapFT[:,k,l], perlen, meanrem=True)
    print
    print "Doing Apertured/Depistoned/Detilted PSD"
    for k in np.arange(phx):
        k20 = False
        if k%20 == 0:
            print
            k20 = True 
            print k, ":"
            
        for l in np.arange(phy):
                if k20 and l%20 == 0:
                    print l,
                dtpsd[:,k,l]    = gapu.gen_avg_per_unb(phdtFT[:,k,l], perlen, meanrem=True)

varpsd   = np.sum(this_psd, axis=0)
varappsd = np.sum(appsd, axis=0)
vardtpsd = np.sum(dtpsd, axis=0)

mp.ion()  # turn on interactive mode
mp.clf()
mp.yscale('log')
mp.xscale('log')
mp.xlim(1,phx)
mp.ylim(1e-9, 1.0)
mp.grid(True)
mp.title(r'Spatial PSD for |$\alpha$|=0.99')
mp.xlabel('Spatial frequency [1/m]')
#mp.ylabel(r'Residual phase [$\mu$m$^2$]')
mp.ylabel(r'Power')

po = mp.plot(kr, varpsd, 'b.')
pa = mp.plot(kr, varappsd,  'r.')
pd = mp.plot(kr, vardtpsd,  'g.')
pt = mp.plot(kr, 0.490*(eff_r0)**(-5./3.)*kr**(-11./3.),'k-')

mp.legend([po[0],pa[0],pd[0], pt[0]], ['Open loop (OL)', 'OL + aperture', 'OL + aperture/depiston/detilt', 'Theoretical'],loc='upper right')

#'GPI Telemetry 2013.11.12_0.27.6', 'GPI Telemetry 2014.5.9_18.49.3', 'GPI Simulation AR alpha=0.99', 'GPI Simulation AR alpha=0.999']
#mp.plot(kr, 0.490*(50.)**(-8./3.)*kr**(-10./3.), 'g-')
#mp.plot(kr, 0.490*(1500.)**(-5./3.)*kr**(-5./3.), 'r-')
#mp.annotate('0.490*50**(-8./3.)*kr**(-10./3.)', xy=(1, 1e-5), xytext=(3, 1e-5),arrowprops=dict(facecolor='black', shrink=0.05),)
#mp.annotate('0.490*1500**(-8./3.)*kr**(-5./3.)', xy=(10, 2e-7), xytext=(10, 1e-6),arrowprops=dict(facecolor='black', shrink=0.05),)
#mp.show()
foo = raw_input('Hit enter to continue:')

mode = '4,4'
while mode is not 'q': 
    mode = mode.split(',')
    k = int(mode[0])
    l = int(mode[1])
    mp.clf()
    mp.yscale('log')
    mp.xlim(-200,200)
    mp.plot(hz, this_psd[:,k,l]/np.abs(1-alpha_mag*np.exp(-1j*2*np.pi*hz/rate))**2, 'r-')
    mp.plot(hz, appsd[:,k,l]/np.abs(1-alpha_mag*np.exp(-1j*2*np.pi*hz/rate))**2, 'b-')
    mp.plot(hz, dtpsd[:,k,l]/np.abs(1-alpha_mag*np.exp(-1j*2*np.pi*hz/rate))**2, 'g-')
    #mp.plot(hz, psd99[:,k,l]/np.abs(1-0.99*np.exp(-1j*2*np.pi*hz/rate))**2, 'k-')
    mp.plot(hz, 0.490*eff_r0**(-5./3.)*kr[k,l]**(-11./3.)/np.abs(1-alpha_mag*np.exp(-1j*2*np.pi*hz/rate))**2, 'k--')
    mp.grid(True)
    mp.show()
    mode = raw_input('Enter mode - k,l: ')