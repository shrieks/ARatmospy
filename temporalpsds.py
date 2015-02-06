import pyfits as pf
import os

import numpy as np
import matplotlib.pyplot as mp
#import os.path as op

import check_ar_atmos as caa
import generate_grids as gg
import gen_avg_per_unb as gapu

perlen = 1024.0
f      = np.arange(perlen)
zoom   = False   
xlog   = False
addgpi = True

rootdir = '/Users/srikar/Data/'

ar8k    = rootdir+'ARMovies/8k/'
ff8k    = '/Users/srikar/ssrinath@ucsc.edu/research/wind/GPIsimulator/Residuals/'
# one layer files, same r0, wind speed, direction, phase in radians
psd8k   = [ar8k+'ar-atmos1l-0.99-48-20141110_172347', 
           ar8k+'ar-atmos1l-0.999-48-20141110_173341', 
           ar8k+'gpi_build_mag8_rate2048_exptime4.00000_atm0_plain_aplcgrey_hband+input_phase', #3-layer
           ar8k+'gpi_build_mag8_rate2048_exptime4.00000_atm0_plain_hband+input_phase']  #1-layer
#          'ar-atmos1l-0.99-48-20140811_151317'

hdu8k     = pf.open(psd8k[0]+'.fits', memmap=True, ignore_missing_end=True)
tsteps    = hdu8k[0].header['n_tsteps']
rate      = hdu8k[0].header['rate']
n         = hdu8k[0].header['n_subaps']
#n         = hdu99[0].header['naxis1']
m         = hdu8k[0].header['sa_npix']
pscale    = hdu8k[0].header['pixscale']
eff_r0    = hdu8k[0].header['r00']

bign      = n*m                 # width of phase screen for aperture

psd8k99   = np.memmap(psd8k[0]+'-psd.dat', dtype='float64', mode='r', shape=(perlen,bign,bign))
psd8k999  = np.memmap(psd8k[1]+'-psd.dat', dtype='float64', mode='r', shape=(perlen,bign,bign))

psd8kffhdu  = pf.open(psd8k[2]+'-psd.fits', memmap=True, ignore_missing_end=True)
psd8kff3l   = psd8kffhdu[0].data
#psd8kff3l2  = np.memmap(psd8k[2]+'-psd.dat', dtype='float64', mode='r', shape=(perlen,bign,bign))
psd8kff1l   = np.memmap(psd8k[3]+'-psd.dat', dtype='float64', mode='r', shape=(perlen,bign,bign))

hz      = np.roll(f-perlen/2, np.int(perlen/2))/perlen*rate
shz     = np.sort(hz)         # sorted array of temporal frequency
ahz     = np.argsort(hz)      # indices of sorted temporal frequency array to use for 
                              # for putting other arrays in the same order
omega   = 2*np.pi*shz/rate
kx, ky  = gg.generate_grids(bign, scalefac=2*np.pi/(bign*pscale), freqshift=True)
kr      = np.sqrt(kx**2 + ky**2)
r2m     = (0.8/(2.0*np.pi))**2  # radians^2 to microns^2 at 0.8 micron wavelength

mp.ion()

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
    mp.title('Temporal PSD (rate=2048, perlen=1024) for mode k=%s, l=%s' %(mode[0],mode[1]))
    p99, = mp.plot(shz, psd8k99[ahz,k,l], 'r-')
    mp.plot(shz, 0.490*eff_r0**(-5./3.)*kr[k,l]**(-11./3.)*1e-5/np.abs(1 - 0.99*np.exp(-1j*omega))**2, 'r--')

    p999, = mp.plot(shz, psd8k999[ahz,k,l], 'b-')
    mp.plot(shz, 0.490*eff_r0**(-5./3.)*kr[k,l]**(-11./3.)*1e-5/np.abs(1 - 0.9991*np.exp(-1j*omega))**2, 'b--')
    
    pff, = mp.plot(shz, psd8kff3l[ahz,k,l], 'k-')
    #pff2,= mp.plot(shz, psd8kff3l2[ahz,k,l], 'k--')
    #pff3,= mp.plot(shz, psd8kff1l[ahz,k,l], 'g-')

    mp.legend([p99, p999, pff],[r'|$\alpha$|=0.99',r'|$\alpha$|=0.999','Frozen flow'],loc=legloc)
    mp.grid(True)
    #mp.show()
    mode = raw_input('Enter mode - k,l: ')

# 3-layer, 22.1s files, phase in radians
f21s = rootdir + 'ARMovies/22s/'
psd21ol = [f21s+'gpi_arf_rate1000_exptime22.1_amag0.99',
           f21s+'gpi_arf_rate1000_exptime22.1_amag0.999',
           f21s+'gpi_arf_rate1000_exptime22.1_amag0.9999']

psdffol = [f21s+'gpi_ff_rate1000_exptime22.1',
           rootdir+'gpi_build_mag8_rate1500_exptime4.00000_atm0_plain_hband+input_phase']

hdu22     = pf.open(psd21ol[0]+'.fits', memmap=True, ignore_missing_end=True)
tsteps    = hdu22[0].header['n_tsteps']
rate      = hdu22[0].header['rate']
#n         = hdu99[0].header['n_subaps']
n         = hdu22[0].header['naxis1']
m         = hdu22[0].header['sa_npix']
pscale    = hdu22[0].header['pixscale']
#eff_r0    = hdu22[0].header['r00']

bign    = n*m                 # width of phase screen for aperture
hz      = np.roll(f-perlen/2, np.int(perlen/2))/perlen*rate
shz     = np.sort(hz)         # sorted array of temporal frequency
ahz     = np.argsort(hz)      # indices of sorted temporal frequency array to use for 
                              # for putting other arrays in the same order
omega   = 2*np.pi*shz/rate
kx, ky  = gg.generate_grids(bign, scalefac=2*np.pi/(bign*pscale), freqshift=True)
kr      = np.sqrt(kx**2 + ky**2)
r2m     = (0.8/(2.0*np.pi))**2  # radians to microns at 0.8 micron wavelength

ffrate  = 1500.0
ffhz    = np.roll(f-perlen/2, np.int(perlen/2))/perlen*ffrate
ffshz   = np.sort(ffhz)         # sorted array of temporal frequency
ffahz   = np.argsort(ffhz)      # indices of sorted temporal frequency array to use for 
                              # for putting other arrays in the same order
ffomega = 2*np.pi*ffshz/ffrate


psd99       = np.memmap(psd21ol[0]+'-psd.dat', dtype='float64', mode='r', shape=(perlen,bign,bign))
psd999      = np.memmap(psd21ol[1]+'-psd.dat', dtype='float64', mode='r', shape=(perlen,bign,bign))
#psd9999     = np.memmap(psd21ol[2]+'-psd.dat', dtype='float64', mode='r', shape=(perlen,bign,bign))
psdff       = np.memmap(psdffol[0]+'-psd.dat', dtype='float64', mode='r', shape=(perlen,bign,bign))
psdffhdu2   = pf.open(psdffol[0]+'-psd.fits', memmap=True, ignore_missing_end=True)
psdff2      = psdffhdu2[0].data
psdffhdu3   = pf.open(psdffol[1]+'-psd.fits', memmap=True, ignore_missing_end=True)
psdff3      = psdffhdu3[0].data


mode = '2,1'
while mode is not 'q': 
    mode = mode.split(',')
    k = int(mode[0])
    l = int(mode[1])
    mp.figure(2)
    mp.clf()
    mp.yscale('log')
    #mp.xscale('symlog')
    if zoom:
        mp.xlim(-150,150)
    else:
        mp.xlim(np.min(hz),np.max(hz))
    mp.xlabel('Temporal frequency [Hz]')
    mp.ylabel('Power')
    mp.title('3-layer Open Loop Temporal PSD for mode k=%s, l=%s' %(mode[0],mode[1]))
    p99, = mp.plot(shz, psd99[ahz,k,l], 'r-')
    #mp.plot(shz, 0.490*eff_r0**(-5./3.)*kr[k,l]**(-11./3.)*r2m/np.abs(1 - 0.99*np.exp(-1j*omega))**2, 'r--')

    p999, = mp.plot(shz, psd999[ahz,k,l], 'b-')
    #p9999,= mp.plot(shz, psd9999[ahz,k,l]*r2m, 'g-')

    #pff,  = mp.plot(shz, psdff[ahz,k,l], 'k-')
    #pff2, = mp.plot(shz, psdff2[ahz,k,l], 'k--')
    pff3, = mp.plot(ffshz, psdff3[ffahz,k,l], 'k-')
    #pgpi, = mp.plot(shz, gpipsd[ahz,k,l]/r2m,'m--', linewidth=2)
    mp.legend([p99, p999, pff3],[r'|$\alpha$|=0.99',r'|$\alpha$|=0.999','Frozen flow'],loc=legloc)
    #mp.legend(#[p99, p999, pff, pgpi],
              #[r'|$\alpha$|=0.99',r'|$\alpha$|=0.999','Frozen Flow', 'GPI Telemetry'],
    #          loc='lower center')
    mp.grid(True)
    #mp.show()
    mode = raw_input('Enter mode - k,l: ')

# 3-layer residuals phase in microns
psd21cl = [f21s+'gpi_ar_mag8_rate1000_exptime22.0000_amag0.99_atm0_plain_aplcgrey_hband+measurements',
           f21s+'gpi_ar_mag8_rate1000_exptime22.0000_amag0.999_atm0_plain_aplcgrey_hband+measurements',
           f21s+'gpi_ar_mag8_rate1000_exptime22.0000_amag0.9999_atm0_plain_aplcgrey_hband+measurements']

psd21clo = [f21s+'gpi_ar_mag8_rate1000_exptime22.0000_amag0.99_atm0_ttopt_optfil+ofc_ar_mag8_rate1000_amag0.99_aplcgrey_hband+measurements',
            f21s+'gpi_ar_mag8_rate1000_exptime22.0000_amag0.999_atm0_ttopt_optfil+ofc_ar_mag8_rate1000_amag0.999_aplcgrey_hband+measurements']
psd21clk = [f21s+'gpi_ar_mag8_rate1000_exptime22.0000_amag0.99_atm0_ttopt_kalman+pfc_ar_mag8_rate1000_amag0.99_aplcgrey_hband+measurements',
            f21s+'gpi_ar_mag8_rate1000_exptime22.0000_amag0.999_atm0_ttopt_kalman+pfc_ar_mag8_rate1000_amag0.999_aplcgrey_hband+measurements']

psdffcl  = ['gpi_build_mag8_rate1500_exptime4.00000_atm0_plain_aplcgrey_hband+measurements']

# telemetry psds phase in microns
fgpi = rootdir+'Telemetry/gpi/'
psdgpi  = [fgpi+'ugp_When_2013.11.12_0.27.6_phase', 
           fgpi+'ugp_When_2014.5.9_18.49.38_phase',
           fgpi+'ugp_When_2013.11.12_18.52.36_phase',
           fgpi+'ugp_When_2013.11.12_21.38.35_phase',
           fgpi+'ugp_When_2013.11.12_22.27.9_phase',
           fgpi+'ugp_When_2013.11.14_2.56.26_phase',
           fgpi+'ugp_When_2013.11.14_3.0.42_phase',
           fgpi+'ugp_When_2013.11.14_3.11.15_phase',
           fgpi+'ugp_When_2013.11.14_3.13.12_phase',
           fgpi+'ugp_When_2013.11.14_5.43.23_phase',
           fgpi+'ugp_When_2013.11.14_5.47.59_phase']

psd99       = np.memmap(psd21cl[0]+'-psd.dat', dtype='float64', mode='r', shape=(perlen,bign,bign))
psd999      = np.memmap(psd21cl[1]+'-psd.dat', dtype='float64', mode='r', shape=(perlen,bign,bign))

psdff       = np.memmap(rootdir+psdffcl[0]+'-psd.dat', dtype='float64', mode='r', shape=(perlen,bign,bign))

if addgpi:
    gpipsd      = np.memmap(psdgpi[6]+'-psd.dat', dtype='float64', mode='r', shape=(perlen,bign,bign))

mode = '2,1'
while mode is not 'q': 
    mode = mode.split(',')
    k = int(mode[0])
    l = int(mode[1])
    mp.figure(3)
    mp.clf()
    mp.yscale('log')
    #mp.xscale('symlog')
    if zoom:
        mp.xlim(-150,150)
    else:
        mp.xlim(np.min(hz),np.max(hz))
    mp.xlabel('Temporal frequency [Hz]')
    mp.ylabel('Power')
    mp.title('3-layer Closed Loop Temporal PSD for mode k=%s, l=%s' %(mode[0],mode[1]))
    p99, = mp.plot(shz, psd99[ahz,k,l], 'r-')
    #mp.plot(shz, 0.490*eff_r0**(-5./3.)*kr[k,l]**(-11./3.)*r2m/np.abs(1 - 0.99*np.exp(-1j*omega))**2, 'r--')

    p999, = mp.plot(shz, psd999[ahz,k,l], 'b-')
    pff,  = mp.plot(ffshz, psdff[ffahz,k,l], 'k-')
    if addgpi:
        pgpi, = mp.plot(shz, gpipsd[ahz,k,l]/r2m,'m--', linewidth=2)
        mp.legend([p99, p999, pff, pgpi],
                  [r'|$\alpha$|=0.99',r'|$\alpha$|=0.999','Frozen Flow', 'GPI Telemetry'],
                  loc=legloc)
    else:
        mp.legend([p99, p999, pff],
                  [r'|$\alpha$|=0.99',r'|$\alpha$|=0.999','Frozen Flow'],
                  loc=legloc)
    
    mp.grid(True)
    mode = raw_input('Enter mode - k,l: ')


#hdu = pf.open(rootdir+'ARMovies/gpi_ff_rate1500_exptime1.fits', memmap=True, ignore_missing_end=True)
#hdu = pf.open(rootdir+psd8k[2]+'.fits', memmap=True, ignore_missing_end=True)
#steps = 8192.0 # 1500.0
#bigD  = 7.770                   # primary diamete
#bigDs = 1.024                   # inner M2 is 1.024 m

# make the aperture
#ax, ay    = gg.generate_grids(bign, scalefac=pscale)
#ar        = np.sqrt(ax**2 + ay**2) # aperture radius
#ap_outer  = (ar <= bigD/2)
#ap_inner  = (ar <= bigDs/2)
#aperture  = ap_outer - ap_inner

#freq_dom_scaling = np.sqrt(bign**2/aperture.sum())    

#hdu[0].header
#ffm = hdu[0].data
#ffm.shape
#ffmFT = np.zeros((steps,phx,phy), dtype='complex')
#print 'FT'
#for t in np.arange(steps):
    #ffmFT[t,:,:] = np.fft.fft2(ffm[t+50,:,:]) *freq_dom_scaling / (phx*phy)
#    ffmFT[t,:,:] = np.fft.fft2(ffm[t,:,:]) *freq_dom_scaling / (phx*phy)
#ffmpsd = np.zeros((perlen,phx,phy))
#print 'PSD'
#for k in np.arange(phx):
#    for l in np.arange(phy):
#        ffmpsd[:,k,l] = gapu.gen_avg_per_unb(ffmFT[:,k,l], perlen, meanrem=True)

#fperlen = 1024.0 #512.0
#frate   =  2048.0 #1500.0
#ff      = np.arange(fperlen)
#fhz     = np.roll(ff-fperlen/2, np.int(fperlen/2))/fperlen*frate
#fshz    = np.sort(fhz)         # sorted array of temporal frequency
#fahz    = np.argsort(fhz)      # indices of sorted temporal frequency array to use for 
                              # for putting other arrays in the same order
#fomega  = 2*np.pi*fshz/frate
#ffmpsd  = np.memmap(rootdir+'ARMovies/gpi_ff_rate1500_exptime1-psd.dat',mode='r',shape=(fperlen,bign,bign))

#for root,dirs,files in os.walk("."):
#    for name in files:
#        if name.endswith((".fits")):
#            fitsfiles.append(name)
			
#for fn in fitsfiles:
#    hdu = pf.open(fn,ignore_missing_end=True,memmap=True)
#    hdu.info()
#    hdu.close()
	
#Filename: ar-atmos1l-0.99-20140730_122722.fits
#No.    Name         Type      Cards   Dimensions   Format
#0    PRIMARY     PrimaryHDU      23   (384, 384, 8192)   float32   
#Filename: ar-atmos1l-0.99-48-20140811_151317.fits
#No.    Name         Type      Cards   Dimensions   Format
#0    PRIMARY     PrimaryHDU      26   (384, 384, 8192)   float32  

#Filename: ar-atmos1l-0.999-48-20140820_093753.fits
#No.    Name         Type      Cards   Dimensions   Format
#0    PRIMARY     PrimaryHDU      26   (384, 384, 8192)   float32   

#Filename: ar-atmos1l-0.99-48-20140820_180148.fits
#No.    Name         Type      Cards   Dimensions   Format
#0    PRIMARY     PrimaryHDU      26   (48, 48, 22100)   float32   
#Filename: ar-atmos1l-0.999-48-20140820_161824.fits
#No.    Name         Type      Cards   Dimensions   Format
#0    PRIMARY     PrimaryHDU      26   (48, 48, 22100)   float32   
