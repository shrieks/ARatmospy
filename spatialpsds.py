import pyfits as pf
import os

import numpy as np
import matplotlib.pyplot as mp
#import os.path as op

#import check_ar_atmos as caa
import generate_grids as gg
import gen_avg_per_unb as gapu

rootdir = '/Users/srikar/Data/'
fext    = '-psd.dat'
perlen  = 1024.0

#psd21   = ['ARMovies/ar-atmos1l-0.99-48-20140820_180148', 'ARMovies/ar-atmos1l-0.999-48-20140820_161824']
psd21   = ['ARMovies/gpi_arf_rate1000_exptime22.1_amag0.99',
           'ARMovies/gpi_arf_rate1000_exptime22.1_amag0.999',
           'ARMovies/gpi_arf_rate1000_exptime22.1_amag0.9999']
psd21ol = ['gpi_ar_mag8_rate1000_exptime22.0000_amag0.99_atm0_plain_aplcgrey_hband+measurements',
           'gpi_ar_mag8_rate1000_exptime22.0000_amag0.999_atm0_plain_aplcgrey_hband+measurements',
           'gpi_ar_mag8_rate1000_exptime22.0000_amag0.9999_atm0_plain_aplcgrey_hband+measurements']
psd21clo= ['gpi_ar_mag8_rate1000_exptime22.0000_amag0.99_atm0_ttopt_optfil+ofc_ar_mag8_rate1000_amag0.99_aplcgrey_hband+measurements',
           'gpi_ar_mag8_rate1000_exptime22.0000_amag0.999_atm0_ttopt_optfil+ofc_ar_mag8_rate1000_amag0.999_aplcgrey_hband+measurements',
           'gpi_ar_mag8_rate1000_exptime22.0000_amag0.9999_atm0_ttopt_optfil+ofc_ar_mag8_rate1000_amag0.9999_aplcgrey_hband+measurements']
psd21clk= ['gpi_ar_mag8_rate1000_exptime22.0000_amag0.99_atm0_ttopt_kalman+pfc_ar_mag8_rate1000_amag0.99_aplcgrey_hband+measurements',
           'gpi_ar_mag8_rate1000_exptime22.0000_amag0.999_atm0_ttopt_kalman+pfc_ar_mag8_rate1000_amag0.999_aplcgrey_hband+measurements',
           'gpi_ar_mag8_rate1000_exptime22.0000_amag0.9999_atm0_ttopt_kalman+pfc_ar_mag8_rate1000_amag0.9999_aplcgrey_hband+measurements']
psdgpi  = ['ugp_When_2013.11.12_0.27.6_phase', 
           'ugp_When_2014.5.9_18.49.38_phase',
           'ugp_When_2013.11.12_18.52.36_phase',
           'ugp_When_2013.11.12_21.38.35_phase',
           'ugp_When_2013.11.12_22.27.9_phase',
           'ugp_When_2013.11.14_2.56.26_phase',
           'ugp_When_2013.11.14_3.0.42_phase',
           'ugp_When_2013.11.14_3.11.15_phase',
           'ugp_When_2013.11.14_3.13.12_phase',
           'ugp_When_2013.11.14_5.43.23_phase',
           'ugp_When_2013.11.14_5.47.59_phase']
psdfffn = ['gpi_build_mag8_rate1500_exptime4.00000_atm0_plain_aplcgrey_hband+measurements']

#hdu99     = pf.open(rootdir+psd8k[0]+'.fits', memmap=True, ignore_missing_end=True)
tsteps    = 22000.0
rate      = 1000.0
n         = 48.0
m         = 1.0
bigD      = 7.770                   # primary diameter
bigDs     = 1.024
#pscale    = hdu99[0].header['pixscale']
#eff_r0    = hdu99[0].header['r00']

bign      = n*m                 # width of phase screen for aperture
phx       = bign
phy       = bign
nacross   = 43.0                # number of subaps across the pupil - latest GPI design
pscale    = bigD/((nacross)*m)  # pixel size (m) of samples in pupil plane
d         = pscale*m  

psd2199   = np.memmap(rootdir+psd21[0]+fext, dtype='float64', mode='r', shape=(perlen,bign,bign))
#psd21999  = np.memmap(rootdir+psd21[1]+fext, dtype='float64', mode='r', shape=(perlen,bign,bign))
#psd219999 = np.memmap(rootdir+psd21[2]+fext, dtype='float64', mode='r', shape=(perlen,bign,bign))
psd21ol99    = np.memmap(rootdir+psd21ol[0]+fext, dtype='float64', mode='r', shape=(perlen,bign,bign))
psd21ol999   = np.memmap(rootdir+psd21ol[1]+fext, dtype='float64', mode='r', shape=(perlen,bign,bign))
psd21ol9999  = np.memmap(rootdir+psd21ol[2]+fext, dtype='float64', mode='r', shape=(perlen,bign,bign))

psd21clo99   = np.memmap(rootdir+psd21clo[0]+fext, dtype='float64', mode='r', shape=(perlen,bign,bign))
psd21clo999  = np.memmap(rootdir+psd21clo[1]+fext, dtype='float64', mode='r', shape=(perlen,bign,bign))
#psd21clo9999 = np.memmap(rootdir+psd21clo[2]+fext, dtype='float64', mode='r', shape=(perlen,bign,bign))

psd21clk99   = np.memmap(rootdir+psd21clk[0]+fext, dtype='float64', mode='r', shape=(perlen,bign,bign))
psd21clk999  = np.memmap(rootdir+psd21clk[1]+fext, dtype='float64', mode='r', shape=(perlen,bign,bign))
#psd21clk9999 = np.memmap(rootdir+psd21clk[2]+fext, dtype='float64', mode='r', shape=(perlen,bign,bign))

#psdgpi13     = np.memmap(rootdir+psdgpi[0]+fext, dtype='float64', mode='r', shape=(perlen,bign,bign))
#psdgpi14     = np.memmap(rootdir+psdgpi[1]+fext, dtype='float64', mode='r', shape=(perlen,bign,bign))

psdff = np.memmap(rootdir+psdfffn[0]+fext, dtype='float64', mode='r', shape=(perlen,48,48))

per_len = perlen
f       = np.arange(per_len)

hz      = np.roll(f-per_len/2, np.int(per_len/2))/per_len*rate
shz     = np.sort(hz)
omega   = 2*np.pi*shz/rate
ahz     = np.argsort(hz)
kx, ky  = gg.generate_grids(phx, scalefac=2*np.pi/(bign*pscale), freqshift=True)
kr      = np.sqrt(kx**2 + ky**2)
r2m     = (0.8/(2.0*np.pi))**2  # radians to microns at 0.8 micron wavelength

mp.ion()

mp.clf()
mp.yscale('log')
mp.xscale('log')
mp.xlim(1,phx)
mp.ylim(1e-7, 1.0)
mp.grid(True)
mp.title('Spatial PSDs')
mp.xlabel('Log(Spatial frequency)')
mp.ylabel('Log(Power)')

mp.plot(kr, np.sum(psd2199,axis=0)*r2m, 'k.')

#mp.plot(kr, np.sum(psd21ol99,axis=0)*r2m, 'r.')
#mp.plot(kr, np.sum(psd21ol999,axis=0)*r2m, 'b.')
#mp.plot(kr, np.sum(psd21ol9999,axis=0)*r2m, 'g.')

#mp.plot(kr, np.sum(psd21clo99,axis=0)*r2m, 'g.')
#mp.plot(kr, np.sum(psd21clo999,axis=0)*r2m, 'c.')
#mp.plot(kr, np.sum(psd21clo9999,axis=0)*r2m, 'y.')

mp.plot(kr, np.sum(psd21clk99,axis=0)*r2m, 'y.')
mp.plot(kr, np.sum(psd21clk999,axis=0)*r2m, 'm.')
#mp.plot(kr, np.sum(psd21clk9999,axis=0)*r2m, 'y.')

#mp.plot(kr, np.sum(psdgpi13,axis=0), 'ko', markersize=1)
#mp.plot(kr, np.sum(psdgpi14,axis=0), 'ko', markersize=1)

mp.plot(kr, np.sum(psdff,axis=0)*r2m, 'mp', markersize=3)

#mp.legend(loc='upper right')
mode = raw_input('Enter to continue')

for file in psdgpi:    
    if mode is 'q':
        break
    mp.clf()
    mp.yscale('log')
    mp.xscale('log')
    mp.xlim(1,phx)
    mp.ylim(1e-9, 1.0)
    mp.grid(True)
    mp.title('Spatial PSDs')
    mp.xlabel('Log(Spatial frequency)')
    mp.ylabel('Log(Power)')

    mp.plot(kr, np.sum(psd2199,axis=0)*r2m, 'k.')

    mp.plot(kr, np.sum(psd21ol99,axis=0)*r2m, 'r.')
    mp.plot(kr, np.sum(psd21ol999,axis=0)*r2m, 'b.')
    mp.plot(kr, np.sum(psd21ol9999,axis=0)*r2m, 'g.')
    mp.plot(kr, np.sum(psdff,axis=0)*r2m, 'mp', markersize=3)
    
    gpipsd = np.memmap(rootdir+file+fext, dtype='float64', mode='r', shape=(perlen,48,48))
    mp.plot(kr, np.sum(gpipsd,axis=0), 'cs', markersize=3)
    #mp.legend(loc='upper right')
    mode = raw_input('Enter to continue, q to quit: ')
