import numpy as np
import numpy.random as ra
import scipy.fftpack as sf
import pyfits as pf
import generate_grids as gg
import os, os.path

def make_ar_atmos(exptime, rate, alphaParams, n, m, telescope='GPI', nofroflo=False, dept=False, vxvy=True):   
    """
    ######################
    ## Srikar Srinath - 2015-02-06
    ## Makes an multi-layer frozen flow atmosphere with boiling sized for the
    ## telescope parameters in the first few lines of the program
    ## Inputs:
    ##       exptime   - (float) exposure time in seconds
    ##       rate      - (float) optical system rate/cadence in Hertz (Gemini uses 1500 Hz)
    ##       alphaParams - array with each row corresponding to parameters for a given layer
    ##                     [alphaMag, r0 (m), vx (m/s), vy (m/s), altitude (m)]
    ##                     If vxvy flag set to False then
    ##                     [alphaMag, r0 (m), v (m/s), direction (degrees clockwise 0=North), altitude (m)]
    ##                     alphaMag in range [0.990,0.999]
    ##                     (1 - alphaMag) determines fraction of phase from
    ##                     prior timestep that is "forgotten" e.g. 
    ##                     alpha_mag = 0.9 implies 10% forgotten
    ##       n         - number of subapertures across screen (not the
    ##                   number of subapertures across the aperture, which
    ##                   is less than this number)
    ##       m         - number of pixels per subaperture (Gemini sim uses n=48, m=8)
    ## Flags:
    ##        nofroflo - flag that gives a "boiling only" atmosphere if True
    ##                   i.e. all frozen flow velocities set to 0
    ##        dept     - depiston and detilt the output and impose the
    ##                   aperture
    ##        vxvy     - If True then alphaParams array has wind information given as vx and vy
    ##                   else wind information is velocity (m/s) and direction (degrees)
    ##        telescope - parameters for various telescopes. Currently popuated below. Should become
    ##                    a passed array or structure
    ##        hdf5     - Save to HDF5 format instead
    ## Outputs:
    ##        phase    - fits file with bign x bign x timesteps
    """

    rootdir = '.'

    # filnename root for a multilayer simulation-worthy datacube
    arfileroot = rootdir +'aratmos'+'_rate'+str(np.round(rate))+'_exptime'+str(exptime)+'_amag'+str(alpha_mag)
    layerfileroot = arfileroot+'-layer'

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
    else:
    ## derived quantities
    bign      = n*m               ## width of phase screen for aperture

    ## for phase samples
    pscale    = bigD/nacross ## pixel size (m) of samples in pupil plane
    d         = pscale*m   ## subap diameter (m)

    ### make the aperture to impose later if desired
    ax, ay    = gg.generate_grids(bign, scalefac=pscale)
    ar        = np.sqrt(ax**2 + ay**2) ## aperture radius
    ap_outer  = (ar <= bigD/2)
    ap_inner  = (ar <= bigDs/2)   
    aperture  = (ap_outer - ap_inner).astype(int)

    timesteps = exptime * rate #np.floor(exptime * rate)   ## number of timesteps 

    # create atmosphere parameter array, 
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
    n_layers  = cp_params.shape[0]

    r0s       = cp_params[:,0]              ## r0 in meters
    vels      = cp_params[:,1]              ## m/s,  set to 0 to get pure boiling
    if nofroflo: 
        vels = vels * 0.0
    dirs      = cp_params[:,2] * np.pi/180. ## in radians

    ## decompose velocities into components
    vels_x    = vels * np.cos(dirs)
    vels_y    = vels * np.sin(dirs)
    
    # generate spatial frequency grids
    screensize_meters = bign * pscale
    deltaf = 1./screensize_meters           ## spatial frequency delta
    fx, fy = gg.generate_grids(bign, scalefac=deltaf, freqshift=True)

    phase = np.zeros((bign,bign,n_layers,timesteps),dtype=float)
    # the method only needs phaseFT from the previous timestep so this is unnecessary if memory
    # constraints exist - just save FT of the phase for the current timestep and update with the new
    # FT instead. This array is useful and saves time for PSD calculations
    phFT  = np.zeros((bign,bign,n_layers,timesteps),dtype=complex)  ## array for FT of phase
    #phrms = np.zeros((timesteps, n_layers),dtype=float)            ## phase rms at each timestep
    #phvar = np.zeros((timesteps, n_layers),dtype=float)            ## phase variance at each timestep

    for i in np.arange(n_layers):
        # Set the noise scaling powerlaw - the powerlaw below if from Johansson & Gavel 1994 for a 
        # Kolmogorov screen
        powerlaw = (2*np.pi*np.sqrt(0.00058)*(r0s[i]**(-5.0/6.0))*
                    (fx**2 + fy**2)**(-11.0/12.0)*bign*np.sqrt(np.sqrt(2.))/screensize_meters)
        powerlaw[0,0] = 0.0
        ## make array for the alpha parameter and populate it
        alpha_phase = - 2 * np.pi * (fx*vels_x[i] + fy*vels_y[i]) / rate

        # alpha magnitude can be different for each layer (and each Fourier mode) - to be
        # implemented later
        #if type(alpha_mag) == list:
        #    alpha = alpha_mag[i] * (np.cos(alpha_phase) + 1j * np.sin(alpha_phase))
        #else:
        #    alpha = alpha_mag * (np.cos(alpha_phase) + 1j * np.sin(alpha_phase)) 
        alpha = alpha_mag * (np.cos(alpha_phase) + 1j * np.sin(alpha_phase))

        noisescalefac = np.sqrt(1 - (np.abs(alpha))**2)
        print 'Layer', str(i+1), 'alpha created'
      
        for t in np.arange(timesteps):
             # generate noise to be added in, FT it and scale by powerlaw
             noise = np.random.randn(bign,bign)

             ## no added noise yet, start with a regular phase screen
             noiseFT = sf.fft2(noise) * powerlaw

             if t == 0:
                 wfFT = noiseFT
                 phFT[:,:,i,t] = noiseFT
             else:      
             # autoregression AR(1)
             # the new wavefront = alpha * wfnow + noise
                 wfFT = alpha * phFT[:,:,i,t-1] + noiseFT * noisescalefac
                 phFT[:,:,i,t] = wfFT
            
             # the new phase is the real_part of the inverse FT of the above
             wf = sf.ifft2(wfFT).real
             #phrms[t,i] = rms(wf)
             #phvar[t,i] = np.var(wf)
 
             # impose aperture, depiston, detilt, if desired
             if dept:
                 import depiston as dp
                 import detilt as dt
                 phase[:,:,i,t] = depiston(detilt(wf,aperture),aperture)*aperture
             else:
                 phase[:,:,i,t] = wf
        
        print 'Writing layer', str(i+1), 'file'  
        phase[:,:,i,:].shape 
        hdu = pf.PrimaryHDU(phase[:,:,i,:].transpose())
        hdu.writeto(layerfileroot+str(i+1)+'.fits', clobber=True)
        print 'Done with Layer', str(i+1)

    phaseout = np.sum(phase, axis=2)  # sum along layer axis
    hdu = pf.PrimaryHDU(phaseout.transpose())
    hdu.writeto(arfileroot+'.fits', clobber=True)
    print 'Done'
    #plot, phvar[*,0], lines=0#, /ylog, yrange=[0.1,3000]
    #oplot, phrms[*,0], lines=n_layers
    #for i = 1, n_layers-1 do begin
      #oplot, phvar[*,i], lines=i
#      oplot, phrms[*,n], lines=n+n_layers


#if keyword_set(savefileflag) then begin
#   if keyword_set(hdf5flag) then begin   
#######*******************************************************
#######        set up header for the HDF5 files
#######*******************************************************
#
#
#      hdf5hdr = {TELINST  : 'GPI', $
#                 PRI_DIA  : bigD, $ # 'Telescope primary diameter'
#                 SEC_DIA  : bigDs,$ # 'Telescope secondary diameter'
#                 COM0     : 'AR Atmosphere parameters follow', $
#                 EXPTIME  : exptime, $    # 'Exposure time'
#                 RATE     : rate, $       # 'System rate in Hz'
#                 ALPHAMAG : alpha_mag,$   # 'Magnitude of alpha vector'
#                 N_TSTEPS : timesteps,$   # 'Number of timesteps in cube'
#                 N_SUBAPS : n, $          # 'Width of screen in number of subaps'
#                 PIXSCALE : pscale, $     # 'Pixel scale in meters/pixel'
#                 N_ACROSS : nacross, $    # 'Number of subaps across aperture'
#                 SA_NPIX  :  m, $         # 'Number of pixels per subap'
#                 SA_DIA   :  d, $         # 'Subaperture diameter'
#                 COM1     : 'wind layer parameters follow', $
#                 N_LAYERS : n_layers } #, $  # 'Number of wind layers'
#                 #ARPHASE  : phase }
#
#      if n_layers gt 1 then begin
#         for i = 0, n_layers-1 do begin
#            struct_add_field, hdf5hdr, 'LAYER'+trim(i), 'Layer number'
#            struct_add_field, hdf5hdr, 'R0'+trim(i),  r0s[i] #'r0 for layer '+trim(i)
#            struct_add_field, hdf5hdr, 'VEL'+trim(i), vels[i] #'wind velocity (m/s) for layer i 
#            struct_add_field, hdf5hdr, 'DIR'+trim(i), dirs[i] #'wind direction (radians) for layer i
#         endfor
#         fname  = arfileroot+'-data.h5'
#         faname = arfileroot+'-attr.h5'
#         # ftname = arfileroot+'-phFT.h5'
#         phase  = reform(total(phase,3)) 
#      endif else begin
#         struct_add_field, hdf5hdr, 'LAYER', 0    # 'Layer number'
#         struct_add_field, hdf5hdr, 'R00',  r0s[0] # 'r0 for layer 0'
#         struct_add_field, hdf5hdr, 'VEL0', vels[0] # 'wind velocity (m/s) for layer 0'
#         struct_add_field, hdf5hdr, 'DIR0', dirs[0] # 'wind direction (radians) for layer 0'
#         fname  = onelfileroot+'-data.h5'
#         faname = onelfileroot+'-attr.h5'
#         #ftname = onelfileroot+'-phFT.h5'
#      endelse
#
#      fid = H5F_CREATE(faname)
#      datatype_id  = H5T_IDL_CREATE(hdf5hdr)
#      dataspace_id = H5S_CREATE_SIMPLE(1)
#      dataset_id   = H5D_CREATE(fid,'AR Phase attributes',datatype_id,dataspace_id)
#      H5D_WRITE, dataset_id, hdf5hdr
#      print, 'HDF5 attr file saved: ', faname
#      H5D_CLOSE,dataset_id
#      H5S_CLOSE,dataspace_id
#      H5T_CLOSE,datatype_id
#      H5F_CLOSE,fid   
#
#      fid = H5F_CREATE(fname)
#      datatype_id  = H5T_IDL_CREATE(phase)
#      dataspace_id = H5S_CREATE_SIMPLE(size(phase,/DIMENSIONS))
#      dataset_id   = H5D_CREATE(fid,'AR Phase data',datatype_id,dataspace_id)
#      H5D_WRITE, dataset_id, phase
#      print, 'HDF5 data file saved: ', fname
#      ## close all open identifiers
#      H5D_CLOSE,dataset_id
#      H5S_CLOSE,dataspace_id
#      H5T_CLOSE,datatype_id
#      H5F_CLOSE,fid   
#
#   endif else begin
#######*******************************************************
#######        set up header for the FITS files
#######*******************************************************
#      sxaddpar, arhdr, 'TELINST', 'GPI',      'Telescope and Instrument'
#      sxaddpar, arhdr, 'PRI_DIA',   bigD,     'Telescope primary diameter'
#      sxaddpar, arhdr, 'SEC_DIA',   bigDs,    'Telescope secondary diameter'
#      sxaddpar, arhdr, 'COM0', 'AR Atmosphere parameters follow'
#      sxaddpar, arhdr, 'EXPTIME',  exptime,   'Exposure time'
#      sxaddpar, arhdr, 'RATE',     rate,      'System rate in Hz'
#      sxaddpar, arhdr, 'ALPHAMAG', alpha_mag, 'Magnitude of alpha vector'
#      sxaddpar, arhdr, 'N_TSTEPS', timesteps, 'Number of timesteps in cube'
#      sxaddpar, arhdr, 'N_SUBAPS', n,         'Width of screen in number of subaps'
#      sxaddpar, arhdr, 'PIXSCALE', pscale,    'Pixel scale in meters/pixel'
#      sxaddpar, arhdr, 'N_ACROSS', nacross,   'Number of subaps across aperture'
#      sxaddpar, arhdr, 'SA_NPIX',  m,         'Number of pixels per subap'
#      sxaddpar, arhdr, 'SA_DIA',   d,         'Subaperture diameter'
#      sxaddpar, arhdr, 'COM1', 'wind layer parameters follow'
#      sxaddpar, arhdr, 'N_LAYERS', n_layers,  'Number of wind layers'
#
#      if n_layers gt 1 then begin
#         for i = 0, n_layers-1 do begin
#            sxaddpar, arhdr, 'LAYER'+trim(i), 'Layer number'
#            sxaddpar, arhdr, 'R0'+trim(i),  r0s[i],  'r0 for layer '+trim(i)
#            sxaddpar, arhdr, 'VEL'+trim(i), vels[i], 'wind velocity (m/s) for layer '+trim(i)
#            sxaddpar, arhdr, 'DIR'+trim(i), dirs[i], 'wind direction (radians) for layer '+trim(i)
#         endfor
#         fname  = arfileroot+'.fits'
#         ftname = arfileroot+'-phFT.fits'
#         phase  = reform(total(phase,3)) 
#      endif else begin
#         sxaddpar, arhdr, 'LAYER', 0, 'Layer number'
#         sxaddpar, arhdr, 'R00',  r0s[0],  'r0 for layer 0'
#         sxaddpar, arhdr, 'VEL0', vels[0], 'wind velocity (m/s) for layer 0'
#         sxaddpar, arhdr, 'DIR0', dirs[0], 'wind direction (radians) for layer 0'
#         fname  = onelfileroot+'.fits'
#         ftname = onelfileroot+'-phFT.fits'
#      endelse
#
#      mwrfits, phase, fname, arhdr, /create
#      print, 'FITS movie file saved: ', fname
#      #mwrfits, phFT, ftname, arhdr, /create
#      #print, 'phase FT file saved: ', ftname
#   endelse
#endif
#
#
#   if keyword_set(stopflag) then stop
#
#   if keyword_set(analyzeflag) then begin
#      print, 'Starting analysis...'
#      phapFT = make_array(bign,bign,timesteps,/comp) # FT of apertured phase 
#      phap   = make_array(bign,bign,timesteps,/float) # phase with aperture imposed, depistoned
#      phdt   = make_array(bign,bign,timesteps,/float) # above detilted
#
#      phrms  = make_array(timesteps, /float)
#      sigFT  = make_array(timesteps, /float) 
#      meanFT = make_array(timesteps, /comp)
#      sigap  = make_array(timesteps, /float) # std dev of phap at each timestep
#      r0     = make_array(timesteps, /float) # r0 calculation from sigap
#      sigdt  = make_array(timesteps, /float) # stddev of phdt at each timestep
#
#   endif
#
#  
#   if keyword_set(stopflag) then stop
#end

