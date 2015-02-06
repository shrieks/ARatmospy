#;; Developed by Lisa A. Poyneer 2001-2008
#;; No warranty is expressed or implied.
#;; --------------------------------------------------------
#
#;; use this to make phasecube for our simulation!

import numpy as np
from get_phase_streamlined import get_phase_streamlined

def get_x_size(xsize0, max_size, ok_sizes, dir='X'):
    if xsize0 < max_size:
        locs = np.where(xsize0 < ok_sizes)
        cnt = len(locs[0])
        if cnt > 0:
          return ok_sizes[locs[0][0]]
    raise RuntimeError("create_multilayer_phasecube: %s size too big %s" 
                       % (dir, xsize0))

def create_multilayer_phasecube(n, m, pscale, time, paramcube, 
                                rflag=None, sizeflag=None):
    bign = n*m
    dims = paramcube.shape
    if dims[0] == 1:
        n_layers = 1
    else:
        n_layers = dims[2]

    r0s = paramcube[0]
    vels = paramcube[1]
    directions = paramcube[2]

    # figure out how big each screen needs to be
    vels_x = vels*np.cos(directions*np.pi/180.)
    vels_y = vels*np.sin(directions*np.pi/180.)

    distances_x = vels_x*time
    distances_y = vels_y*time

    pixels_x = distances_x/pscale
    pixels_y = distances_y/pscale

    xsize0 = np.floor(max(np.abs(pixels_x) + bign) + 1)*1.0
    ysize0 = np.floor(max(np.abs(pixels_y) + bign) + 1)*1.0

    # these are nice sizes, given FFTW
    powers_of_2 = np.array([64,128,256,512,1024,2048,4096], dtype=np.float)*1.0 
    an_extra_3 = 3.*np.array([32, 64, 128, 256, 512, 1024, 2048],dtype=np.float)
    two_extra_3s = 9.*np.array([16, 32, 64, 128, 256, 512], dtype=np.float)

    ok_sizes0 = np.concatenate([powers_of_2, an_extra_3, two_extra_3s])
    ok_sizes = np.sort(ok_sizes0)
    max_size = max(ok_sizes)

    xsize = get_x_size(xsize0, max_size, ok_sizes)
    ysize = get_x_size(xsize0, max_size, ok_sizes, dir='Y')


    mysize = max([xsize, ysize])
    required_minimum_size = bign*4.
    if mysize < required_minimum_size:
        mysize = require_minimum_size

    if sizeflag is not None:
        mysize = max(mysize, sizeflag)

    print '********** ', xsize0, ysize0, xsize, ysize, mysize, ' ************'
    effectiven = mysize/m

    # now make the phase screens!

    if rflag is None:
        rflag = 104812094812

    phasecube = []
    for j in range(n_layers):
        phasecube.append(get_phase_streamlined(effectiven, m, pscale, r0s[j], rflag))
    # Match expected array format in original IDL code:
    #phasecube = make_array(mysize, mysize, n_layers, double=dflag)
    #for j=0, n_layers-1 do begin
    #    phasecube[*,*,j] = get_phase_streamlined(effectiven, m, $
    #                                             pscale, r0s[j], rflag, double=dflag, nofftw=nofftwflag)
    #endfor

    phasecube = np.array(phasecube).transpose(2, 0, 1)

    return phasecube
