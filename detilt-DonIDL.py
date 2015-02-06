import numpy as np

def detilt(phase, aperture=np.zeros(1)):
    """
    ;  detilt - remove tilt over an aperture
    ;
    ;  USAGE:
    ;    phdt = detilt(ph,ap)
    ;
    ;  INPUTS:
    ;    ph - phase    - 2D numpy array
    ;    ap - aperture - optional 2D numpy array
    ;
    ;  OUTPUTS:
    ;    phdt - phase with tilt removed
    ;    tx, ty - (optional) tip and tilt coefficients (units: phase/pixel)
    ;
    """
    
    n = np.float(phase.shape[0])  # number of rows
    m = np.float(phase.shape[1])  # number of columns
    if len(aperture) == 1: 
        aperture = np.ones(phase.shape)

    # x = ((findgen(n) # (fltarr(m)+1)) - n/2) in IDL, 
    # where n = n_columns and m = n_rows
    x = np.multiply.outer(np.arange(m),np.ones(n)).T - np.int(m)/2
    x = x - np.sum(x*aperture)/aperture.sum()
    # y = (((fltarr(n)+1) #   findgen(m)) - m/2) 
    y = np.multiply.outer(np.ones(m),np.arange(n)).T - np.int(n)/2
    y = y - np.sum(y*aperture)/aperture.sum()

    tx = np.sum(phase*x*aperture)/np.sum(x*x*aperture)
    ty = np.sum(phase*y*aperture)/np.sum(y*y*aperture)

    phdt = (phase - tx*x - ty*y)
    return phdt
    