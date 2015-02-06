import numpy as np

def detilt(phase, aperture):
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
    
    nx = aperture.shape[1]  # number of columns
    ny = aperture.shape[0]  # number of rows
    
    a = np.arange(float(nx))
    b = np.arange(float(ny))
    
    xind = np.vstack((a,)*ny)
    yind = np.transpose(np.vstack((b,)*nx))
 
    xind = aperture*(xind - np.sum(xind*aperture)/aperture.sum())
    yind = aperture*(yind - np.sum(yind*aperture)/aperture.sum())   
    
    phdt = aperture*(phase - xind*np.sum(phase*xind)/np.sum(xind**2))
    phdt = aperture*(phase - yind*np.sum(phase*yind)/np.sum(yind**2))
    return phdt
    