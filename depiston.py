import numpy as np

def depiston(phase, aperture=np.zeros(1)):
    """
    ;  depiston - remove piston over an aperture
    ;
    ;  USAGE:
    ;    phdp = depiston(ph,ap)
    ;
    ;  INPUTS:
    ;    ph - 2D numpy array of phase [n,m]
    ;    ap - numpy array defining aperture[n,m] - optional
    ;
    ;  OUTPUTS:
    ;    phdp - phase with piston removed
    """
        
    if len(aperture) == 1: 
        aperture = np.ones(phase.shape)
          
    piston = np.sum(phase*aperture)/aperture.sum()
    phdp   = aperture*(phase - piston)

    return phdp
      