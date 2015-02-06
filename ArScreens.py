import numpy as np
import pyfits
from create_multilayer_arbase import create_multilayer_arbase
from get_ar_atmos import get_ar_atmos

class ArScreens(object):
    def __init__(self, n, m, pscale, rate, alpha_mag, paramcube):
        self.pl, self.alpha = create_multilayer_arbase(n, m, pscale, rate,
                                                       paramcube, alpha_mag)
        self._phaseFT = None
        self.screens = [[] for x in paramcube]
    def run(self, nframes, verbose=False):
        for j in range(nframes):
            if verbose:
                print "time step", j
            self._phaseFT, screens = get_ar_atmos(self._phaseFT, self.pl,
                                                  self.alpha,
                                                  first=(self._phaseFT is None))
            for i, item in enumerate(screens):
                self.screens[i].append(item)
    def write(self, outfile, clobber=True):
        output = pyfits.HDUList()
        output.append(pyfits.PrimaryHDU())
        for i, screen in enumerate(self.screens):
            output.append(pyfits.ImageHDU(np.array(screen)))
            output[-1].name = "Layer %i" % i
        output.writeto(outfile, clobber=clobber)

if __name__ == '__main__':
    n = 48
    m = 8
    bigD = 8.4
    pscale = bigD/(n*m)
    rate = 1000.
    alpha_mag = 0.99
    paramcube = np.array([(0.85, 23.2, 259, 7600),
                          (1.08, 5.7, 320, 16000)])

    my_screens = ArScreens(n, m, pscale, rate, alpha_mag, paramcube)
    my_screens.run(100, verbose=True)
    my_screens.write('my_screens_0.999.fits')
    
