# -*- coding: utf-8 -*-
"""
Part of the pyconvolve framework for convolution and deconvolution. 
Author: Lukas Küpper, 2018
License: GPLv3
"""
import numpy as np

from decon.abstract_decon import AbstractDecon as __DA


########################################################################
class InverseFilter(__DA):
    """
    Calculates the Deconvolution of a given set of Image and Point Spread Function according to the simple inverse Filtering. Very unstable and succeptable to noise.
    How to use:
    - Initialize the Object. e.g. decon = InverseFilter(image_array, psf_array)
    - call the solve method. decon.solve()
    - the solution is stored in the out member of the object. decon.out is an array of the same shape as the input image and of dtype=float
    
    https://en.wikipedia.org/wiki/Deconvolution
    
    U = D / P
    
    D is the Fourier Transform of the observed image
    U is the Fourier Transform of the estimated original image
    P is the Fourier Transform of the PSF
    
    
    """ 
    #----------------------------------------------------------------------
    def __init__(self, img, 
                 psf = None, 
                 sample = None, 
                 groundTruth = None,
                 solveFor = 'sample',
                 isPsfCentered = True,
                 cutoff = 0.001, 
                 relativeCutoff = True,
                 cutoffInFourierDomain = False,
                 constraints = None,
                 useCpxFFT = False,
                 debugInt = 0,
                 compareWithTruth = False):
        """
        Parameters:
        
        - img: image array, will be converted (with numpy.astype('float')) to 'float' array
        - psf: array containing the point spread function. Does not have to be the same shape as the input image. PSF will be zero-padded. Will also be converted to float.
        - useCpxFFT: bool to decide if the real-valued FFTs of numpy.fft or the complex-valued ones should be used
        - psf_cutoff: cutoff value to avoid zero division in the reconstruction. Values in the frequency domain below the cutoff are ignored in the division. 
        - isPSFCutoffRelative: denotes whether the psf_cutoff value is in units of the maximum value of the psf (in freq domain)
        """
        
        super(InverseFilter, self).__init__(img, psf, sample, groundTruth, 'InverseFilter', solveFor, constraints, isPsfCentered, useCpxFFT, debugInt, compareWithTruth)
        
        self.cutoff = cutoff
        self.relCutoff = relativeCutoff        
        self.cutoffFourier = cutoffInFourierDomain
        
    #----------------------------------------------------------------------
    def getAlgoParameters(self):
        """Function has to be implemented by all reconstruction algorithms."""
        return {'isRelativeCutoff':self.relCutoff, 'cutOffInFourierDomain':self.cutoffFourier, 'cutoff':self.cutoff}
        
    #----------------------------------------------------------------------
    def _prepare(self):
        """"""
        self.matchArrayShapes()

        
        if self.solveFor == 'sample':
            if not self.cutoffFourier:
                if self.relCutoff:
                    cut = self.cutoff * self.psf.max()
                else:
                    cut = self.cutoff
                self.psf[self.psf < cut] = cut
                self.print_dbg('Applying Cutoff on PSF in Real Space...',3)
            
            if self.useCpxFFT:           
                self.f_img = np.fft.fftn(self.img)
                if self.isPsfCentered:
                    self.f_psf = np.fft.fftn(np.fft.ifftshift(self.psf))
                else:
                    self.f_psf = np.fft.fftn(self.psf)
            else:           
                self.f_img = np.fft.rfftn(self.img)
                if self.isPsfCentered:
                    self.f_psf = np.fft.rfftn(np.fft.ifftshift(self.psf))
                else:  
                    self.f_psf = np.fft.rfftn(self.psf)
                    
            if self.cutoffFourier:
                if self.relCutoff:
                    cut = self.cutoff * self.f_psf.max()
                else:
                    cut = self.cutoff
                self.f_psf[self.f_psf < cut] = cut
                self.print_dbg('Applying Cutoff on PSF in Fourier Space...',3)
                    
        elif self.solveFor == 'psf':
            if not self.cutoffFourier:
                if self.relCutoff:
                    cut = self.cutoff * self.sample.max()
                else:
                    cut = self.cutoff
                self.sample[self.sample < cut] = cut          
                self.print_dbg('Applying Cutoff on Sample in Real Space...',3)
            
            if self.useCpxFFT:
                self.f_img = np.fft.fftn(self.img)
                self.f_sample = np.fft.fftn(self.sample)
            else:
                self.f_img = np.fft.rfftn(self.img)
                self.f_sample = np.fft.rfftn(self.sample)
                
            if self.cutoffFourier:
                if self.relCutoff:
                    cut = self.cutoff*self.f_sample.max()
                else:
                    cut = self.cutoff
                self.f_sample[self.f_sample < cut] = cut
                self.print_dbg('Applying Cutoff on Sample in Fourier Space...',3)
            
        else:
            print("'solveFor' has to be either 'psf' or 'sample'. ")
            raise ValueError
        
        
        pass
        
    #----------------------------------------------------------------------
    def _solve(self):
        """"""
        
        if self.solveFor == 'sample':
            self.print_dbg('Calculating Fourier-Space Sample representation...', 3)
            
            eps = self.f_psf*self.f_psf.conj()
            eps[eps < 1e-6] = 1e-6
            self.f_sample = ((self.f_img.real * self.f_psf.real + self.f_img.imag*self.f_psf.imag) + 1j*(self.f_img.imag*self.f_psf.real - self.f_img.real*self.f_psf.imag))/eps
            
            if self.useCpxFFT:
                self.print_dbg('Done. Doing Inverse Fourier Transformr (Using Complex)...' ,3)
                self.sample = np.fft.ifftn(self.f_sample).real
            else:
                self.print_dbg('Done. Doing Inverse Fourier Transformr (Using Real)...' ,3)
                self.sample = np.fft.irfftn(self.f_sample, self.min_mut_shape)
            self.print_dbg('Done. Applying Constraints and freeing memory.', 3)
            self.sample[self.sample < 0] = 0.
            self.sample[self.sample > self.img.max()] = self.img.max()
        else:
            eps = self.f_sample*self.f_sample.conj()
            eps[eps < 1e-6] = 1e-6       
            
            self.f_psf = ((self.f_img.real * self.f_sample.real + self.f_img.imag*self.f_sample.imag) + 1j*(self.f_img.imag*self.f_sample.real - self.f_img.real*self.f_sample.imag))/eps            
            
            if self.useCpxFFT:
                self.psf = np.fft.ifftn(self.f_psf).real
            else:
                self.psf = np.fft.irfftn(self.f_psf, self.min_mut_shape)
            self.psf[self.psf < 0] = 0.
            self.psf[self.psf > self.img.max()] = self.img.max()
        self.applyConstraint()
        self.f_img = None
        self.f_psf = None
        self.f_sample = None
        if self.compareWithTruth:
            self.print_dbg('Done. Comparing result with ground truth.', 3)
            self.curError = self.calcError()
            if self.solveFor == 'sample':
                self.diff = self.groundTruth - self.sample
            else:
                self.diff = self.groundTruth - self.psf
                
        self.print_dbg('Done. Exiting _solve()...', 3)
    
    



########################################################################
class WienerFilter(__DA):
    """
    Calculates the Deconvolution of a given set of Image and Point Spread Function according to the Wiener Filtering Deconvolution Algorithm.
    How to use:
    - Initialize the Object. e.g. decon = WienerFilter(image_array, psf_array)
    - call the solve method. decon.solve()
    - the solution is stored in the out member of the object. decon.out is an array of the same shape as the input image and of dtype=float
    
    https://en.wikipedia.org/wiki/Wiener_deconvolution
    
    U = D P*/(P P* + SNR) 
    
    D is the Fourier Transform of the observed image
    U is the Fourier Transform of the estimated original image
    P is the Fourier Transform of the PSF
    P* is the complex conjugate of P
    SNR is the Signal to Noise Ratio of the Image. 
    
    The Algorithm stores in decon.out the Inverse Fourier Transform of U (the estimate in real space)
    """

    #----------------------------------------------------------------------
    def __init__(self,
                 img,
                 psf = None,
                 sample = None,
                 groundTruth = None,
                 solveFor = 'sample',
                 isPsfCentered = True,
                 noise = 0.001,
                 relativeNoise = True,
                 constraints = None,
                 useCpxFFT = False,
                 debugInt = 0,
                 compareWithTruth = False):
        """
        Parameters:
        
        - img: image array, will be converted (with numpy.astype('float')) to 'float' array
        - psf: array containing the point spread function. Does not have to be the same shape as the input image. PSF will be zero-padded. Will also be converted to float.
        - useCpxFFT: bool to decide if the real-valued FFTs of numpy.fft or the complex-valued ones should be used
        - cutoff_noise: Tuning parameter. Corresponding to the inverse SNR of the image.
        - noiseRelative: Annotes if the given cutoff_noise is in relative terms to the maximum Value in the Image Array
        
        """
        super(WienerFilter, self).__init__(img, psf, sample, groundTruth, 'WienerFilter', solveFor, constraints, isPsfCentered, useCpxFFT, debugInt, compareWithTruth)
        
        self.noise = noise
        self.relNoise = relativeNoise
    
    #----------------------------------------------------------------------
    def getAlgoParameters(self):
        """Function has to be implemented by all reconstruction algorithms."""
        return {'NoiseParameter':self.noise, 'NoiseParameterIsRelative':self.relNoise}
    #----------------------------------------------------------------------
    def _prepare(self):
        """"""
        self.matchArrayShapes()
        
        if self.solveFor == 'sample':
            
            if self.useCpxFFT:
                if self.isPsfCentered: f_psf = np.fft.fftn(np.fft.ifftshift(self.psf))
                else: f_psf = np.fft.fftn(self.psf)
            else:
                if self.isPsfCentered: f_psf = np.fft.rfftn(np.fft.ifftshift(self.psf))
                else: f_psf = np.fft.rfftn(self.psf)
                    
            if self.relNoise: cutoff = self.psf.max()/self.noise
            else: cutoff = self.noise
            
            #Bug Fix Addendum
            f_psf = np.fft.fftshift(f_psf)
            
            self.f_filter = f_psf.conj()/(f_psf.conj()*f_psf + 1./cutoff)
                    
        elif self.solveFor == 'psf':
            
            if self.useCpxFFT:
                f_sample = np.fft.fftn(self.sample)
            else:
                f_sample = np.fft.rfftn(self.sample)
            if self.relNoise: cutoff = self.psf.max()/self.noise
            else: cutoff = self.noise
            
            #Bug Fix Addendum
            f_sample = np.fft.fftshift(f_sample)
            
            self.f_filter = f_sample.conj()/(f_sample.conj()*f_sample + 1./cutoff)
        
        else:
            print("'solveFor' has to be either 'psf' or 'sample'. ")
            raise ValueError
  
  
    #----------------------------------------------------------------------
    def _solve(self):
        """"""
        
        if self.solveFor == 'sample':
            if self.useCpxFFT:
                #Addendum of shift
                f_img = np.fft.fftshift(np.fft.fftn(self.img))
                f_sample = f_img * self.f_filter
                #Addendum of shift
                self.sample = np.fft.ifftn(np.fft.ifftshift(f_sample)).real
            else:
                #Addendum of shift
                f_img = np.fft.fftshift(np.fft.rfftn(self.img))
                f_sample = f_img * self.f_filter
                #Addendum of shift                
                self.sample = np.fft.irfftn(np.fft.ifftshift(f_sample), self.min_mut_shape)
        else:
            if self.useCpxFFT:
                f_img = np.fft.fftn(self.img)
                f_psf = f_img * self.f_filter
                self.psf = np.fft.fftshift(np.fft.ifftn(f_psf).real)
            else:
                f_img = np.fft.rfftn(self.img)
                f_psf = f_img * self.f_filter
                self.psf = np.fft.fftshift(np.fft.irfftn(f_psf, self.min_mut_shape))
                
        self.applyConstraint()
        self.f_filter = None
        
        if self.compareWithTruth:
            self.curError = self.calcError()
            if self.solveFor == 'sample':
                self.diff = self.groundTruth - self.sample
            else:
                self.diff = self.groundTruth - self.psf        
        
        
        
        

        
    
########################################################################
class RegularizedTikhonov(__DA):
    """"""

    #----------------------------------------------------------------------
    def __init__(self,
                 img,
                 psf = None,
                 sample = None,
                 groundTruth = None,
                 solveFor = 'sample',
                 isPsfCentered = True,
                 lam = 0.001,
                 tolerance = 1e-6,
                 constraints = None,
                 useCpxFFT = False,
                 debugInt = 0,
                 compareWithTruth = False):
        """
        Parameters:
        
        - img: image array, will be converted (with numpy.astype('float')) to 'float' array
        - psf: array containing the point spread function. Does not have to be the same shape as the input image. PSF will be zero-padded. Will also be converted to float.
        - useCpxFFT: bool to decide if the real-valued FFTs of numpy.fft or the complex-valued ones should be used
        - cutoff_noise: Tuning parameter. Corresponding to the inverse SNR of the image.
        - noiseRelative: Annotes if the given cutoff_noise is in relative terms to the maximum Value in the Image Array
        
        """
        super(RegularizedTikhonov, self).__init__(img, psf, sample, groundTruth, 'RegularizedTikhonov', solveFor, constraints, isPsfCentered, useCpxFFT, debugInt, compareWithTruth)
                
        self.lam = lam
        self.toler = tolerance
        
        
    #----------------------------------------------------------------------
    def getAlgoParameters(self):
        """Function has to be implemented by all reconstruction algorithms."""
        return {'Lambda':self.lam, 'DivisionTolerance':self.toler}
        
    #----------------------------------------------------------------------
    def _prepare(self):
        """"""
        self.matchArrayShapes()
        
        if self.solveFor == 'sample':
            
            if self.useCpxFFT:
                if self.isPsfCentered: f_psf = np.fft.fftn(np.fft.ifftshift(self.psf))
                else: f_psf = np.fft.fftn(self.psf)
                
                
            else:
                if self.isPsfCentered: f_psf = np.fft.rfftn(np.fft.ifftshift(self.psf))
                else: f_psf = np.fft.rfftn(self.psf)
                
                
            f_temp = f_psf*f_psf + self.lam
            self.f_filt = np.zeros(f_temp.shape, dtype = f_temp.dtype)
            
            denom = np.power(f_temp.real,2) + np.power(f_temp.imag,2)
            denom[denom < self.toler] = self.toler
            
            self.f_filt.real = (f_psf.real * f_temp.real + f_psf.imag * f_temp.imag)/(denom)
            self.f_filt.imag = (f_psf.imag * f_temp.real - f_psf.real * f_temp.imag)/(denom)
                    
                    
        elif self.solveFor == 'psf':
            
            if self.useCpxFFT:
                f_sample = np.fft.fftn(self.sample)
                
            else:
                f_sample = np.fft.rfftn(self.sample)
                
            
            f_temp = f_sample*f_sample + self.lam
            self.f_filt = np.zeros(f_temp.shape, dtype = f_temp.dtype)
            
            denom = np.power(f_temp.real,2) + np.power(f_temp.imag,2)
            denom[denom < self.toler] = self.toler            
            
            self.f_filt.real = (f_psf.real * f_temp.real + f_psf.imag * f_temp.imag)/(denom)
            self.f_filt.imag = (f_psf.imag * f_temp.real - f_psf.real * f_temp.imag)/(denom)            
        
        else:
            print("'solveFor' has to be either 'psf' or 'sample'. ")
            raise ValueError        
        
        
    #----------------------------------------------------------------------
    def _solve(self):
        """"""
        
        if self.solveFor == 'sample':
            if self.useCpxFFT:
                f_img = np.fft.fftn(self.img)
                f_sample = f_img * self.f_filt
                self.sample = np.fft.ifftn(f_sample).real
            else:
                f_img = np.fft.rfftn(self.img)
                f_sample = f_img * self.f_filt
                self.sample = np.fft.irfftn(f_sample, self.min_mut_shape)
        else:
            if self.useCpxFFT:
                f_img = np.fft.fftn(self.img)
                f_psf = f_img * self.f_filt
                self.psf = np.fft.fftshift(np.fft.ifftn(f_psf).real)
            else:
                f_img = np.fft.rfftn(self.img)
                f_psf = f_img * self.f_filt
                self.psf = np.fft.fftshift(np.fft.irfftn(f_psf, self.min_mut_shape))
                
        self.applyConstraint()
        self.f_filt = None
        
        if self.compareWithTruth:
            self.curError = self.calcError()
            if self.solveFor == 'sample':
                self.diff = self.groundTruth - self.sample
            else:
                self.diff = self.groundTruth - self.psf   
                
                
                
########################################################################
class SingleStep(object):
    """"""

    ALGORITHMS = ['WienerFilter', 'InverseFilter', 'RegularizedTikhonov']

    #----------------------------------------------------------------------
    def __init__(self, img, 
                 psf = None, 
                 sample = None, 
                 groundTruth = None, 
                 name = 'WienerFilter', 
                 solveFor = 'sample', 
                 constraints = None, 
                 isPsfCentered = True, 
                 useCpxFFT = False, 
                 debugInt = 0, 
                 compareWithTruth = False, 
                 **kwargs):
        """Constructor"""
        
        
        if not name in SingleStep.ALGORITHMS:
            raise ValueError
        elif name == SingleStep.ALGORITHMS[0]:
            #valid keyword arguments: noise [float], relativeNoise [bool]
            self.algo = WienerFilter(img = img, 
                                     psf = psf, 
                                     sample = sample, 
                                     groundTruth = groundTruth, 
                                     solveFor = solveFor, 
                                     isPsfCentered = isPsfCentered, 
                                     #noise = 0.001, 
                                     #relativeNoise = True, 
                                     constraints = constraints, 
                                     useCpxFFT = useCpxFFT, 
                                     debugInt = debugInt, 
                                     compareWithTruth = compareWithTruth,
                                     **kwargs)
        elif name == SingleStep.ALGORITHMS[1]:
            #valid keyword arguments: cutoff [float], relativeCutoff [bool], cutoffInFourierDomain [bool]
            self.algo = InverseFilter(img = img, 
                                      psf = psf, 
                                      sample = sample, 
                                      groundTruth = groundTruth, 
                                      solveFor = solveFor, 
                                      isPsfCentered = isPsfCentered, 
                                      constraints = constraints, 
                                      useCpxFFT = useCpxFFT, 
                                      debugInt = debugInt, 
                                      compareWithTruth = compareWithTruth,
                                      **kwargs)
        elif name == SingleStep.ALGORITHMS[2]:
            #valid keyword arguments: lam[float], tolerance[float]
            self.algo = RegularizedTikhonov(img=img, 
                                            psf=psf, 
                                            sample=sample, 
                                            groundTruth=groundTruth, 
                                            solveFor=solveFor, 
                                            isPsfCentered=isPsfCentered, 
                                            constraints=constraints, 
                                            useCpxFFT=useCpxFFT, 
                                            debugInt= debugInt, 
                                            compareWithTruth=compareWithTruth,
                                            **kwargs)
            
        self.out = None
        self.cunError = -1.
    
    #----------------------------------------------------------------------
    def prepare(self):
        """"""
        self.algo.prepare()
    
        
    #----------------------------------------------------------------------
    def solve(self):
        """"""
        self.algo.solve()
        self.out = self.algo.out
        
        if self.algo.compareWithTruth:
            self.curError = self.algo.curError
        
        
        
        
    
        
    
            
        
        
        
        
        
        
    
    