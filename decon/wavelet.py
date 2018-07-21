# -*- coding: utf-8 -*-
"""
Part of the pyconvolve framework for convolution and deconvolution. 
Author: Lukas Küpper, 2018
License: GPLv3
"""
import numpy as np
import time
import math as mt


from decon.abstract_decon import AbstractDecon as __AD
from decon.iterative import AbstractIterative as __AI
import util.stack_loader



########################################################################
class AbstractWaveletDecon(__AI):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, 
                 image, 
                 psf = None, 
                 sample = None, 
                 groundTruth = None, 
                 solveFor = 'sample', 
                 initialGuess = 'orig_array', 
                 iterSteps = 1000, 
                 depth = 3,
                 threshold = 1.,
                 algoName = 'AbstractWavelet',
                 errTol = 1e-5, 
                 constraints = None, 
                 isPsfCentered = True, 
                 useCpxFFT = False, 
                 debugInt = 0, 
                 compareWithTruth = False,
                 saveIntermediateSteps = 0):  
        """Constructor"""
        
      
        
        
        super(AbstractWaveletDecon, self).__init__(image, psf, sample, groundTruth, solveFor, initialGuess, algoName, 
                                                   iterSteps, errTol, constraints, isPsfCentered, useCpxFFT, debugInt, compareWithTruth, 
                                                   saveIntermediateSteps)	    
        self.depth = depth
        self.thresh = threshold
    
    
    #----------------------------------------------------------------------
    def analysis(self, in_arr):
        """"""
        
        orig_shape = in_arr.shape        
        out_arr = in_arr.copy()
        
        #new_shape = [ni if ni % 2 == 0 else ni+1 for ni in orig_shape]
        
        new_shape = [ni if ni % 8 == 0 else ni + (8 - ni % 8) for ni in orig_shape] 
        
        out_arr = np.zeros(new_shape)
        out_arr[:orig_shape[0],:orig_shape[1],:orig_shape[2]] = in_arr[:,:,:].copy()
        
        for i in range(self.depth):
            t_arr = out_arr[:new_shape[0],:new_shape[1],:new_shape[2]].copy()
            
            t_arr = self._analysis_sub(t_arr)
            
            out_arr[:new_shape[0],:new_shape[1],:new_shape[2]] = t_arr.copy()
            new_shape = [max(1, int(nx/2)) for nx in new_shape]
            
            
        return out_arr            
    
        
    
    #----------------------------------------------------------------------
    def _analysis_sub(self, in_arr):
        """"""
        
        out_arr = in_arr.copy()
        temp_arr = np.zeros(in_arr.shape)
        shp_2 = [int(ni/2) if ni % 2 == 0 else int(ni/2-1) for ni in in_arr.shape]
        sqrt2 = mt.sqrt(2)
    
        
        for ix in range(shp_2[0]):
            temp_arr[ix,:,:] = (out_arr[2*ix,:,:] + out_arr[2*ix+1,:,:]) / sqrt2
            temp_arr[ix+shp_2[0],:,:] = (out_arr[2*ix,:,:] - out_arr[2*ix+1,:,:]) / sqrt2
        out_arr = temp_arr.copy()
        
        for iy in range(shp_2[1]):
            temp_arr[:,iy,:] = (out_arr[:,2*iy, :] + out_arr[:,2*iy+1,:]) / sqrt2
            temp_arr[:,iy+shp_2[1],:] = (out_arr[:,2*iy,:] - out_arr[:,2*iy+1,:]) /sqrt2
        out_arr = temp_arr.copy()
        
        for iz in range(shp_2[2]):
            temp_arr[:,:,iz] = (out_arr[:,:,2*iz] + out_arr[:,:,2*iz+1]) /sqrt2
            temp_arr[:,:,iz+shp_2[2]] = (out_arr[:,:,2*iz] - out_arr[:,:,2*iz+1]) /sqrt2
        out_arr = temp_arr.copy()
        
        
        return out_arr          
        
        
        
        
    #----------------------------------------------------------------------
    def synthesize(self, in_arr):
        """"""
        
        
        cur_shp = [max(1, ni/ 2**(self.depth-1)) for ni in in_arr.shape] 
        out_arr = in_arr.copy()
        
        for i in range(self.depth):
            t_arr = out_arr[:cur_shp[0],:cur_shp[1],:cur_shp[2]].copy()
            
            t_arr = self._synthesize_sub(t_arr)
            
            out_arr[:cur_shp[0], :cur_shp[1], :cur_shp[2]] = t_arr
            
            cur_shp = [ni*2 for ni in cur_shp]
            
          
        #if [s1+1 == s2 for s1,s2 in zip(out_arr.shape, self.curGuess.shape)]:
            #out_arr = out_arr[:-1,:-1,:-1]
            
        if not self.curGuess.shape[0] % 8 == 0:
            mod = 8 - self.curGuess.shape[0] % 8
            out_arr = out_arr[:-mod,:,:]
        if not self.curGuess.shape[1] % 8 == 0:
            mod = 8 - self.curGuess.shape[1] % 8            
            out_arr = out_arr[:,:-mod,:]
        if not self.curGuess.shape[2] % 8 == 0:
            mod = 8 - self.curGuess.shape[2] % 8
            out_arr = out_arr[:,:,:-mod]
        
        
        return out_arr          
        
    #----------------------------------------------------------------------
    def _synthesize_sub(self, in_arr):
        """"""
        
        out_arr = in_arr.copy()
        temp_arr = np.zeros(in_arr.shape)
        shp_2 = [int(ni/2) if ni % 2 == 0 else int(1+ni/2) for ni in in_arr.shape]
        sqrt2 = mt.sqrt(2)
    
        
        for ix in range(shp_2[0]):
            temp_arr[2*ix,:,:] = (out_arr[ix,:,:] + out_arr[ix + shp_2[0],:,:]) / sqrt2
            temp_arr[2*ix+1,:,:] = (out_arr[ix,:,:] - out_arr[ix + shp_2[0],:,:]) / sqrt2
        out_arr = temp_arr.copy()
        
        for iy in range(shp_2[1]):
            temp_arr[:,2*iy,:] = (out_arr[:,iy, :] + out_arr[:,iy+shp_2[1],:]) / sqrt2
            temp_arr[:,2*iy+1,:] = (out_arr[:,iy,:] - out_arr[:,iy+shp_2[1],:]) /sqrt2
        out_arr = temp_arr.copy()
        
        for iz in range(shp_2[2]):
            temp_arr[:,:,2*iz] = (out_arr[:,:,iz] + out_arr[:,:,iz+shp_2[2]]) /sqrt2
            temp_arr[:,:,2*iz+1] = (out_arr[:,:,iz] - out_arr[:,:,iz+shp_2[2]]) /sqrt2
        out_arr = temp_arr.copy()
        
        
        return out_arr
    
    
    #----------------------------------------------------------------------
    def threshold(self, inp_arr):
        """"""
        low = inp_arr <= -self.thresh
        high = inp_arr >= self.thresh
        between = np.logical_not(np.logical_or(low, high))
        inp_arr[low] -= -self.thresh
        inp_arr[high] -= self.thresh
        inp_arr[between] = 0.
        
        return inp_arr
    
    
    
    
########################################################################
class ISTA(AbstractWaveletDecon):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, 
                 image, 
                 psf = None, 
                 sample = None, 
                 groundTruth = None, 
                 solveFor = 'sample', 
                 initialGuess = 'orig_array', 
                 iterSteps = 1000, 
                 depth = 3,
                 gamma = 1.,
                 lamb = 1.,
                 errTol = 1e-5, 
                 constraints = None, 
                 isPsfCentered = True, 
                 useCpxFFT = False, 
                 debugInt = 0, 
                 compareWithTruth = False,
                 saveIntermediateSteps = 0):  
        """Constructor"""
        
        threshold = 0.5*gamma*lamb
        self.gamma = gamma
        self.lamb = lamb
        
        super(ISTA, self).__init__(image, psf, sample, groundTruth, solveFor, initialGuess, iterSteps, depth, threshold, 
                                                   'ISTA', errTol, constraints, isPsfCentered, useCpxFFT, debugInt, compareWithTruth, 
                                                   saveIntermediateSteps)
    
        
        
    #----------------------------------------------------------------------
    def _prepare(self):
        """"""
        
        if self.solveFor == 'sample':
            if self.isPsfCentered:
                temp = np.fft.ifftshift(self.psf)
            else:
                temp = self.psf
            if self.useCpxFFT:
                self.f_img = np.fft.fftn(self.img) * self.gamma
                f_compl = np.fft.fftn(temp)
            else:
                self.f_img = np.fft.rfftn(self.img) * self.gamma
                f_compl = np.fft.rfftn(temp)

        else:
            if self.useCpxFFT:
                self.f_img = np.fft.fftn(self.img) * self.gamma
                f_compl = np.fft.fftn(self.sample)
            else:
                self.f_img = np.fft.rfftn(self.img)  * self.gamma
                f_compl = np.fft.rfftn(self.sample)
                
        self.f_filter = 1. - self.gamma * f_compl.conj() * f_compl
        self.f_add = self.gamma * f_compl.conj() * self.f_img
        
        if self.useCpxFFT:
            self.prevGuess = np.fft.ifftn(self.f_add).real
        else:
            self.prevGuess = np.fft.irfftn(self.f_add, s=self.img.shape)
        
    #----------------------------------------------------------------------
    def _solve(self):
        """"""
        
        
        if self.useCpxFFT:
            f_curGuess = self.f_filter* np.fft.fftn(self.prevGuess) + self.f_add
            temp = np.fft.ifftn(f_curGuess).real

        else:
            f_curGuess = self.f_filter * np.fft.rfftn(self.prevGuess) + self.f_add
            temp = np.fft.irfftn(f_curGuess, s= self.img.shape)
            
        temp = self.analysis(temp)
        temp = self.threshold(temp)
        self.curGuess = self.synthesize(temp)        
        
        
        
        
########################################################################
class FISTA(AbstractWaveletDecon):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, 
                 image, 
                 psf = None, 
                 sample = None, 
                 groundTruth = None, 
                 solveFor = 'sample', 
                 initialGuess = 'orig_array', 
                 iterSteps = 1000, 
                 depth = 3,
                 gamma = 1.,
                 lamb = 1.,
                 errTol = 1e-5, 
                 constraints = None, 
                 isPsfCentered = True, 
                 useCpxFFT = False, 
                 debugInt = 0, 
                 compareWithTruth = False,
                 saveIntermediateSteps = 0):  
        """Constructor"""
        
        threshold = 0.5*gamma*lamb
        self.gamma = gamma
        self.lamb = lamb
        
        super(FISTA, self).__init__(image, psf, sample, groundTruth, solveFor, initialGuess, iterSteps, depth, threshold, 
                                                   'FISTA', errTol, constraints, isPsfCentered, useCpxFFT, debugInt, compareWithTruth, saveIntermediateSteps)
    
    
        
        
    #----------------------------------------------------------------------
    def _prepare(self):
        """"""
        
        if self.solveFor == 'sample':
            if self.isPsfCentered:
                temp = np.fft.ifftshift(self.psf)
            else:
                temp = self.psf
            if self.useCpxFFT:
                self.f_img = np.fft.fftn(self.img) * self.gamma
                f_compl = np.fft.fftn(temp)
            else:
                self.f_img = np.fft.rfftn(self.img) * self.gamma
                f_compl = np.fft.rfftn(temp)

        else:
            if self.useCpxFFT:
                self.f_img = np.fft.fftn(self.img) * self.gamma
                f_compl = np.fft.fftn(self.sample)
            else:
                self.f_img = np.fft.rfftn(self.img)  * self.gamma
                f_compl = np.fft.rfftn(self.sample)
                
        self.f_filter = 1. - self.gamma * f_compl.conj() * f_compl
        self.f_add = self.gamma * f_compl.conj() * self.f_img
        
        if self.useCpxFFT:
            self.prevGuess = np.fft.ifftn(self.f_add).real
        else:
            self.prevGuess = np.fft.irfftn(self.f_add, s=self.img.shape)
        self.s_buffer = self.prevGuess.copy()
        self.pk0 = 1.
        self.pk1 = 1.
        
    #----------------------------------------------------------------------
    def _solve(self):
        """"""
        
        
        if self.useCpxFFT:
            f_guess = np.fft.fftn(self.s_buffer)* self.f_filter + self.f_add
            guess = np.fft.ifftn(f_guess)
        else:
            f_guess = np.fft.rfftn(self.s_buffer)* self.f_filter + self.f_add
            guess = np.fft.irfftn(f_guess, s=self.img.shape)            
            
        guess = self.analysis(guess)
        guess = self.threshold(guess)
        self.curGuess = self.synthesize(guess)
        self.pk0 = self.pk1
        self.pk1 = 0.5*(1. + mt.sqrt(1. + 4.*self.pk0**2 ))
        
        self.s_buffer = self.curGuess + ((self.pk0 - 1.)/self.pk1) * (self.curGuess - self.prevGuess)
              
        
        
        
        