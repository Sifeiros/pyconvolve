# -*- coding: utf-8 -*-
"""
Part of the pyconvolve framework for convolution and deconvolution. 
Author: Lukas Küpper, 2018
License: GPLv3
"""
import numpy as np
import time
from decon.abstract_decon import AbstractDecon as __AD
from decon.single_step import SingleStep
import datetime as dt

import util.stack_loader

########################################################################
class AbstractIterative(__AD):
    """Abstract class for iterative Deconvolution Algorithms.

    Parameters:

    initialGuess: 		has to be string given in list INIT_GUESS. Determines weither a single step algorithm is used as the 
    				initial reconstruction condition or if the array given to the class is used.
    iterSteps:			[int], number of maximum iteration steps until the reconstruction is terminated
    errTol:			[float], when the error determined after each steps falls under this variable, the iteration is terminated
    				error determination is based on the groundTruth, if given, otherwise on the difference between the last two iterations
    saveIntermediateSteps:	[int] algorithm saves every given step. If given variable is 1, every step is saved, 2 every second, 0 only the end result

    All other parameters are the same as in 'AbstractDecon'


    """

    INIT_GUESS = SingleStep.ALGORITHMS + ['orig_array']

    #----------------------------------------------------------------------
    def __init__(self, image, 
                 psf = None, 
                 sample = None, 
                 groundTruth = None, 
                 solveFor = 'sample', 
                 initialGuess = 'WienerFilter', 
                 algoName = 'AbstractIterative',
                 iterSteps = 1000, 
                 errTol = 1e-5, 
                 constraints = None, 
                 isPsfCentered = True, 
                 useCpxFFT = False, 
                 debugInt = 0, 
                 compareWithTruth = False,
                 saveIntermediateSteps = 0):
        """Constructor"""

        super(AbstractIterative, self).__init__(image, psf, sample, groundTruth, algoName, solveFor, constraints, isPsfCentered, useCpxFFT, debugInt, compareWithTruth)

        self.maxIteration = iterSteps
        self.errorTolerance = errTol
        self.saveImageStep = saveIntermediateSteps

        if initialGuess in AbstractIterative.INIT_GUESS:
            self.initialGuess = initialGuess
        else:
            self.print_dbg('Warning: {} is not a valid initial guess parameter. Using {} instead.'.format(initialGuess, AbstractIterative.INIT_GUESS[0]), 0)
            self.initialGuess = AbstractIterative.INIT_GUESS[0]

        self.timePerStep = np.zeros(self.maxIteration+1)
        self.avgTimeperStep = 0.
        self.errors = np.zeros(self.maxIteration + 2)

        self.curIter = 0
        self.curError = np.inf


    #----------------------------------------------------------------------
    def getAlgoParameters(self):
        """Function has to be implemented by all reconstruction algorithms."""
        return {'initialGuess':self.initialGuess, 'iterationSteps':self.maxIteration, 'errorTolerance':self.errorTolerance, 'saveAfterSteps':self.saveImageStep}

    #----------------------------------------------------------------------
    def getCurrentSolution(self, forSaving = False):
        """"""
        if not forSaving:
            return self.curGuess
        else:
            tmp = self.curGuess.copy()
                
            max_v = tmp.max()
            tmp = tmp/max_v*255
            tmp = tmp.round()
            if tmp.max() > 255:
                raise ValueError
            tmp = tmp.astype('uint8')
            return tmp

    #----------------------------------------------------------------------
    def prepare(self):
        """"""

        self.print_dbg('Starting preparation for {}...'.format(self.algoName), 3)
        time_st = time.time()
        self.matchArrayShapes()
        self.curGuess = np.zeros(self.img.shape, dtype = 'float')
        self.prevGuess = np.zeros(self.img.shape, dtype = 'float')
        self._initGuess()
        self._prepare()
        time_end = time.time()
        self.timeToPrepare = time_end-time_st
        self.timeOverall += self.timeToPrepare
        self.print_dbg('Finished preparations for {} in {:.3f} s'.format(self.algoName, self.timeToPrepare), 1)        




    #----------------------------------------------------------------------
    def _initGuess(self):
        """"""
        self.print_dbg('Preparing initial guess of array with {}...'.format(self.initialGuess), 2)
        if self.initialGuess in SingleStep.ALGORITHMS:
            tmp_initial = SingleStep(img = self.img, 
                                     psf= self.psf, 
                                     sample= self.sample, 
                                     groundTruth= self.groundTruth, 
                                     name= self.initialGuess, 
                                     solveFor= self.solveFor, 
                                     constraints= self.constraints, 
                                     isPsfCentered= self.isPsfCentered, 
                                     useCpxFFT= self.useCpxFFT, 
                                     compareWithTruth= self.compareWithTruth)
            self.print_dbg('Starting preparation...', 3)
            tmp_initial.prepare()
            self.print_dbg('Done. Starting Solution...', 3)
            tmp_initial.solve()
            self.print_dbg('Done. Solution took {:.3f} s'.format(tmp_initial.algo.timeOverall), 3)
            self.prevGuess = tmp_initial.out.copy()
            if self.compareWithTruth:
                self.print_dbg('Error after solution: {}'.format(tmp_initial.curError), 3)
                self.errors[0] = tmp_initial.curError
            del(tmp_initial)

        elif self.initialGuess == AbstractIterative.INIT_GUESS[3]:
            #orig array
            self.img.copy()
            self.errors[0] = -1.



    #----------------------------------------------------------------------
    def solve(self):
        """"""

        self.print_dbg('Starting reconstruction for {} by {}...'.format(self.solveFor, self.algoName), 3)



        while self.curIter <= self.maxIteration and self.curError > self.errorTolerance:

            t_start = time.time()
            self._solve()
            self.applyConstraint()
            t_end = time.time()
            self.timePerStep[self.curIter] = t_end - t_start
            self.curError = self.calcError()
            self.errors[self.curIter +1] = self.curError

            if self.saveImageStep != 0 and self.curIter % self.saveImageStep == 0:
                self._saveIntermediate()

            self.prevGuess = self.curGuess
            self.curIter += 1

        self.timeToSolve = self.timePerStep.sum()
        self.avgTimeperStep = self.timeToSolve / self.curIter

        self.timeOverall = self.timeToSolve + self.timeToPrepare

        self.print_dbg('Finished reconstructions by {} in {:.3f} s'.format(self.algoName, self.timeOverall), 1)	
        self.print_dbg('Reconstruction terminated after {} steps with a final error of {} per pixel.'.format(self.curIter, self.curError), 2)
        self.print_dbg('time for pre-processing: {:.3f} s, time for iterative solution: {:.3f} s, Avg. time per iteration step {:.3f} s'.format(self.timeToPrepare, self.timeToSolve, self.timePerStep.mean()), 2)
        if self.solveFor == 'psf':
            self.out = self.psf
        if self.solveFor == 'sample':
            self.out = self.sample
            
        self.timestring = dt.datetime.now().strftime('%Y-%m-%d %H:%M')


    #----------------------------------------------------------------------
    def calcError(self):
        """"""
        if self.compareWithTruth:
            if self.solveFor == 'sample':
                return np.abs(self.groundTruth - self.curGuess).sum()/self.sample.size
            if self.solveFor == 'psf':
                return np.abs(self.groundTruth - self.curGuess).sum()/self.psf.size
        else:
            if self.solveFor == 'sample':
                return np.abs(self.curGuess - self.prevGuess).sum()/self.sample.size
            if self.solveFor == 'psf':
                return np.abs(self.curGuess - self.prevGuess).sum()/self.sample.size

    #----------------------------------------------------------------------
    #def getCurrentSolution(self):
        #""""""
        #return self.curGuess


    #----------------------------------------------------------------------
    def initSaveParameters(self, save_path, save_fident, intermediate_path = 'inter_[IND]', orig_img_path=[None, None], orig_psf_path=[None, None], orig_sample_path=[None, None], orig_truth_path=[None, None], overwrite = False):
        """"""
        
        super(AbstractIterative, self).initSaveParameters(save_path, save_fident, orig_img_path, orig_psf_path, orig_sample_path, orig_truth_path, overwrite)

        self.intermediate_path_add = intermediate_path
        
        #if util.checkAllPaths(intermediate_paths[0]):        
            #self.intermediate_path = intermediate_paths[0]
        #else:
            #raise ValueError('Given path for original image is not valid.')        
        #if not intermediate_paths is None:
            #self.intermediate_fident = intermediate_paths[1]
        #else:
            #self.intermediate_fident = save_fident


    #----------------------------------------------------------------------
    def _saveIntermediate(self):
        """"""

        temp_path = self.intermediate_path_add
        temp_path = temp_path.replace('[IND]', '{0}').format(self.curIter)
        temp_path = util.ptjoin(self.save_path, temp_path)
        
        util.path_declarations.createAllPaths(temp_path)

        meta = self.meta.getIntermediateMeta(self, temp_path)

        util.stack_loader.write_image_stack(self.getCurrentSolution(forSaving=True), temp_path, self.save_fident, 0, meta_data=meta)




########################################################################
class JannsonVCittert(AbstractIterative):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, 
                 image, 
                 psf = None, 
                 sample = None, 
                 groundTruth = None, 
                 solveFor = 'sample', 
                 initialGuess = 'orig-array', 
                 iterSteps = 1000, 
                 errTol = 1e-5, 
                 gamma = 1.,
                 constraints = None, 
                 isPsfCentered = True, 
                 useCpxFFT = False, 
                 debugInt = 0, 
                 compareWithTruth = False,
                 saveIntermediateSteps = 0):
        """Constructor"""

        super(JannsonVCittert, self).__init__(image, psf, sample, groundTruth, solveFor, initialGuess, 'JannsonVCittert', 
                                              iterSteps, errTol, constraints, isPsfCentered, useCpxFFT, debugInt, compareWithTruth, 
                                              saveIntermediateSteps)
        self.gamma = gamma


    #----------------------------------------------------------------------
    def getAlgoParameters(self):
        """Function has to be implemented by all reconstruction algorithms."""
        tmp_dict = super(JannsonVCittert, self).getAlgoParameters()
        tmp_dict['gamma'] = self.gamma
        return tmp_dict


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

        self.filt = 1.0 - self.gamma * f_compl.real - 1j*self.gamma * f_compl.imag



    #----------------------------------------------------------------------
    def _solve(self):
        """"""
        if self.useCpxFFT:
            #self.curGuess = np.fft.ifftn(np.fft.fftn(self.prevGuess) * self.filt + self.f_img).real
            self.curGuess = self.img + np.fft.ifftn(np.fft.fftn(self.prevGuess) * self.filt).real            
        else:
            #self.curGuess = np.fft.irfftn(np.fft.rfftn(self.prevGuess) * self.filt + self.f_img, self.min_mut_shape)
            self.curGuess = self.img + np.fft.irfftn(np.fft.rfftn(self.prevGuess) * self.filt, self.min_mut_shape)
            



########################################################################
class Gold(AbstractIterative):
    """
    EXPERIMENTAL
    """

    #----------------------------------------------------------------------        
    def __init__(self, 
                 image, 
                 psf = None, 
                 sample = None, 
                 groundTruth = None, 
                 solveFor = 'sample', 
                 initialGuess = 'orig-array', 
                 experimentalMode = True,
                 iterSteps = 1000, 
                 errTol = 1e-5, 
                 constraints = None, 
                 isPsfCentered = True, 
                 useCpxFFT = False, 
                 debugInt = 0, 
                 compareWithTruth = False,
                 saveIntermediateSteps = 0):
        """Constructor"""
        super(Gold, self).__init__(image, psf, sample, groundTruth, solveFor, initialGuess, 'GoldAlgorithm', 
                                   iterSteps, errTol, constraints, isPsfCentered, useCpxFFT, debugInt, compareWithTruth, 
                                   saveIntermediateSteps)    
        self.experimentalMode = experimentalMode

    #----------------------------------------------------------------------
    def getAlgoParameters(self):
        """Function has to be implemented by all reconstruction algorithms."""
        tmp_dict = super(Gold, self).getAlgoParameters()
        tmp_dict['experimentalMode'] = self.experimentalMode
        return tmp_dict

    #----------------------------------------------------------------------
    def _prepare(self):
        """"""
        if self.experimentalMode:
            if self.solveFor == 'sample':
                if self.isPsfCentered:
                    psf = np.fft.ifftshift(self.psf)
                else:
                    psf = self.psf
                if self.useCpxFFT:
                    self.img_conv_compl = np.fft.ifftn(np.fft.fftn(self.img) * np.fft.fftn(psf)).real
                    self.f_compl2 = np.power(np.fft.fftn(psf),2)
                else:
                    self.img_conv_compl = np.fft.irfftn(np.fft.rfftn(self.img) * np.fft.rfftn(psf), self.min_mut_shape)
                    self.f_compl2 = np.power(np.fft.rfftn(psf),2)
            else:
                if self.useCpxFFT:
                    self.img_conv_compl = np.fft.ifftn(np.fft.fftn(self.img) * np.fft.fftn(self.sample)).real
                    self.f_compl2 = np.power(np.fft.fftn(self.sample),2)		
                else:
                    self.img_conv_compl = np.fft.irfftn(np.fft.rfftn(self.img) * np.fft.rfftn(self.sample), self.min_mut_shape)
                    self.f_compl2 = np.power(np.fft.rfftn(self.sample),2)
        else:
            if self.solveFor == 'sample':
                if self.isPsfCentered:
                    psf = np.fft.ifftshift(self.psf)
                else:
                    psf = self.psf
                if self.useCpxFFT:
                    self.f_compl = np.fft.fftn(psf)
                else:
                    self.f_compl = np.fft.rfftn(psf)
            else:
                if self.useCpxFFT:
                    self.f_compl = np.fft.fftn(self.sample)
                else:
                    self.f_compl = np.fft.rfftn(self.sample)


    #----------------------------------------------------------------------
    def _solve(self):
        """"""
        if self.experimentalMode:
            if self.useCpxFFT:
                temp = np.fft.ifftn(self.f_compl2 * np.fft.fftn(self.prevGuess)).real
            else:
                temp = np.fft.irfftn(self.f_compl2 * np.fft.rfftn(self.prevGuess), self.min_mut_shape)

            self.curGuess = (self.img_conv_compl / temp) * self.prevGuess
        else:
            if self.useCpxFFT:
                temp = np.fft.ifftn( self.f_compl * np.fft.fftn(self.prevGuess)).real
            else:
                temp = np.fft.irfftn(self.f_compl * np.fft.rfftn(self.prevGuess), self.min_mut_shape)

            self.curGuess = (self.img / temp) * self.prevGuess




########################################################################
class RichardsonLucy(AbstractIterative):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, 
                 image, 
                 psf = None, 
                 sample = None, 
                 groundTruth = None, 
                 solveFor = 'sample', 
                 initialGuess = 'WienerFilter', 
                 iterSteps = 1000, 
                 errTol = 1e-5, 
                 p = 1.,
                 constraints = None, 
                 isPsfCentered = True, 
                 useCpxFFT = False, 
                 debugInt = 0, 
                 compareWithTruth = False,
                 saveIntermediateSteps = 0):
        """Constructor"""

        super(RichardsonLucy, self).__init__(image, psf, sample, groundTruth, solveFor, initialGuess, 'RichardsonLucy', 
                                             iterSteps, errTol, constraints, isPsfCentered, useCpxFFT, debugInt, compareWithTruth, 
                                             saveIntermediateSteps)
        self.p = p

    #----------------------------------------------------------------------
    def getAlgoParameters(self):
        """Function has to be implemented by all reconstruction algorithms."""
        tmp_dict = super(RichardsonLucy, self).getAlgoParameters()
        tmp_dict['pParameter'] = self.p
        return tmp_dict


    #----------------------------------------------------------------------
    def _prepare(self):
        """"""
        if self.solveFor == 'sample':
            if self.isPsfCentered:
                psf = np.fft.ifftshift(self.psf)
            else:
                psf = self.psf
            if self.useCpxFFT:
                self.f_compl = np.fft.fftn(psf)
            else:
                self.f_compl = np.fft.rfftn(psf)

        else:
            if self.useCpxFFT:
                self.f_compl = np.fft.fftn(self.sample)
            else:
                self.f_compl = np.fft.rfftn(self.sample)

    #----------------------------------------------------------------------
    def _solve(self):
        """"""
        
        if self.useCpxFFT:
            f_guess = np.fft.fftn(self.prevGuess)
            temp = np.fft.ifftn(f_guess * self.f_compl).real
            #temp = np.fft.ifftn(np.fft.fftn(self.img / temp)* self.f_compl.conj()).real
            temp = np.fft.ifftn(np.fft.fftn(self.img / temp) * self.f_compl[::-1,::-1,::-1])
            if self.p == 1:
                self.curGuess = self.prevGuess * temp
            else:
                self.curGuess = self.prevGuess * np.power(temp, self.p)
        else:
            f_guess = np.fft.rfftn(self.prevGuess)
            temp = np.fft.irfftn(f_guess * self.f_compl, self.min_mut_shape)
            #self.print_dbg('DEBUG: crgx{}, crgm{}'.format(self.curGuess.max(), self.curGuess.min()),3)
            #self.print_dbg('DEBUG: fgx{}, fgm{}, tpx{}, tpm{}'.format(f_guess.max(), f_guess.min(), temp.max(), temp.min()), 3)
            #temp = np.fft.irfftn(np.fft.rfftn(self.img / temp) * self.f_compl.conj(), self.min_mut_shape)
            temp = np.fft.irfftn(np.fft.rfftn(self.img / temp) * self.f_compl[::-1,::-1,::-1], self.min_mut_shape)
            #self.curGuess = self.prevGuess * temp
            if self.p == 1:
                self.curGuess = self.prevGuess * temp
            else:
                self.curGuess = self.prevGuess * np.power(temp, self.p)


########################################################################
class Landweber(AbstractIterative):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, 
                 image, 
                 psf = None, 
                 sample = None, 
                 groundTruth = None, 
                 solveFor = 'sample', 
                 initialGuess = 'WienerFilter', 
                 iterSteps = 1000, 
                 errTol = 1e-5, 
                 gamma = 1.,
                 constraints = None, 
                 isPsfCentered = True, 
                 useCpxFFT = False, 
                 debugInt = 0, 
                 compareWithTruth = False,
                 saveIntermediateSteps = 0):
        """Constructor"""

        super(Landweber, self).__init__(image, psf, sample, groundTruth, solveFor, initialGuess, 'Landweber', 
                                        iterSteps, errTol, constraints, isPsfCentered, useCpxFFT, debugInt, compareWithTruth, 
                                        saveIntermediateSteps)
        self.gamma = gamma

    #----------------------------------------------------------------------
    def getAlgoParameters(self):
        """Function has to be implemented by all reconstruction algorithms."""
        tmp_dict = super(Landweber, self).getAlgoParameters()
        tmp_dict['gamma'] = self.gamma
        return tmp_dict


    #----------------------------------------------------------------------
    def _prepare(self):
        """"""
        if self.solveFor == 'sample':
            if self.isPsfCentered:
                psf = np.fft.ifftshift(self.psf)
            else:
                psf = self.psf
            if self.useCpxFFT:
                f_compl = np.fft.fftn(psf)
                f_img = np.fft.fftn(self.img)
            else:
                f_compl = np.fft.rfftn(psf)
                f_img = np.fft.rfftn(self.img)
        else:
            if self.useCpxFFT:
                f_compl = np.fft.fftn(self.sample)
                f_img = np.fft.rfftn(self.img)
            else:
                f_compl = np.fft.rfftn(self.sample)
                f_img = np.fft.rfftn(self.img)

        self.f_filt = 1.0 - self.gamma * f_compl * f_compl.conj()
        self.f_add = self.gamma*(f_compl.real*f_img.real + f_compl.imag * f_img.imag) + 1j*self.gamma*(f_compl.real*f_img.imag - f_compl.imag*f_img.real)



    #----------------------------------------------------------------------
    def _solve(self):
        """"""

        if self.useCpxFFT:
            f_guess = np.fft.fftn(self.prevGuess) * self.f_filt + self.f_add
            self.curGuess = np.fft.ifftn(f_guess).real
        else:
            f_guess = np.fft.rfftn(self.prevGuess) * self.f_filt + self.f_add
            self.curGuess = np.fft.irfftn(f_guess, self.min_mut_shape)



########################################################################
class TikhonovMiller(AbstractIterative):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, 
                 image, 
                 psf = None, 
                 sample = None, 
                 groundTruth = None, 
                 solveFor = 'sample', 
                 initialGuess = 'Wiener', 
                 iterSteps = 1000, 
                 errTol = 1e-5, 
                 gamma = 1.,
                 lamb = 1.,
                 constraints = None, 
                 isPsfCentered = True, 
                 useCpxFFT = False, 
                 debugInt = 0, 
                 compareWithTruth = False,
                 saveIntermediateSteps = 0):
        """Constructor"""



        super(TikhonovMiller, self).__init__(image, psf, sample, groundTruth, solveFor, initialGuess, 'TikhonovMiller', 
                                        iterSteps, errTol, constraints, isPsfCentered, useCpxFFT, debugInt, compareWithTruth, 
                                        saveIntermediateSteps)	
        self.gamma = gamma
        self.lamb = lamb
        
        
    #----------------------------------------------------------------------
    def getAlgoParameters(self):
        """Function has to be implemented by all reconstruction algorithms."""
        tmp_dict = super(TikhonovMiller, self).getAlgoParameters()
        tmp_dict['gamma'] = self.gamma
        tmp_dict['lambda'] = self.lamb
        return tmp_dict

    #----------------------------------------------------------------------
    def _prepare(self):
        """"""
        print('CHECK WORKING PRINCIPLE BC OF ORDER OF FOURIER COEEFFICIENTS')
        
        if self.solveFor == 'sample':
            if self.isPsfCentered:
                psf = np.fft.ifftshift(self.psf)
            else:
                psf = self.psf
            if self.useCpxFFT:
                f_img = np.fft.fftn(self.img)
                f_compl = np.fft.fftn(psf)
            else:
                f_img = np.fft.rfftn(self.img)
                f_compl = np.fft.rfftn(psf)
        else:
            if self.useCpxFFT:
                f_img = np.fft.fftn(self.img)
                f_compl = np.fft.fftn(self.sample)
            else:
                f_img = np.fft.fftn(self.img)
                f_compl = np.fft.fftn(self.sample)
        if self.useCpxFFT:
            #A = 1.0 - self.gamma*(f_compl.real**2 + f_compl.imag**2)
            A = 1.0 - self.gamma*np.abs(f_compl)**2
            L2 = self.gamma*self.lamb*self._laplacian2(self.img.shape)
            self.f_filt = A - L2
            self.f_add = self.gamma * f_compl.conj() * f_img
        else:
            A = 1.0 - self.gamma*(f_compl.real**2 + f_compl.imag**2)
            L2 = self.gamma*self.lamb*self._laplacian2(self.img.shape)            
            self.f_filt = A - L2[:,:,:A.shape[2]]
            self.f_add = self.gamma * f_compl.conj() * f_img



    #----------------------------------------------------------------------
    def _laplacian2(self, shp):
        """"""
        [x_red, y_red, z_red] = [int(round(ni/2.)) for ni in shp]


        [X,Y,Z] = np.meshgrid(np.arange(x_red), np.arange(y_red), np.arange(z_red))
        X = X/x_red
        Y = Y/y_red
        Z = Z/z_red

        lapl_sec = np.pi**6 * (X**2 + Y**2 + Z**2)

        mid_low = [int((i/2)) for i in shp]    
        mid_high = [int(round(i/2.)) if i % 2 else (i/2) for i in shp]    

        ret = np.zeros(shp, dtype = 'float')

        ret[0:mid_high[0],   0:mid_high[1],    0:mid_high[2]]    = lapl_sec[:,:,:]
        ret[mid_low[0]:,      0:mid_high[1],    0:mid_high[2]]    = lapl_sec[::-1,:,:]
        ret[0:mid_high[0],   mid_low[1]:,       0:mid_high[2]]    = lapl_sec[:,::-1,:]
        ret[mid_low[0]:,      mid_low[1]:,       0:mid_high[2]]    = lapl_sec[::-1,::-1,:]

        ret[0:mid_high[0],   0:mid_high[1],    mid_low[2]:]       = lapl_sec[:,:,::-1]
        ret[mid_low[0]:,      0:mid_high[1],    mid_low[2]:]       = lapl_sec[::-1,:,::-1]
        ret[0:mid_high[0],   mid_low[1]:,       mid_low[2]:]       = lapl_sec[:,::-1,::-1]
        ret[mid_low[0]:,      mid_low[1]:,       mid_low[2]:]       = lapl_sec[::-1,::-1,::-1]    

        return np.fft.fftshift(ret)


    #----------------------------------------------------------------------
    def _solve(self):
        """"""

        if self.useCpxFFT:
            f_guess = np.fft.fftn(self.prevGuess)
            self.curGuess = np.fft.ifftn(f_guess * self.f_filt + self.f_add)
        else:
            f_guess = np.fft.rfftn(self.prevGuess)
            self.curGuess = np.fft.irfftn(f_guess * self.f_filt+self.f_add, self.min_mut_shape)
            
            
            
            

########################################################################
class ICTM(TikhonovMiller):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, 
                 image, 
                 psf = None, 
                 sample = None, 
                 groundTruth = None, 
                 solveFor = 'sample', 
                 initialGuess = 'Wiener', 
                 iterSteps = 1000, 
                 errTol = 1e-5, 
                 gamma = 1.,
                 lamb = 1.,
                 constraints = None, 
                 isPsfCentered = True, 
                 useCpxFFT = False, 
                 debugInt = 0, 
                 compareWithTruth = False,
                 saveIntermediateSteps = 0):
        """Constructor"""


        
        if constraints is None:
            constraints = ['non_negative',{}]
        elif type(constraints) == str:
            constraints = [['non_negative', {}], constraints]
        elif type(constraints) == list:
            if len(constraints) == 2 and type(constraints[0]) == str and type(constraints[1]) == dict:
                constraints = [['non_negative', {}], constraints]
            else:
                constraints = [['non_negative', {}]] + constraints
        else:
            raise TypeError
        
        
        super(ICTM, self).__init__(image, psf, sample, groundTruth, solveFor, initialGuess, 
                                        iterSteps, errTol, gamma, lamb, constraints, isPsfCentered, useCpxFFT, debugInt, compareWithTruth, 
                                        saveIntermediateSteps)	
        
        self.algoName = 'ICTM'
    
    


            
########################################################################
class StarkParker(AbstractIterative):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, 
                 image, 
                 psf = None, 
                 sample = None, 
                 groundTruth = None, 
                 solveFor = 'sample', 
                 initialGuess = 'Wiener', 
                 iterSteps = 1000, 
                 errTol = 1e-5, 
                 gamma = 1.0,
                 vmin = -1,
                 vmax = -1,
                 constraints = None, 
                 isPsfCentered = True, 
                 useCpxFFT = False, 
                 debugInt = 0, 
                 compareWithTruth = False,
                 saveIntermediateSteps = 0):
        """Constructor"""


        if vmin == -1:
            vmin = image.min()
        if vmax == -1:
            vmax = image.max()


        
        if constraints is None:
            constraints = ['clipped',{'vmin':vmin, 'vmax':vmax}]
        elif type(constraints) == str:
            constraints = [['clipped', {'vmin':vmin, 'vmax':vmax}], constraints]
        elif type(constraints) == list:
            if len(constraints) == 2 and type(constraints[0]) == str and type(constraints[1]) == dict:
                constraints = [['clipped', {'vmin':vmin, 'vmax':vmax}], constraints]
            else:
                constraints = [['clipped', {'vmin':vmin, 'vmax':vmax}]] + constraints
        else:
            raise TypeError
 
        super(StarkParker, self).__init__(image, psf, sample, groundTruth, solveFor, initialGuess, 'StarkParker', 
                                        iterSteps, errTol, constraints, isPsfCentered, useCpxFFT, debugInt, compareWithTruth, 
                                        saveIntermediateSteps)	
        self.gamma = gamma
    
    #----------------------------------------------------------------------
    def getAlgoParameters(self):
        """Function has to be implemented by all reconstruction algorithms."""
        tmp_dict = super(StarkParker, self).getAlgoParameters()
        tmp_dict['gamma'] = self.gamma
        return tmp_dict

    
    #----------------------------------------------------------------------
    def _prepare(self):
        """"""
        if self.solveFor == 'sample':
            if self.isPsfCentered:
                psf = np.fft.ifftshift(self.psf)
            else:
                psf = self.psf
            if self.useCpxFFT: 
                f_img = np.fft.fftn(self.img)
                f_compl = np.fft.fftn(psf)
            else:
                f_img = np.fft.rfftn(self.img)
                f_compl = np.fft.rfftn(psf)
        else:
            if self.useCpxFFT:
                f_img = np.fft.fftn(self.img)
                f_compl = np.fft.fftn(self.sample)
            else:
                f_img = np.fft.rfftn(self.img)
                f_compl = np.fft.rfftn(self.sample)
        self.f_filter = 1.0  - self.gamma*f_compl*f_compl.conj()
        self.f_add = self.gamma * f_compl.conj() * f_img
        

    #----------------------------------------------------------------------
    def _solve(self):
        """"""

        if self.useCpxFFT:
            f_guess = np.fft.fftn(self.prevGuess)
            self.curGuess = np.fft.ifftn(f_guess * self.f_filter + self.f_add).real
            
        else:
            f_guess = np.fft.rfftn(self.prevGuess)
            self.curGuess = np.fft.irfftn(f_guess * self.f_filter+self.f_add, self.min_mut_shape)
    
            