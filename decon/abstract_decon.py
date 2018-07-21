import numpy as np
import time
import util
import util.stack_loader 
import datetime as dt


########################################################################
class AbstractDecon(object):
    """
    Abstract Class. All Deconvolution algorithms should inherit from this
    Initializes and / or references the base image and the arrays for the PSF and the sample
    image:	has to be given. the basis for deconvolution reconstruction. dtype float or int
    psf:	can be existing np.array of dtype float or int or None.
    sample:	can be existing np.array of dtype float or int or None
    algoName:	references the short name of the used algorithm
    solveFor: 	determines if the algorithm solves for the 'sample' or the 'psf' depending on the 
    		intended purpose. Different constraints may be applied for each case
    constraints:string or list of strings. valid constraints are 'non_negative', 'clipped', 'normalized', 
    		'rescaled', 'energy_preserved'
    out_dtype:	'float', 'int' or variants (e.g. int8)
    out_rescale:'original', 'dtype_max'
    useCpxFFT: 	determines if the algorithm should use the complex implementation of the FFT supplied by the 
    		numpy.fft package or the version specifically designed for real input signals
    debugInt: 	determines the output intensity of the algorithm. 0 refers to silent reconstruction, 1 to 
    		some time relevant output like iteration steps, 2 reveals additional meta information and 3 
                outputs extensive additional information about the reconstruction process
    """
    CONSTRAINTS = ['non_negative', 'clipped', 'normalized_mean', 'normalized_abs' 'rescaled', 'energy_preserved']


    #----------------------------------------------------------------------
    def __init__(self, image, psf = None, sample = None, groundTruth = None, 
                 algoName = 'Abstr', 
                 solveFor = 'sample', 
                 constraints = None, 
                 isPsfCentered = True,
                 useCpxFFT = False, 
                 debugInt = 0, 
                 compareWithTruth = False):
        """Constructor:"""

        self.img = image.astype('float')
        self.algoName = algoName
        self.useCpxFFT = useCpxFFT
        self.isPsfCentered = isPsfCentered
        self.debugInt = debugInt
        self.constraints = []
        self.constr_params = []
        self.meta = util.stack_loader.MetaData()

        if psf is None:
            self.psf = np.zeros(self.img.shape, dtype='float')
        else:
            self.psf = psf.astype('float')
        if sample is None:
            self.sample = np.zeros(self.img.shape, dtype='float')
        else:
            self.sample = sample.astype('float')

        if solveFor == 'sample' and psf is not None:
            self.solveFor = solveFor
        elif solveFor == 'psf' and sample is not None:
            self.solveFor = solveFor
        else:
            raise ValueError('Can only solve for PSF or sample. Use \'sample\' or \'psf\' and ensure that the other array is valid.')


        self.init_constraints(constraints)

        if compareWithTruth and groundTruth is None:
            raise ValueError('If the algorithm shall compare results with a ground truth, "groundTruth" has to be different to None')
        else:
            self.compareWithTruth = compareWithTruth
            self.groundTruth = groundTruth


        self.timeToPrepare = 0.
        self.timeToSolve = 0.
        self.timeOverall = 0.
        
        self.out = None
        
        self.save_fident = None
        self.save_path = None
        self.orig_img_path = None
        self.orig_psf_path = None
        self.orig_sample_path = None
        self.orig_truth_path = None
        self.orig_type = 'unknown'
        
        self.comments = ''




    #----------------------------------------------------------------------
    def prepare(self):
        """Starts preparations and pre-processing for the reconstruction process. Steps are timed. Elapsed time is stored in .timeToPrepare"""

        self.print_dbg('Starting preparation for {}...'.format(self.algoName), 3)
        time_st = time.time()
        self._prepare()
        time_end = time.time()
        self.timeToPrepare = time_end-time_st
        self.timeOverall += self.timeToPrepare
        self.print_dbg('Finished preparations for {} in {:.3f} s'.format(self.algoName, self.timeToPrepare), 1)


    #----------------------------------------------------------------------
    def _prepare(self):
        """Has to be implemented by all Reconstruction Algorithms. Includes all pre-processing steps."""
        raise NotImplementedError('Every Deconvolution algorithm has to implement the "_prepare" method individually.')




    #----------------------------------------------------------------------
    def matchArrayShapes(self):
        """Pads the PSF to the size of the input image."""
        if not self.psf.shape == self.img.shape == self.sample.shape:
            self.print_dbg('Dimensions of PSF, Image and Sample do not match. Attempting to zero-pad arrays to minimal mutual shape.', 2)
            self.print_dbg('PSF shape: {} Image shape: {} Sample shape: {}'.format(self.psf.shape, self.img.shape, self.sample.shape), 3)	    

            if self.compareWithTruth:
                min_mut_shape = [max(self.psf.shape[i], self.img.shape[i], self.sample.shape[i], self.groundTruth.shape[i]) for i in range(len(self.sample.shape))]
            else:
                min_mut_shape = [max(self.psf.shape[i], self.img.shape[i], self.sample.shape[i]) for i in range(len(self.sample.shape))]
            self.min_mut_shape = min_mut_shape

            diff_psf = [(min_mut_shape[i] - self.psf.shape[i]) for it in range(len(self.sample.shape))]
            diff_img = [(min_mut_shape[i] - self.img.shape[i]) for it in range(len(self.sample.shape))]
            diff_samp = [(min_mut_shape[i] - self.sample.shape[i]) for it in range(len(self.sample.shape))]
            if self.compareWithTruth:
                diff_ground = [(min_mut_shape[i] - self.groundTruth.shape[i]) for it in range(len(self.sample.shape))]

            if any([sh > 0 for sh in diff_psf]):
                if self.solveFor == 'psf':
                    self.psf = np.zeros(min_mut_shape)
                else:
                    self.print_dbg('Zero-Padding PSF...', 3)
                    self.psf = AbstractDecon._padArray(self.psf, min_mut_shape)
            if any([sh > 0 for sh in diff_img]):
                self.print_dbg('Zero-Padding Image...', 3)
                self.img = AbstractDecon._padArray(self.img, min_mut_shape)
            if any([sh > 0 for sh in diff_samp]):
                if self.solveFor == 'sample':
                    self.sample = np.zeros(min_mut_shape)
                else:
                    self.print_dbg('Zero-Padding Sample...', 3)
                    self.sample = AbstractDecon._padArray(self.sample, min_mut_shape)
            if self.compareWithTruth and any([sh > 0 for sh in diff_ground]):
                self.print_dbg('Zero-Padding Ground Truth...', 3)
                self.groundTruth = AbstractDecon._padArray(self.groundTruth, min_mut_shape)
            self.min_mut_shape = min_mut_shape
        else:
            self.min_mut_shape = self.img.shape



    #----------------------------------------------------------------------
    @staticmethod
    def _padArray(arr, shape):
        """
        Pads the given array to the given shape.
        Dtype of the original array is preserved. 
        """
        #print('Debugging: {}'.format(shape))
        diff_shape = [shape[i] - arr.shape[i] for i in range(len(shape))]
        new_arr2 = np.zeros(shape, dtype=arr.dtype)

        sl_list = []
        sl_temp_in = []
        sl_temp_out = []

        for sh_it in arr.shape:
            sl_list.append(slice(None,sh_it, 1))
            sl_temp_in.append(slice(None, None, 1))
            sl_temp_out.append(slice(None,None, 1))
        new_arr2[sl_list] = arr.copy()

        for it in range(len(diff_shape)):
            if diff_shape[it] == 0:
                continue
            if diff_shape[it] % 2 == 0:
                slic_new_pos = slice(diff_shape[it]/2, arr.shape[it]+diff_shape[it]/2,1)
                slic_zeros = slice(None, diff_shape[it]/2, 1)
            else:
                print('Warning: asymmetrical padding of PSF. Previously centered features are no longer centered.')
                slic_new_pos = slice(int(diff_shape[it]/2), arr.shape[it]+int(diff_shape[it]/2),1)
                slic_zeros = slice(None, int(diff_shape[it]/2), 1)
            sl_temp_in[it] = sl_list[it]
            sl_temp_out[it] = slic_new_pos

            new_arr2[sl_temp_out] = new_arr2[sl_temp_in] 
            sl_temp_out[it] = slic_zeros
            new_arr2[sl_temp_out] = 0.
            sl_temp_in[it] = slice(None, None, 1)
            sl_temp_out[it] = slice(None, None, 1)

        return new_arr2



    #----------------------------------------------------------------------
    def solve(self):
        """Starts the reconstruction process of the algorithm. Process is timed. Timing result is stored in .timeToSolve. 
        Result of the reconstruction is referenced by .out as well as the respective array defined by .solveFor"""


        self.print_dbg('Starting reconstruction by {}...'.format(self.algoName), 3)
        time_st = time.time()
        self._solve()
        time_end = time.time()
        self.timeToSolve = time_end-time_st
        self.timeOverall += self.timeToSolve

        self.print_dbg('Finished reconstructions by {} in {:.3f} s'.format(self.algoName, self.timeToSolve), 1)	
        if self.solveFor == 'psf':
            self.out = self.psf
        if self.solveFor == 'sample':
            self.out = self.sample
            
        self.timestring = dt.datetime.now().strftime('%Y-%m-%d %H:%M')

    #----------------------------------------------------------------------
    def _solve(self):
        """Abstract function. Starts the solving process."""
        raise NotImplementedError('Error in '+self.__class__+'. Function "solve" not implemented.')



    #----------------------------------------------------------------------
    def print_dbg(self, string, dbg):
        """"""
        if dbg <= self.debugInt:
            print string    

    #----------------------------------------------------------------------
    def calcError(self):
        """"""

        if self.compareWithTruth:
            if self.solveFor == 'sample':
                return np.abs(self.groundTruth - self.sample).sum()/self.sample.size
            if self.solveFor == 'psf':
                return np.abs(self.groundTruth - self.psf).sum()/self.psf.size
        else:
            raise NotImplementedError('Can only calculate deviation to ground truth. No ground truth given at creation.')

    #----------------------------------------------------------------------
    def init_constraints(self, constr):
        """"""


        tmp_constr = []
        tmp_constr_params = []
        
        if constr is None:
            pass
        elif type(constr) == str:
            #Single constraint, represented by single string
            if constr in AbstractDecon.CONSTRAINTS:
                tmp_constr.append(constr)
                tmp_constr_params.append({})
            else:
                #print(constr)
                raise ValueError('constraint not known')
        
        elif type(constr) == list:
            #Several possibilities:
            #- more than one constraint, each represented by [single string, single string]
            #- more than one constraint, each represented by [[string + dict], [string + dict]]
            #- single constraint, represented by [string,dict]
            
            if len(constr) == 2 and type(constr[0]) == str and type(constr[1]) == dict:
                if constr[0] in AbstractDecon.CONSTRAINTS:
                    tmp_constr.append(constr[0])
                    tmp_constr_params.append(constr[1])
                else:
                    #print(constr[0])
                    raise ValueError('constraint not known')
            else:
                
                for u_c in constr:
                    if type(u_c) == str:
                        if u_c in AbstractDecon.CONSTRAINTS:
                            tmp_constr.append(u_c)
                            tmp_constr_params.append({})
                        else:
                            #print(u_c)
                            raise ValueError('constraint not known')
                    elif type(u_c) == list:
                        if type(u_c[0]) == str and type(u_c[1] == dict):
                            if u_c[0] in AbstractDecon.CONSTRAINTS:
                                tmp_constr.append(u_c[0])
                                tmp_constr_params.append(u_c[1])
                            else:
                                #print(u_c)
                                raise ValueError('constraint not known')
                        else:
                            raise TypeError('wrong type for constraint casting')
                    else:
                        raise TypeError('wrong type for constraint casting')
        else:
            #print(constr)
            raise ValueError('\'constraints\' variable has to be of type \'list\' or \'str\'')	
        
            
            
            
        constr_params = []
        for ind in range(len(tmp_constr)):
            if tmp_constr[ind] == 'non_negative':
                constr_params.append({})
            elif tmp_constr[ind] == 'clipped':
                if tmp_constr_params[ind]:
                    constr_params.append({'vmin':self.img.min(), 'vmax':self.img.max()})
                else:
                    constr_params.append(tmp_constr_params[ind])
            elif tmp_constr[ind] == 'normalized_mean':
                if tmp_constr_params[ind]:
                    constr_params.append({'vmean':self.img.mean(), 'vstd':self.img.std()})
                else:
                    constr_params.append(tmp_constr_params[ind])
            elif tmp_constr[ind] == 'normalized_abs':
                constr_params.append({})
            elif tmp_constr[ind] == 'rescaled':
                if tmp_constr_params[ind]:
                    constr_params.append({'vmax': self.img.min(), 'vmin':self.img.max()})
                else:
                    constr_params.append(tmp_constr_params[ind])
            elif tmp_constr[ind] == 'energy_preserved':
                if tmp_constr_params[ind]:
                    constr_params.append({'energy': self.img.sum()})
                constr_params.append(tmp_constr_params[ind])

        self.constraints = tmp_constr
        self.constr_params = constr_params


    #----------------------------------------------------------------------
    def applyConstraint(self, receiv = None, constraints = None, **kwargs):
        """
        Apply given constraints to the 'receiv' array. 
        If 'receiv' is None the solvedFor array is used and the solution is stored in the current solution. Otherwise the constrained array is returned
        If 'constraint' is None the constraints defined at creation are applied.

        """
        if constraints is None:
            constraints = self.constraints
            no_con = True
        else:
            no_con = False

        if receiv is None:
            receiv = self.getCurrentSolution()
            no_arr = True
        else:
            no_arr = False

        if receiv.dtype == 'complex' or receiv.dtype == 'complex64':
            #raise ValueError('Application of constraints needs non-complex value array.')
            receiv_compl = receiv
            receiv = receiv.real

        if type(constraints) == list:
            for ind in range(len(constraints)):
                if no_con:
                    receiv = self._interpretConstraint(constraints[ind], receiv, **self.constr_params[ind])
                else:
                    receiv = self._interpretConstraint(constraints[ind], receiv, **kwargs)
                    
        elif type(constraints) == str:
            receiv = self._interpretConstraint(constraints, receiv, kwargs)

        if receiv.dtype == 'complex' or receiv.dtype == 'complex64':
            receiv_compl.real = receiv
            receiv = receiv_compl

        if no_arr:
            self.setCurrentSolution(receiv)
            
        else:
            return receiv

    #----------------------------------------------------------------------
    def _interpretConstraint(self, constr, arr, **kwargs):
        """"""
        if not constr in AbstractDecon.CONSTRAINTS:
            raise ValueError
        elif constr == 'non_negative':
            return AbstractDecon._applyNonnegativity(arr)
        elif constr == 'clipped':
            if 'vmin' in kwargs.keys(): vmin = kwargs['vmin']
            else: vmin = self.img.min()
            if 'vmax' in kwargs.keys(): vmax = kwargs['vmax']
            else: vmax = self.img.max()
            return AbstractDecon._applyClipped(arr, vmin, vmax)
        elif constr == 'normalized_mean':
            if 'vmean' in kwargs.keys(): vmean = kwargs['vmean']
            else: vmean = 0.
            if 'vstd' in kwargs.keys(): vstd = kwargs['vstd']
            else: vstd = 1.
            return AbstractDecon._applyNormalizedMean(arr, vmean, vstd)
        elif constr == 'normalized_abs':
            return AbstractDecon._applyNormalizedAbs(arr)
        elif constr == 'rescaled':
            if 'vmax' in kwargs.keys(): vmax = kwargs['vmax']
            else: vmax = self.img.max()
            if 'vmin' in kwargs.keys(): vmin = kwargs['vmin']
            else: vmin = self.img.min()
            return AbstractDecon._applyRescaled(arr, vmax, vmin)
        elif constr == 'energy_preserved':
            if 'energy' in kwargs.keys(): energy = kwargs['energy']
            else: energy = self.img.sum()
            return AbstractDecon._applyEnergyPreservation(arr, energy)



    #----------------------------------------------------------------------
    @staticmethod
    def _applyNonnegativity(arr):
        """"""
        ind = arr < 0
        if ind.any():
            arr[ind] = 0
        return arr

    #----------------------------------------------------------------------
    @staticmethod
    def _applyClipped(arr, vmin, vmax):
        """"""
        ind1 = arr < vmin
        ind2 = arr > vmax
        if ind1.any():
            arr[ind1] = vmin
        if ind2.any():
            arr[ind2] = vmax
        return arr

    #----------------------------------------------------------------------
    @staticmethod
    def _applyNormalizedMean(arr, mean, std):
        """"""
        c_mean = arr.mean()
        c_std = arr.std()

        return std*((arr - c_mean)/c_std)+mean

    #----------------------------------------------------------------------
    @staticmethod
    def _applyNormalizedAbs(arr):
        """"""
        arr = arr - arr.min()
        arr = arr / arr.sum()
        return arr

    #----------------------------------------------------------------------
    @staticmethod
    def _applyRescaled(arr, vmax, vmin):
        """"""
        c_max = arr.max()
        c_min = arr.min()
        scale = (vmax-vmin)/(c_max-c_min)
        return scale*(arr - c_min) + vmin

    #----------------------------------------------------------------------
    @staticmethod
    def _applyEnergyPreservation(arr, energy):
        """"""
        c_energy = arr.sum()
        scale = energy/c_energy
        return scale*arr
    
    
    #----------------------------------------------------------------------
    def getAlgoParameters(self):
        """Function has to be implemented by all reconstruction algorithms."""
        raise NotImplementedError
        

    #----------------------------------------------------------------------
    def getConstraints(self):
        """"""
        ret = []
        for ind in range(len(self.constraints)):
            ret.append([self.constraints[ind], self.constr_params[ind]])
        if len(ret) == 0:
            ret.append(['No Constraints',{'NoParams':'None'}])
        return ret
        

    #----------------------------------------------------------------------
    def getCurrentSolution(self, forSaving = False):
        """"""
        if not forSaving:
            if self.solveFor == 'psf':
                return self.psf
            else:
                return self.sample
        else:
            if self.solveFor == 'psf':
                tmp = self.psf.copy()
            else:
                tmp = self.sample.copy()
                
            max_v = tmp.max()
            #print(tmp.min())
            #print(tmp.max())
            tmp = tmp/max_v*255
            tmp = tmp.round()
            if tmp.max() > 255:
                raise ValueError
            tmp = tmp.astype('uint8')
            return tmp

    #----------------------------------------------------------------------
    def setCurrentSolution(self, arr):
        """"""
        if self.solveFor == 'psf':
            self.psf = arr.astype(self.psf.dtype)
        elif self.solveFor == 'sample':
            self.sample = arr.astype(self.sample.dtype)
        else:
            raise ValueError


    #----------------------------------------------------------------------
    def saveSolution(self, meta_only = False):
        """"""
        
        meta = self.getMetaData()
        
        util.stack_loader.write_image_stack(self.getCurrentSolution(forSaving=True), 
                                            self.save_path, 
                                            self.save_fident, 
                                            0, 
                                            meta_data=meta,
                                            meta_only=meta_only)
    
    
    #----------------------------------------------------------------------
    def getMetaData(self):
        """
        Image Data:
        -	Image type: reconstruction (psf or sample)
        -	data type
        -	image size
        -	save path
        -	save file identifier
        -	max value after rescaling
        -	PSF or sample reconstruction
        
        
        Reconstruction Data:
        -	used algorithm
        -	algorithm parameters
        -	errortype (diff or groundtruth)
        -	takenTimes
        -	applied constraints
        -	fourier transform type
        
        
        Information about input arrays:
        
        Original image:
        -	image size
        -	file path
        -	file identifier
        -	image type (synthetic, experimental)
        -	additional information about origin
        -	max value after rescaling
        
        PSF image:
        -	image size
        -	file path
        -	file identifier

        """
        self.meta.loadFromReconAlgo(self)
        
        return self.meta.toList()
        
    
    #----------------------------------------------------------------------
    def initSaveParameters(self, save_path, save_fident, orig_img_path = [None, None], orig_psf_path = [None,None], orig_sample_path = [None,None], orig_truth_path = [None,None], overwrite = False):
        """"""
        
        self.save_fident = save_fident
        self.overwrite = overwrite
        
        util.createAllPaths(save_path)
        
        if not util.isEmptyDir(save_path) and not overwrite:
            raise ValueError('Save destination is not empty and overwriting is disabled.')
        else:
            self.save_path = save_path
            
        if orig_img_path[0] is None:
            self.orig_img_path = 'NA'
            self.orig_img_fident = 'NA'
        elif util.checkAllPaths(orig_img_path[0]):
            self.orig_img_path = orig_img_path[0]
            self.orig_img_fident = orig_img_path[1]
        else:
            raise ValueError('Given path for original image is not valid.')

        if orig_psf_path[0] is None:
            self.orig_psf_path = 'NA'
            self.orig_psf_fident = 'NA'        
        elif util.checkAllPaths(orig_psf_path[0]):
            self.orig_psf_path = orig_psf_path[0]
            self.orig_psf_fident = orig_psf_path[1]
        else:
            raise ValueError('Given path for original psf is not valid.')        
        
        if orig_sample_path[0] is None:
            self.orig_sample_path = 'NA'
            self.orig_sample_fident = 'NA'        
        elif util.checkAllPaths(orig_sample_path[0]):
            self.orig_sample_path = orig_sample_path[0]
            self.orig_sample_fident = orig_sample_path[1]
        else:
            raise ValueError('Given path for original sample is not valid.')        
        
        if orig_truth_path[0] is None:
            self.orig_truth_path = 'NA'
            self.orig_truth_fident = 'NA'        
        elif util.checkAllPaths(orig_truth_path[0]):
            self.orig_truth_path = orig_truth_path[0]
            self.orig_truth_fident = orig_truth_path[1]
        else:
            raise ValueError('Given path for ground truth is not valid.')        
        