import numpy as np
import numpy.random as npra
import numpy.linalg as npla
import scipy.linalg as splin
import scipy.stats as scista
import util
import util.stack_loader 
import datetime as dt


########################################################################
class Convoluter(object):
    """"""

    REAL_FFT = 0
    CPX_FFT = 1
    LOOP = 2
    MATRIX = 3
    CONV_METHODS = {'real_fft':REAL_FFT, 'cpx_fft':CPX_FFT, 'loop':LOOP, 'matrix':MATRIX}
    CONV_METHODS_INV = {v:k for k,v in CONV_METHODS.items()}
    
    GAUSSIAN = 0
    LORENTZIAN = 1
    POISSON = 2
    NOISE_TYPES = {'gaussian':GAUSSIAN, 'lorentzian':LORENTZIAN, 'poisson':POISSON, 'no_noise': -1}
    NOISE_TYPES_INV = {v:k for k,v in NOISE_TYPES.items()}
    #----------------------------------------------------------------------
    def __init__(self,
                 image,
                 psf,
                 isPsfCentered = True,
                 conv_method = 'real_fft',
                 noise_type = 'no_noise',
                 noise_params = None,
                 debug_int = 0,
                 comment = ''):
        """Constructor"""
        
        self.img = image.astype('float')
        self.psf = psf.astype('float')
        
        
        if type(conv_method) == str and conv_method in Convoluter.CONV_METHODS:
            self.conv_method = Convoluter.CONV_METHODS[conv_method]
        elif type(conv_method) == int and conv_method in Convoluter.CONV_METHODS.values():
            self.conv_method = conv_method
        else:
            raise ValueError
        
        if type(noise_type) == str and noise_type in Convoluter.NOISE_TYPES:
            self.noise_type = Convoluter.NOISE_TYPES[noise_type]
            self.noise_params = noise_params
        elif type(noise_type) == int and noise_type in Convoluter.NOISE_TYPES.values():
            self.noise_type = noise_type
            self.noise_params = noise_params
        else:
            raise ValueError        
        
        self.isPsfCentered = isPsfCentered
        self.debugInt = debug_int    
    
        self.meta = util.stack_loader.MetaData()

        self.out = np.zeros(image.shape, dtype='float')
        
        
        self.timestring = ''
        self.comment = comment
        
        self.save_fident = None
        self.save_path = None
        self.orig_img_path = None
        self.orig_psf_path = None        
        
        
        
        
    #----------------------------------------------------------------------
    def convolute(self):
        """"""
        
        if self.conv_method == Convoluter.REAL_FFT:
            self._conv_fft_real()
        elif self.conv_method == Convoluter.CPX_FFT:
            self._conv_fft_cpx()
        elif self.conv_method == Convoluter.MATRIX:
            self._conv_matrix()
        elif self.conv_method == Convoluter.LOOP:
            self._conv_loop()
        
        self.out = (self.img.max()/self.out.max()) * self.out
        
        self.timestring = dt.datetime.now().strftime('%Y-%m-%d %H:%M')              
        
    #----------------------------------------------------------------------
    def _conv_fft_real(self):
        """"""
        self.print_dbg('Starting Convolution process...', 1)
        
        safe_shape = [i+p/2 for i,p in zip(self.img.shape, self.psf.shape)]
        self.print_dbg('Minimal size in Fourier Space: {}'.format(safe_shape), 3)
        self.print_dbg('Zero-Padding Image...', 3)
        img = self._padArray(self.img, safe_shape)
        
        self.print_dbg('Zero-Padding PSF...', 3)
        psf = self._padArray(self.psf, safe_shape)
        
        norm_arr = np.ones(self.img.shape)
        norm_arr = self._padArray(norm_arr, safe_shape)
        
        if self.isPsfCentered:
            self.print_dbg('Transforming shifted PSF to Fourier-Domain...', 3)
            f_psf = np.fft.rfftn(np.fft.ifftshift(psf))
            
        else:
            self.print_dbg('Transforming PSF to Fourier-Domain...', 3)
            f_psf = np.fft.rfftn(psf)
            
            
        self.print_dbg('Transforming Image to Fourier-Domain...', 3)
        f_img = np.fft.rfftn(img)
        f_norm = np.fft.rfftn(norm_arr)
        
        self.print_dbg('Performing Multiplication in Fourier space and performing inverse Fourier Transform...',3)
        out = np.fft.irfftn(f_img * f_psf, safe_shape)
        norm = np.fft.irfftn(f_norm * f_psf, safe_shape)
        #self.out = self.out / norm
        self.print_dbg('Cropping image to original size...', 3)
        
        shape_diff = [sa_sh - i for sa_sh,i in zip(safe_shape, self.img.shape)]
        slices = [slice(int(sh_di/2), int(sh_di/2)+im_s, 1) for sh_di, im_s in zip(shape_diff, self.img.shape)]
        
        self.out = out[slices] / norm[slices]
        
        self.print_dbg('Convolution Done.', 1)
        
        
    #----------------------------------------------------------------------
    def _conv_fft_cpx(self):
        """"""
        self.print_dbg('Starting Convolution process...', 1)
        
        safe_shape = [i+p/2 for i,p in zip(self.img.shape, self.psf.shape)]
        self.print_dbg('Minimal size in Fourier Space: {}'.format(safe_shape), 3)
        self.print_dbg('Zero-Padding Image...', 3)
        img = self._padArray(self.img, safe_shape)
        
        norm_arr = np.ones(self.img.shape)
        norm_arr = self._padArray(norm_arr, safe_shape)        
        
        self.print_dbg('Zero-Padding PSF...', 3)
        psf = self._padArray(self.psf, safe_shape)
        
        if self.isPsfCentered:
            self.print_dbg('Transforming shifted PSF to Fourier-Domain...', 3)            
            f_psf = np.fft.fftn(np.fft.ifftshift(psf))
        else:
            self.print_dbg('Transforming PSF to Fourier-Domain...', 3)
            f_psf = np.fft.fftn(psf)
            
        self.print_dbg('Transforming Image to Fourier-Domain...', 3)
        f_img = np.fft.fftn(img)
        f_norm = np.fft.fftn(norm_arr)
        
        self.print_dbg('Performing Multiplication in Fourier space and performing inverse Fourier Transform...',3)
        out = np.fft.ifftn(f_img * f_psf).real
        norm = np.fft.ifftn(f_norm * f_psf).real
        
        self.print_dbg('Cropping image to original size...', 3)
        shape_diff = [sa_sh - i for sa_sh,i in zip(safe_shape, self.img.shape)]
        slices = [slice(int(sh_di/2), int(sh_di/2)+im_s, 1) for sh_di, im_s in zip(shape_diff, self.img.shape)]
        self.out = out[slices]/norm[slices]
        
        self.print_dbg('Convolution Done.', 1)
        
        
    #----------------------------------------------------------------------
    def _conv_matrix(self):
        """"""
        
        
        img_flat = self.img.flatten(order = 'C')            
        psf_flat = self._padArray(self.psf, self.img.shape).flatten(order = 'C')
        
        m = img_flat.size
        
        psf_flat = np.pad(psf_flat, (0,m-1), mode='constant', constant_values=0)
        
        psf_toeplitz = splin.toeplitz(psf_flat, r=np.zeros(m))        
        
        out_vec = np.zeros(img_flat.shape)
        ref_vec = np.ones(self.inp_flat.shape)
        
        out_vec = psf_toeplitz.dot(img_flat)
        ref_vec = psf_toeplitz.dot(ref_vec)
        out_vec = ref_conv.max()*out_vec/ref_vec #???
        
        mid_element = (out_vec.size-1)/2
        if self.inp.size % 2 == 1:
            out_vec = out_vec[mid_element-mid_element/2:mid_element+mid_element/2+1]
            #out_vec = out_vec[mid_element+mid_element/2+1:mid_element-mid_element/2:-1]
            out_vec.shape = self.inp.shape
            self.out = out_vec
        else:
            out_vec = out_vec[mid_element-mid_element/2-1:mid_element+mid_element/2+1]
            #out_vec = out_vec[mid_element+mid_element/2+1:mid_element-mid_element/2-1:-1]
            out_vec.shape = self.inp.shape
            self.out = np.fliplr(out_vec)        
            
            
            
    #----------------------------------------------------------------------
    def _conv_loop(self):
        """"""
        img_shape = self.img.shape
        psf_shape = [i/2 for i in self.psf.shape]
        out_temp = np.zeros(self.inp.shape)
        temp_add = np.zeros(self.psf.shape)
        
        
        
        for ix in range(img_shape[0]):
            for iy in range(img_shape[1]):
                for iz in range(img_shape[2]):
                    
                    ri_img = [[max(ix-psf_shape[0], 0), min(ix+psf_shape[0]+1, img_shape[0])],
                              [max(iy-psf_shape[1], 0), min(iy+psf_shape[1]+1, img_shape[1])],
                              [max(iz-psf_shape[2], 0), min(iz+psf_shape[2]+1, img_shape[2])]
                             ]

                    
                    ri_psf = [[max(0, psf_shape[0]-ix), min(2*psf_shape[0]+1, psf_shape[0]+img_shape[0]-ix)],
                              [max(0, psf_shape[1]-iy), min(2*psf_shape[1]+1, psf_shape[1]+img_shape[1]-iy)],
                              [max(0, psf_shape[2]-iz), min(2*psf_shape[2]+1, psf_shape[2]+img_shape[2]-iz)]
                             ]                    
                    
                    temp_add[ri_psf[0][0]:ri_psf[0][1], ri_psf[1][0]:ri_psf[1][1], ri_psf[2][0]:ri_psf[2][1]] = self.inp[ix,iy,iz]*self.psf[ri_psf[0][0]:ri_psf[0][1], ri_psf[1][0]:ri_psf[1][1], ri_psf[2][0]:ri_psf[2][1]]
                    out_temp[ri_img[0][0]:ri_img[0][1], ri_img[1][0]:ri_img[1][1], ri_img[2][0]:ri_img[2][1]] += temp_add[ri_psf[0][0]:ri_psf[0][1], ri_psf[1][0]:ri_psf[1][1], ri_psf[2][0]:ri_psf[2][1]]
                    
        self.out = out_temp        
        
    #----------------------------------------------------------------------
    def print_dbg(self, string, dbg):
        """"""
        if dbg <= self.debugInt:
            print string        
            
            
    #----------------------------------------------------------------------
    def _padArray(self,arr, shape):
        """
        Pads the given array to the given shape.
        Dtype of the original array is preserved. 
        """
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
                self.print_dbg('Padding dimension {} from {} to {}'.format(it, arr.shape[it], shape[it]), 2)
                self.print_dbg('Warning: asymmetrical padding of Array. Previously centered features are no longer centered.', 0)
                
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
    def addNoise(self):
        """"""
        
        if self.noise_type == Convoluter.GAUSSIAN:
            self._addGaussian()
        elif self.noise_type == Convoluter.LORENTZIAN:
            self._addLorentzian()
        elif self.noise_type == Convoluter.POISSON:
            self._addPoisson()
        elif self.noise_type == -1:
            raise ValueError('Called addNoise method without defining noise parameters first')
        else:
            raise ValueError

    
    #----------------------------------------------------------------------
    def _addGaussian(self):
        """"""
        arr_mean = self.out.mean()
        if self.noise_params is None:
            mean = 0.
            dev = 0.1 * arr_mean
        else:
            if 'mean' in self.noise_params.keys():
                typ, val = self.noise_params['mean'] 
                if typ == 'rel':
                    mean = val*arr_mean
                elif typ == 'abs':
                    mean = val
                else:
                    raise ValueError
            else:
                mean = 0.
                
            if 'dev' in self.noise_params.keys():
                typ, val = self.noise_params['dev']
                if typ == 'rel':
                    dev = val*arr_mean
                elif typ == 'abs':
                    dev = val
                else:
                    raise ValueError
            else:
                dev = 0.1 * arr_mean
            
        noise = npra.normal(mean, dev, self.out.shape)
        
        self.noise_params = {'mean':mean, 'dev':dev}
        self.out += noise
        
    
    #----------------------------------------------------------------------
    def _addLorentzian(self):
        """"""
        raise NotImplementedError
    #----------------------------------------------------------------------
    def _addPoisson(self):
        """"""
        raise NotImplementedError
    
    
    #----------------------------------------------------------------------
    def initSaveParameters(self, save_path, save_fident, orig_img_path = [None, None], orig_psf_path = [None,None], overwrite = False):
        """"""
        
        self.save_fident = save_fident
        self.overwrite = overwrite
        
        util.createAllPaths(save_path)
        
        if not util.isEmptyDir(save_path) and not overwrite:
            raise ValueError('Save destination is not empty and overwriting is disabled.')
        else:
            self.save_path = save_path
            
        if util.checkAllPaths(orig_img_path[0]):
            self.orig_img_path = orig_img_path[0]
            self.orig_img_fident = orig_img_path[1]
        else:
            raise ValueError('Given path for original image is not valid.')

        if util.checkAllPaths(orig_psf_path):
            self.orig_psf_path = orig_psf_path[0]
            self.orig_psf_fident = orig_psf_path[1]
        else:
            raise ValueError('Given path for original psf is not valid.')        
        
        
    

    #----------------------------------------------------------------------
    def saveSolution(self):
        """"""
        self.meta.loadFromConvoluter(self)
        meta = self.meta.toList()
        
        
        tmp = self.out.copy()
        max_v = tmp.max()
        tmp = tmp/max_v*255
        tmp = tmp.round()
        if tmp.max() > 255:
            raise ValueError
        tmp = tmp.astype('uint8')
            
        util.stack_loader.write_image_stack(tmp, 
                                            self.save_path, 
                                            self.save_fident, 
                                            0, 
                                            meta_data=meta)
        
        
########################################################################
class Noise_Adder(object):
    """"""
    GAUSSIAN = 0
    LORENTZIAN = 1
    POISSON = 2
    NOISE_TYPES = {'gaussian':GAUSSIAN, 'lorentzian':LORENTZIAN, 'poisson':POISSON, 'no_noise': -1}
    NOISE_TYPES_INV = {v:k for k,v in NOISE_TYPES.items()}
    
    DEF_PLAIN = 0
    DEF_SNR = 1
    DEF_CNR = 2
    NOISE_DEF = {'plain':DEF_PLAIN, 'snr':DEF_SNR, 'cnr':DEF_CNR}
    NOISE_DEF_INV = {v:k for k,v in NOISE_TYPES.items()}


    #----------------------------------------------------------------------
    def __init__(self,
                 image,
                 old_meta = None,
                 img_type = 'sample',
                 noise_type = 'gaussian',
                 noise_params = None,
                 debug_int = 0,
                 comment = ''):
        """Constructor"""    
        
        self.img = image.astype('float')
        
        if type(noise_type) == str and noise_type in Noise_Adder.NOISE_TYPES:
            self.noise_type = Noise_Adder.NOISE_TYPES[noise_type]
            self.noise_params = noise_params
        elif type(noise_type) == int and noise_type in Noise_Adder.NOISE_TYPES.values():
            self.noise_type = noise_type
            self.noise_params = noise_params
        else:
            raise ValueError      
        
        if old_meta:
            self.old_meta = old_meta
        else:
            self.old_meta = False
        
        self.debugInt = debug_int    
        
        if img_type in ['sample', 'psf']:
            self.img_type = img_type
        self.meta = util.stack_loader.MetaData()

        self.out = np.zeros(image.shape, dtype='float')
        
        
        self.timestring = ''
        self.comment = comment
        
        self._defNoiseParams()
        
    
    #----------------------------------------------------------------------
    def _defNoiseParams(self):
        """
        Possibilities for Noise definition:
        - Direct definition by noise parameters (e.g. gaussian deviation)
        - Definition by Signal-to-Noise: SNR = mu_signal/sigma_signal, mu_signal is the average of the signal, sigma_signal is the noise standard deviation
           - Supplying of wanted SNR value
           - optional: supplying of averaging signal value to use
           - Output: noise parameters to fit the other parameters
        - Definition by Contrast-to-Noise: CNR = |Signal_A - Signal_B|/sigma
           - Supplying of wanted CNR value
           - supplying of BG value
           - optional: supplying of averaging signal value
        
        """
        
        
        para = self.noise_params
        para_k = self.noise_params.keys()
        
        if 'noise_def' in para_k:
            noi_def = para['noise_def']
            if noi_def == Noise_Adder.DEF_PLAIN:
                if 'sigma' in para_k:
                    sigma = para['sigma']
                else:
                    raise ValueError
                
            elif noi_def == Noise_Adder.DEF_CNR:
                if 'cnr' in para_k:
                    cnr = para['cnr']
                else:
                    raise ValueError
                
                if 'bg' in para_k and 'avg' in para_k:
                    bg = para['bg']
                else:
                    max_v = self.img.max()
                    min_v = self.img.min()
                    avg_v = self.img.mean()
                    
                    high = self.img[self.img >= avg_v].mean()
                    
                    #High Quartil and Rest
                    bg = self.img[self.img >= high].mean()
                    avg = self.img[self.img <= high].mean()
                    
                
                if 'avg' in para_k:
                    avg = para['avg']
                if 'bg' in para_k:
                    bg = para['bg']
                
                sigma = abs(avg - bg)/cnr
                
            elif noi_def == Noise_Adder.DEF_SNR:
                if 'snr' in para_k:
                    snr = para['snr']
                else:
                    raise ValueError
                
                if 'avg' in para_k:
                    avg = para['avg']
                else:
                    avg = self.img.mean()
                
                sigma = avg/snr
                
            para['sigma'] = sigma
            
            
        elif 'sigma' in para_k:
            para['sigma'] = float(para['sigma'])
        else:
            raise ValueError
        
        if 'mean' in para_k:
            para['mean'] = float(para['mean'])
        else:
            para['mean'] = 0.
        
        if self.noise_type == Noise_Adder.GAUSSIAN:
            pass
        elif self.noise_type == Noise_Adder.LORENTZIAN:
            raise NotImplementedError
        elif self.noise_type == Noise_Adder.POISSON:
            raise NotImplementedError
        
            
        
        
            
        
    #----------------------------------------------------------------------
    def addNoise(self):
        """"""
        
        if self.noise_type == Convoluter.GAUSSIAN:
            self._addGaussian()
        elif self.noise_type == Convoluter.LORENTZIAN:
            self._addLorentzian()
        elif self.noise_type == Convoluter.POISSON:
            self._addPoisson()
        elif self.noise_type == -1:
            raise ValueError('Called addNoise method without defining noise parameters first')
        else:
            raise ValueError

    
    #----------------------------------------------------------------------
    def _addGaussian(self):
        """"""
        st_dev = self.noise_params['sigma']
        mean = self.noise_params['mean']
        
        noise = npra.normal(mean, st_dev, self.img.shape)
        
        self.print_dbg('Adding Gaussian Noise with StDev = {} and Mean = {}'.format(st_dev, mean), 2)
        
        self.out = self.img + noise
        
        
    #----------------------------------------------------------------------
    def _addLorentzian(self):
        """"""
        raise NotImplementedError
    #----------------------------------------------------------------------
    def _addPoisson(self):
        """"""
        raise NotImplementedError    
        
        
    #----------------------------------------------------------------------
    def print_dbg(self, string, dbg):
        """"""
        if dbg <= self.debugInt:
            print string                
        
        
    #----------------------------------------------------------------------
    def initSaveParameters(self, save_path, save_fident, orig_img_path = [None, None], overwrite = False):
        """"""
        
        self.save_fident = save_fident
        self.overwrite = overwrite
        
        util.createAllPaths(save_path)
        
        if not util.isEmptyDir(save_path) and not overwrite:
            raise ValueError('Save destination is not empty and overwriting is disabled.')
        else:
            self.save_path = save_path
            
        if util.checkAllPaths(orig_img_path[0]):
            self.orig_img_path = orig_img_path[0]
            self.orig_img_fident = orig_img_path[1]
        else:
            raise ValueError('Given path for original image is not valid.')

    
        
        
        
        
    #----------------------------------------------------------------------
    def saveSolution(self):
        """"""
        self.meta.loadFromNoiseAdder(self)
        meta = self.meta.toList()
        
        
        tmp = self.out.copy()
        max_v = tmp.max()
        tmp = tmp/max_v*255
        tmp = tmp.round()
        if tmp.max() > 255:
            raise ValueError
        tmp = tmp.astype('uint8')
            
        util.stack_loader.write_image_stack(tmp, 
                                            self.save_path, 
                                            self.save_fident, 
                                            0, 
                                            meta_data=meta)    