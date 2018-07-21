import math as mt
import numpy as np
import numpy.random as np_rand

import util
import util.stack_loader

import datetime as dt

########################################################################
class PSF_Generator(object):
    """"""

    PSF_TYPES = {'positive':0, 'negative':1}
    PSF_TYPES_REV = {v:k for k,v in PSF_TYPES.items()}
    PSF_MODELS = {'richard_wolf':0, 'gibson_lanni':1, 'born_wolf':2, 'defocus':3, 'gaussian':4, 'gaussian_simple':5}
    PSF_MODELS_REV = {v:k for k,v in PSF_MODELS.items()}
    PSF_ODDITY = {'auto':0, 'odd':1, 'even':2}
    #----------------------------------------------------------------------
    def __init__(self, size, resolution, psf_type = 'positive', psf_model = 'defocus', psf_oddity = 'auto', psf_params = {} , name = 'TestPSF', comment = ['']):
        """Constructor"""
        
        self.name = name
        self.comment = comment
        self.timestring = ''
        self.meta = util.stack_loader.MetaData()
        self.psf_params = []
        
        if type(size) == int or type(size) == float:
            self.size = [float(size),float(size),float(size)]
        elif type(size) == np.ndarray and size.size == 3:
            self.size = np.zeros((3), dtype = 'float')
            self.size[:] = size[:]
        elif type(size) == list and len(size) == 3:
            size = [float(it) for it in size]
            self.size = np.array(size)        
        else:
            raise ValueError
            
        if type(resolution) == int or type(resolution) == float:
            self.res = [float(resolution), float(resolution), float(resolution)]
        elif type(resolution) == np.ndarray and resolution.size == 3:
            self.res = np.zeros((3))
            self.res[:] = resolution[:]
        elif type(resolution) == list and len(size) == 3:
            res = [float(it) for it in resolution ]
            self.res = np.array(res)
        else:
            raise ValueError            
        
        
        self.psf_type = None
        if psf_type in PSF_Generator.PSF_TYPES.keys():
            self.psf_type = PSF_Generator.PSF_TYPES[psf_type]
        elif psf_type in PSF_Generator.PSF_TYPES.values():
            self.psf_type = psf_type
        else:
            raise ValueError
        
        self.psf_model = None
        if psf_model in PSF_Generator.PSF_MODELS.keys():
            self.psf_model = PSF_Generator.PSF_MODELS[psf_model]
        elif psf_model in PSF_Generator.PSF_MODELS.values():
            self.psf_model = psf_model
        else:
            raise ValueError
        
        self.psf_oddity = None
        if psf_oddity in PSF_Generator.PSF_ODDITY.keys():
            self.psf_oddity = PSF_Generator.PSF_ODDITY[psf_oddity]
        elif psf_oddity in PSF_Generator.PSF_ODDITY.values():
            self.psf_oddity = psf_oddity
        else:
            raise ValueError
        
        self._calc_shape()
        
        
        self.setPSFParams(psf_params)
        self.noise_type = None
            
        self.out = np.zeros(self.shape)
        
        self.save_fident = None
        self.save_path = None
        
    
    
    #----------------------------------------------------------------------
    def _calc_shape(self):
        """"""
        
        if self.psf_oddity == PSF_Generator.PSF_ODDITY['auto']:
            self.psf_oddity = PSF_Generator.PSF_ODDITY['odd']
            
        if self.psf_oddity == PSF_Generator.PSF_ODDITY['odd']:
            i_center = [int(siz/res) for siz,res in zip(self.size, self.res)]
            shape = [1+2* i_c for i_c in i_center]
            self.size = [res*i_c for i_c,res in zip(i_center, self.res)]
        elif self.psf_oddity == PSF_Generator.PSF_ODDITY['even']:
            i_center = [int((siz-0.5*res)/res) for siz,res in zip(self.size, self.res)]
            shape = [2 + 2*i_c for i_c in i_center]
            self.size = [res*(0.5+i_c) for i_c,res in zip(i_center, self.res)]
        else:
            raise ValueError
        
        self.i_center = i_center
        self.shape = shape
        
        
    #----------------------------------------------------------------------
    def setPSFParams(self, psf_params):
        """"""
        if not type(psf_params) == dict:
            raise ValueError
        
        keys = psf_params.keys()
        para_dict = {}
        if self.psf_model == PSF_Generator.PSF_MODELS['richard_wolf']:
            raise NotImplementedError
        elif self.psf_model == PSF_Generator.PSF_MODELS['gibson_lanni']:
            raise NotImplementedError
        elif self.psf_model == PSF_Generator.PSF_MODELS['born_wolf']:
            raise NotImplementedError
        elif self.psf_model == PSF_Generator.PSF_MODELS['defocus']:
            #sigma(z) = sigma * (z - zi), linear dependence of sigma in respect to z (coordinate along z-axis)
            #needed parameters:
            #sigma: sigma at z=zi
            #zi: z_coordinate of focus point
            #K: normalization parameter
            if 'sigma' in keys:
                para_dict['sigma'] = float(psf_params['sigma'])
            else:
                para_dict['sigma'] = mt.sqrt(3)
            if 'zi' in keys:
                para_dict['zi'] = float(psf_params['zi'])
            else:
                para_dict['zi'] = 2000.
            if 'K' in keys:
                para_dict['K'] = float(psf_params['K'])
            else:
                para_dict['K'] = 275.
            
        elif self.psf_model == PSF_Generator.PSF_MODELS['gaussian']:
            #sigma(z) = sigma_0 + sigma_1 * z + sigma_2 * z**2
            #needed parameters:
            #sigma_0
            #sigma_1
            #sigma_2
            if 'sigma_0' in keys:
                para_dict['sigma_0'] = float(psf_params['sigma_0'])
            else:
                para_dict['sigma_0'] = 1.
            if 'sigma_1' in keys:
                para_dict['sigma_1'] = float(psf_params['sigma_1'])
            else:
                para_dict['sigma_1'] = 0.
            if 'sigma_2' in keys:
                para_dict['sigma_2'] = float(psf_params['sigma_2'])
            else:
                para_dict['sigma_2'] = 0.2
            
        elif self.psf_model == PSF_Generator.PSF_MODELS['gaussian_simple']:
            #simple 3D gaussian. only one parameter needed
            if 'sigma' in keys:
                para_dict['sigma'] = float(psf_params['sigma'])
            else:
                para_dict['sigma'] = 1.
        else:
            raise ValueError
        
        self.psf_params = para_dict
        
    ##----------------------------------------------------------------------
    #def createPSF(self):
        #""""""
        
        #if self.psf_model == PSF_Generator.PSF_MODELS['richard_wolf']:
            #self._richard_wolf()
        #elif self.psf_model == PSF_Generator.PSF_MODELS['gibson_lanni']:
            #self._gibson_lanni()
        #elif self.psf_model == PSF_Generator.PSF_MODELS['born_wolf']:
            #self._born_wolf()
        #elif self.psf_model == PSF_Generator.PSF_MODELS['defocus']:
            #self._defocus()
        #elif self.psf_model == PSF_Generator.PSF_MODELS['gaussian']:
            #self._gaussian()
        #elif self.psf_model == PSF_Generator.PSF_MODELS['gaussian_simple']:
            #self._gaussian_simple()
        #else:
            #raise ValueError
        
        #self.timestring = dt.datetime.now().strftime('%Y-%m-%d %H:%M')              
        #self.out = self.out / self.out.sum()
        
    #----------------------------------------------------------------------
    def createPSF(self, oversampling = 1):
        """"""
        
        if oversampling == 1:
            pass
        elif 1 < oversampling <= 5:
            end_res = self.res
            end_shape = self.shape
            end_size = self.size            

            cur_res = [r/float(oversampling) for r in end_res]
            if self.psf_oddity == 1:
                if oversampling == 2:
                    cur_shape = [esh*2+1 for esh in end_shape]
                elif oversampling == 3:
                    cur_shape = [3*esh for esh in end_shape]
                elif oversampling == 4:
                    cur_shape = [4*esh+1 for esh in end_shape]
                elif oversampling == 5:
                    cur_shape = [5*esh for esh in end_shape]                
            else:
                if oversampling == 2:
                    cur_shape = [2*esh+1 for esh in end_shape]
                elif oversampling == 3:
                    cur_shape = [3*esh+1 for esh in end_shape]
                elif oversampling == 4:
                    cur_shape = [4*esh+1 for esh in end_shape]
                elif oversampling == 5:
                    cur_shape = [5*esh+1 for esh in end_shape]
            cur_size = [cr*(cs-1)/2 for cs,cr in zip(cur_shape,cur_res)]
            cur_out = np.zeros(cur_shape)    
                
            print('End Shape: {}, End Resolution: {}, End Size: {}'.format(end_shape, end_res, end_size))
            print('Cur Shape: {}, Cur Resolution: {}, Cur Size: {}'.format(cur_shape, cur_res, cur_size))
            self.res = cur_res
            self.shape = cur_shape
            self.size = cur_size
            self.out = cur_out            
            
        else:
            raise ValueError('oversampling parameters has to be between 1 and 4. was {}'.format(oversampling))
            
        
        if self.psf_model == PSF_Generator.PSF_MODELS['richard_wolf']:
            self._richard_wolf()
        elif self.psf_model == PSF_Generator.PSF_MODELS['gibson_lanni']:
            self._gibson_lanni()
        elif self.psf_model == PSF_Generator.PSF_MODELS['born_wolf']:
            self._born_wolf()
        elif self.psf_model == PSF_Generator.PSF_MODELS['defocus']:
            self._defocus()
        elif self.psf_model == PSF_Generator.PSF_MODELS['gaussian']:
            self._gaussian()
        elif self.psf_model == PSF_Generator.PSF_MODELS['gaussian_simple']:
            self._gaussian_simple()
        else:
            raise ValueError        
        
        if oversampling > 1:
            arr_hyb = self.out.copy()
        
        
            if self.psf_oddity == 1:

                if oversampling == 2:
                    for dim in range(3):
                        slic_ind1 = [slice(0,-1,2) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                        slic_ind2 = [slice(1,None,2) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                        slic_ind3 = [slice(2,None,2) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                        arr_hyb = 0.25 * arr_hyb[slic_ind1]+ 0.5 *  arr_hyb[slic_ind2] + 0.25 * arr_hyb[slic_ind3]
                    self.comment = 'Two times oversampled'
                    
                        
                elif oversampling == 3:
                    for dim in range(3):
                        slic_ind1 = [slice(0,-2,3) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                        slic_ind2 = [slice(1,None,3) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                        slic_ind3 = [slice(2,None,3) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                        arr_hyb = (arr_hyb[slic_ind1] + arr_hyb[slic_ind2] + arr_hyb[slic_ind3])/3.
                    self.comment = 'Three times oversampled'
                        
                elif oversampling == 4:
                    
                    for dim in range(3):
                        slic_ind1 = [slice(0,-3,4) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                        slic_ind2 = [slice(1,None,4) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                        slic_ind3 = [slice(2,None,4) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                        slic_ind4 = [slice(3,None,4) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]                        
                        slic_ind5 = [slice(4,None,4) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]                        
                        arr_hyb = 0.25*(0.5*arr_hyb[slic_ind1] + arr_hyb[slic_ind2] + arr_hyb[slic_ind3] + arr_hyb[slic_ind4] + 0.5*arr_hyb[slic_ind5])
                    
                    self.comment = 'Four times oversampled'
                    
                elif oversampling == 5:
                    
                    for dim in range(3):
                        slic_ind1 = [slice(0,-4,5) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                        slic_ind2 = [slice(1,None,5) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                        slic_ind3 = [slice(2,None,5) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                        slic_ind4 = [slice(3,None,5) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]                        
                        slic_ind5 = [slice(4,None,5) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]              
                        arr_hyb = 0.2*(arr_hyb[slic_ind1] + arr_hyb[slic_ind2] + arr_hyb[slic_ind3] + arr_hyb[slic_ind4] + arr_hyb[slic_ind5])

                    self.comment = 'Five times oversampled'

            else:
                
                if oversampling == 2:
                    for dim in range(3):
                        slic_ind1 = [slice(0,-1,2) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                        slic_ind2 = [slice(1,None,2) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                        slic_ind3 = [slice(2,None,2) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]                
                        arr_hyb = 0.25 * arr_hyb[slic_ind1]+ 0.5 *  arr_hyb[slic_ind2] + 0.25 * arr_hyb[slic_ind3]
                    self.comment = 'Two times oversampled'
                
                elif oversampling == 3:
                    for ind in range(3):
                        slic_ind1 = [slice(0,-2,3) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                        slic_ind2 = [slice(1,None,3) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                        slic_ind3 = [slice(2,None,3) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                        slic_ind4 = [slice(3,None,3) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                        arr_hyb = (0.5*arr_hyb[slic_ind1] + arr_hyb[slic_ind2] + arr_hyb[slic_ind3] + 0.5*arr_hyb[slic_ind4])/3.
                    self.comment = 'Three times oversampled'
                        
                elif oversampling == 4:
                    
                    for dim in range(3):
                        slic_ind1 = [slice(0,-3,4) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                        slic_ind2 = [slice(1,None,4) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                        slic_ind3 = [slice(2,None,4) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                        slic_ind4 = [slice(3,None,4) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]                        
                        slic_ind5 = [slice(4,None,4) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                        arr_hyb = 0.25*(0.5*arr_hyb[slic_ind1] + arr_hyb[slic_ind2] + arr_hyb[slic_ind3] + arr_hyb[slic_ind4] + 0.5*arr_hyb[slic_ind5])
                    self.comment = 'Four times oversampled'
                elif oversampling == 5:                    
                    for dim in range(3):
                        slic_ind1 = [slice(0,-4,5) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                        slic_ind2 = [slice(1,None,5) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                        slic_ind3 = [slice(2,None,5) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                        slic_ind4 = [slice(3,None,5) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]                        
                        slic_ind5 = [slice(4,None,5) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]                        
                        slic_ind6 = [slice(5,None,5) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]      
                        arr_hyb = 0.2*(0.5*arr_hyb[slic_ind1] + arr_hyb[slic_ind2] + arr_hyb[slic_ind3] + arr_hyb[slic_ind4] + arr_hyb[slic_ind5] + 0.5*arr_hyb[slic_ind6])
                    self.comment = 'Five times oversampled'
                
            self.out = arr_hyb
            self.res = end_res
            self.shape = end_shape            
                
        self.timestring = dt.datetime.now().strftime('%Y-%m-%d %H:%M')
        
        min_v = self.out.min()
        if min_v < 0.:
            self.out = self.out - min_v        

        
        self.out = self.out / self.out.sum()        

        print(self.out.min())
        print(self.out.max())

 


        
    #----------------------------------------------------------------------
    def _richard_wolf(self):
        """"""
        raise NotImplementedError
        
    #----------------------------------------------------------------------
    def _gibson_lanni(self):
        """"""
        raise NotImplementedError
        
    #----------------------------------------------------------------------
    def _born_wolf(self):
        """"""
        raise NotImplementedError
        
        
    #----------------------------------------------------------------------
    def _defocus(self):
        """"""

        ranges = [np.linspace(-siz, siz, shp) for siz,shp in zip(self.size, self.shape)]
        z_range = ranges.pop()
        
        
        X,Y = np.meshgrid(*ranges, indexing='ij')
        siz2 = [int(shp/2) for shp in self.shape]
        Omega = np.sqrt((mt.pi*X/(siz2[0]*self.res[0]))**2 + (mt.pi*Y/(siz2[1]*self.res[1]))**2)
        
        sigma = self.psf_params['sigma']
        zi = 1.e-6*self.psf_params['zi']
        K = 1.e-6*self.psf_params['K']

        
        
        f_temp_layer = np.zeros(X.shape, dtype = 'complex')
        temp_layer = np.zeros(X.shape, dtype = 'float')
        
        for z,ind in zip(z_range, range(z_range.size)):
            z = z*1.e-6
            temp_layer = (z* Omega*(1.-Omega))/(K*(zi - z))
            
            f_temp_layer = np.exp(-(sigma)**2 * Omega**2) * np.abs(np.sinc(temp_layer))
            
            temp_layer = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(f_temp_layer)).real)
            
            #self.out[:,:,ind] = np.power(temp_layer.copy(), 2)
            self.out[:,:,ind] = temp_layer.copy()
            
        

        
        
        
        
    #----------------------------------------------------------------------
    def _gaussian(self):
        """"""
        ranges = [np.linspace(-siz, siz, shp) for siz,shp in zip(self.size, self.shape)]
        z_range = ranges.pop()
        
        
        X,Y = np.meshgrid(*ranges, indexing='ij')
        Omega2 = (X**2 + Y**2)
        
        sigma_0 = self.psf_params['sigma_0']
        sigma_1 = self.psf_params['sigma_1']
        sigma_2 = self.psf_params['sigma_2']
        
        temp_layer = np.zeros(X.shape, dtype = 'float')
        
        for z,ind in zip(z_range, range(z_range.size)):
            
            cur_sig2 = (sigma_0 + sigma_1*abs(z) + sigma_2*z**2)**2
            
            temp_layer = 1/cur_sig2 * np.exp(-Omega2/(2*cur_sig2))
            
            
            
            self.out[:,:,ind] = temp_layer.copy()
            
        
        
        
    #----------------------------------------------------------------------
    def _gaussian_simple(self):
        """"""
        
        ranges = [np.linspace(-siz, siz, shp) for siz,shp in zip(self.size, self.shape)]
        
        X,Y,Z = np.meshgrid(*ranges, indexing='ij')
        
        sigma = self.psf_params['sigma']
        
        Omega2 = X**2 + Y**2 + Z**2
        
        self.out = 1/sigma**2 * np.exp(-Omega2/sigma**2)
        
        
    #----------------------------------------------------------------------
    def initSaveParameters(self, save_path, save_fident, overwrite = False):
        """"""
        
        self.save_fident = save_fident
        self.overwrite = overwrite
        
        util.createAllPaths(save_path)
        
        if not util.isEmptyDir(save_path) and not overwrite:
            raise ValueError('Save destination is not empty and overwriting is disabled.')
        else:
            self.save_path = save_path    
        
    #----------------------------------------------------------------------
    def saveSolution(self):
        """"""
        
        self.meta.loadFromPSFGen(self, PSF_Generator)
        
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
        
        
        
