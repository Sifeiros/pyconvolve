# -*- coding: utf-8 -*-
"""
Part of the pyconvolve framework for convolution and deconvolution. 
Author: Lukas Küpper, 2018
License: GPLv3
"""
import numpy as np
import math as mt

import util
import util.stack_loader
import util.visu_util as v_util

import sim_creator.convoluter as convoluter

import matplotlib.pyplot as plt
import matplotlib.animation as anim


#----------------------------------------------------------------------
def PSF_noise_additions_batch():
    """"""
    
    main_path = util.SIM_PSF
    
    #path_adds = ['odd', 'even']
    path_adds = ['odd']
    
    #res_list = [[0.1,0.1,0.1], [0.2,0.2,0.2]]
    #res_list = [[0.4,0.4,0.4]]
    res_list = [[0.8,0.8,0.8]]
    res_path_def = 'res_[{0[0]:.1f},{0[1]:.1f},{0[2]:.1f}]'
    
    #Defocus parameter lists
    
    sig_defocus_list = [.05,0.1,0.2,0.3]
    zi_list = [800, 1000, 2000]
    K_list = [100,200,300,400]
    
    defocus_noise_list = [['snr', 60.],['snr',30.],['snr',15.], ['snr',10.], ['snr',5.],['snr',2.]]
    
    f_ident_defocus = 'psf_defocus_{:0>3}.tif'
    path_def_defocus = 'defocussing_sig-{:.2f}_zi-{:.0f}_K-{:.0f}'
    
    for oddity in path_adds:
        for res in res_list:
            for sig in sig_defocus_list:
                for zi in zi_list:
                    for K in K_list:
                        for noise_par in defocus_noise_list:
                            t_noise_free_path = util.ptjoin(main_path, oddity, path_def_defocus.format(sig,zi,K), res_path_def.format(res))
                            t_new_path = util.ptjoin(main_path, oddity, path_def_defocus.format(sig,zi,K), 'res_[{0[0]:.1f},{0[1]:.1f},{0[2]:.1f}]_{1}-{2}'.format(res, noise_par[0], noise_par[1]))
                        
                            util.createAllPaths(t_new_path)
                            
                            temp_stack = util.stack_loader.read_image_stack(t_noise_free_path, f_ident_defocus, meta=True)
        
                            t_noise_dict = {'noise_def':convoluter.Noise_Adder.NOISE_DEF[noise_par[0]], noise_par[0]:noise_par[1]}
                    
                            temp_noise = convoluter.Noise_Adder(temp_stack[0], old_meta=temp_stack[1], img_type='psf', noise_type='gaussian', noise_params=t_noise_dict, debug_int=3)
                            temp_noise.addNoise()
                            
                            temp_noise.initSaveParameters(t_new_path, f_ident_defocus, orig_img_path=[t_noise_free_path, f_ident_defocus], overwrite=True)
                            temp_noise.saveSolution()                            
    
    
    #Gaussian parameter lists
    
    sig0_list = [0.05,0.10,0.15,0.2,0.3,0.4]
    sig1_list = [0]
    sig2_list = [0.15, 0.2, 0.25, 0.3, 0.4]    
    
    gauss_noise_list = [['snr', 60.],['snr',30.],['snr',15.], ['snr',10.], ['snr',5.],['snr',2.]]    
    
    f_ident_gauss = 'psf_gauss_{:0>3}.tif'
    path_def_gauss = 'gauss_sig0-{:.2f}_sig1-{:.2f}_sig2-{:.2f}'
    
    for oddity in path_adds:
        for res in res_list:
            for sig0 in sig0_list:
                for sig1 in sig1_list:
                    for sig2 in sig2_list:
                        for noise_par in gauss_noise_list:
                            
                            t_noise_free_path = util.ptjoin(main_path, oddity, path_def_gauss.format(sig0,sig1,sig2), res_path_def.format(res))
                            t_new_path = util.ptjoin(main_path,oddity,path_def_gauss.format(sig0,sig1,sig2), 'res_[{0[0]:.1f},{0[1]:.1f},{0[2]:.1f}]_{1}-{2}'.format(res, noise_par[0], noise_par[1]))
                            
                            util.createAllPaths(t_new_path)
                            
                            temp_stack = util.stack_loader.read_image_stack(t_noise_free_path, f_ident_gauss, meta=True)
                            
                            t_noise_dict = {'noise_def':convoluter.Noise_Adder.NOISE_DEF[noise_par[0]], noise_par[0]:noise_par[1]}
                            
                            temp_noise = convoluter.Noise_Adder(temp_stack[0], old_meta=temp_stack[1], img_type= 'psf', noise_type='gaussian', noise_params=t_noise_dict, debug_int=3)
                            temp_noise.addNoise()
                            
                            temp_noise.initSaveParameters(t_new_path, f_ident_gauss, orig_img_path=[t_noise_free_path, f_ident_gauss], overwrite=True)
                            temp_noise.saveSolution()
    
    #Gaussian simple lists
    
    sig_gauss_simple_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    
    gauss_simple_noise_list = [['snr', 60.],['snr',30.],['snr',15.], ['snr',10.], ['snr',5.],['snr',2.]]        
    
    f_ident_gauss_simple = 'psf_gauss_simple_{:0>3}.tif'
    path_def_gauss_simple = 'gauss_simple_sig-{:.2f}'
    
    
    for oddity in path_adds:
        for res in res_list:
            for sig in sig_gauss_simple_list:
                for noise_par in gauss_simple_noise_list:
                    
                    t_noise_free_path = util.ptjoin(main_path, oddity, path_def_gauss_simple.format(sig), res_path_def.format(res))
                    t_new_path = util.ptjoin(main_path,oddity,path_def_gauss_simple.format(sig), 'res_[{0[0]:.1f},{0[1]:.1f},{0[2]:.1f}]_{1}-{2}'.format(res, noise_par[0], noise_par[1]))
                    
                    util.createAllPaths(t_new_path)
                    
                    temp_stack = util.stack_loader.read_image_stack(t_noise_free_path, f_ident_gauss_simple, meta=True)
                    
                    t_noise_dict = {'noise_def':convoluter.Noise_Adder.NOISE_DEF[noise_par[0]], noise_par[0]:noise_par[1]}
                    
                    temp_noise = convoluter.Noise_Adder(temp_stack[0], old_meta=temp_stack[1], img_type= 'psf', noise_type='gaussian', noise_params=t_noise_dict, debug_int=3)
                    temp_noise.addNoise()
                    
                    temp_noise.initSaveParameters(t_new_path, f_ident_gauss_simple, orig_img_path=[t_noise_free_path, f_ident_gauss_simple], overwrite=True)
                    temp_noise.saveSolution()                    
    

#----------------------------------------------------------------------
def PSF_noise_additions():
    """"""
    psfs = {
        'defocus1': ['defocussing_sig-0.05_zi-800_K-100', ['res_[0.1,0.1,0.1]', 'res_[0.2,0.2,0.2]'], 'psf_defocus_{:0>3}.tif', 
                    [['snr',15.], ['snr',10.], ['snr',5.],['snr',1.]]],
        'defocus2': ['defocussing_sig-0.05_zi-1000_K-200', ['res_[0.1,0.1,0.1]', 'res_[0.2,0.2,0.2]'], 'psf_defocus_{:0>3}.tif', 
                    [['snr',15.], ['snr',10.], ['snr',5.],['snr',1.]]],    
        'defocus3': ['defocussing_sig-0.05_zi-2000_K-200', ['res_[0.1,0.1,0.1]', 'res_[0.2,0.2,0.2]'], 'psf_defocus_{:0>3}.tif', 
                    [['snr',15.], ['snr',10.], ['snr',5.],['snr',1.]]],            
        'defocus4': ['defocussing_sig-0.10_zi-1000_K-200', ['res_[0.1,0.1,0.1]', 'res_[0.2,0.2,0.2]'], 'psf_defocus_{:0>3}.tif', 
                    [['snr',15.], ['snr',10.], ['snr',5.],['snr',1.]]],        
        'defocus5': ['defocussing_sig-0.10_zi-2200_K-400', ['res_[0.1,0.1,0.1]', 'res_[0.2,0.2,0.2]'], 'psf_defocus_{:0>3}.tif', 
                    [['snr',15.], ['snr',10.], ['snr',5.],['snr',1.]]],        
        'defocus6': ['defocussing_sig-0.20_zi-2200_K-400', ['res_[0.1,0.1,0.1]', 'res_[0.2,0.2,0.2]'], 'psf_defocus_{:0>3}.tif', 
                    [['snr',15.], ['snr',10.], ['snr',5.],['snr',1.]]], 
        'defocus7': ['defocussing_sig-0.30_zi-2200_K-400', ['res_[0.1,0.1,0.1]', 'res_[0.2,0.2,0.2]'], 'psf_defocus_{:0>3}.tif', 
                    [['snr',15.], ['snr',10.], ['snr',5.],['snr',1.]]],
        
        'gauss1':   ['gauss_sig0-0.15_sig1-0.00_sig2-0.22', ['res_[0.1,0.1,0.1]', 'res_[0.2,0.2,0.2]'], 'psf_gauss_{:0>3}.tif', 
                    [['snr',15.], ['snr',10.], ['snr',5.],['snr',1.]]],            
        
        'gauss_simple1': ['gauss_simple_sig-0.15', ['res_[0.1,0.1,0.1]', 'res_[0.2,0.2,0.2]'], 'psf_gauss_simple_{:0>3}.tif', 
                    [['snr',15.], ['snr',10.], ['snr',5.],['snr',1.]]],            
        'gauss_simple2': ['gauss_simple_sig-0.30', ['res_[0.1,0.1,0.1]', 'res_[0.2,0.2,0.2]'], 'psf_gauss_simple_{:0>3}.tif', 
                    [['snr',15.], ['snr',10.], ['snr',5.],['snr',1.]]]
    }   
    
    
    main_path = util.SIM_PSF
    
    for name,parameters in psfs.items():
        print('Processing PSF: {}'.format(name))
        
        add_path = parameters[0]
        res_folders = parameters[1]
        f_ident = parameters[2]
        noise_params = parameters[3]
        
        for res in res_folders:
            
            for noise in noise_params:
                
                noise_free_path = util.ptjoin(main_path, add_path, res)
                new_path = util.ptjoin(main_path, add_path, '{}_{}-{}'.format(res, noise[0], noise[1]))
                
                util.createAllPaths(new_path)
                
                temp_stack = util.stack_loader.read_image_stack(noise_free_path, f_ident, meta=True)
        
                t_noise_dict = {'noise_def':convoluter.Noise_Adder.NOISE_DEF[noise[0]], noise[0]:noise[1]}
        
                temp_noise = convoluter.Noise_Adder(temp_stack[0], old_meta=temp_stack[1], img_type='psf', noise_type='gaussian', noise_params=t_noise_dict, debug_int=3)
                temp_noise.addNoise()
                
                temp_noise.initSaveParameters(new_path, f_ident, orig_img_path=[noise_free_path, f_ident], overwrite=True)
                temp_noise.saveSolution()
    
    
#----------------------------------------------------------------------
def sample_noise_additions():
    """"""
    convs = {
        'defocus1': ['cyto_defocus_sig-0.10_zi-1000_K-200', ['res_[0.2,0.2,0.2]'], 'cyto_conv_{:0>3}.tif', 
                    [['cnr',30.], ['cnr',10.],['cnr',5.]]],
        'defocus2': ['cyto_defocus_sig-0.05_zi-2000_K-200', ['res_[0.2,0.2,0.2]'], 'cyto_conv_{:0>3}.tif', 
                    [['cnr',30.], ['cnr',10.],['cnr',5.]]],
        'defocus3': ['cyto_defocus_sig-0.05_zi-1000_K-200', ['res_[0.2,0.2,0.2]'], 'cyto_conv_{:0>3}.tif', 
                    [['cnr',30.], ['cnr',10.],['cnr',5.]]],
        'defocus4': ['cyto_defocus_sig-0.05_zi-800_K-100', ['res_[0.1,0.1,0.1]', 'res_[0.2,0.2,0.2]'], 'cyto_conv_{:0>3}.tif', 
                    [['cnr',30.], ['cnr',10.],['cnr',5.]]],

        
        'gauss1':   ['cyto_gauss_sig0-0.15_sig1-0.00_sig2-0.22', ['res_[0.1,0.1,0.1]', 'res_[0.2,0.2,0.2]'], 'cyto_conv_{:0>3}.tif', 
                    [['cnr',30.], ['cnr',10.],['cnr',5.]]],
        
        'gauss_simple1': ['cyto_gauss_simple_sig-0.15', ['res_[0.1,0.1,0.1]', 'res_[0.2,0.2,0.2]'], 'cyto_conv_{:0>3}.tif', 
                    [['cnr',30.], ['cnr',10.],['cnr',5.]]],
        'gauss_simple2': ['cyto_gauss_simple_sig-0.30', ['res_[0.1,0.1,0.1]'], 'cyto_conv_{:0>3}.tif', 
                    [['cnr',30.], ['cnr',10.],['cnr',5.]]],
    }       
    
    main_path = util.SIM_CONVOLUTED
    
    for name,parameters in psfs.items():
        print('Processing Convoluted Image Stack: {}'.format(name))
        
        add_path = parameters[0]
        res_folders = parameters[1]
        f_ident = parameters[2]
        noise_params = parameters[3]
        
        for res in res_folders:
            
            for noise in noise_params:
                
                noise_free_path = util.ptjoin(main_path, add_path, res)
                new_path = util.ptjoin(main_path, add_path, '{}_{}-{}'.format(res, noise[0], noise[1]))
                
                util.createAllPaths(new_path)
                
                temp_stack = util.stack_loader.read_image_stack(noise_free_path, f_ident, meta=True)
        
                t_noise_dict = {'noise_def':convoluter.Noise_Adder.NOISE_DEF[noise[0]], noise[0]:noise[1]}
        
                temp_noise = convoluter.Noise_Adder(temp_stack[0], old_meta=temp_stack[1], img_type='sample', noise_type='gaussian', noise_params=t_noise_dict, debug_int=3)
                temp_noise.addNoise()
                
                temp_noise.initSaveParameters(new_path, f_ident, orig_img_path=[noise_free_path, f_ident], overwrite=True)
                temp_noise.saveSolution()    


if __name__ == '__main__':
    PSF_noise_additions_batch()