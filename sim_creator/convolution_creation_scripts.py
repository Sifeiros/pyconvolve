# -*- coding: utf-8 -*-
"""
Part of the pyconvolve framework for convolution and deconvolution. 
Author: Lukas Küpper, 2018
License: GPLv3
"""
import numpy as np
import math as mt
import os

import util
import util.stack_loader
import util.visu_util as v_util

import sim_creator.convoluter as convoluter

import matplotlib.pyplot as plt
import matplotlib.animation as anim



    

#----------------------------------------------------------------------
def conv_gaussian_simple():
    """"""
    
    conv_resolution = 'hi_res'
    
    inp_fident = 'cyto_sim_{:0>3}.tif'
    inp_paths = {'lo_res':'cyto_sim_resol-[0.2,0.2,0.2]', 'hi_res':'cyto_sim_resol-[0.1,0.1,0.1]'}
    inp_paths = {k:util.ptjoin(util.SIM_DATA, pt) for k,pt in inp_paths.items()}    
    
    psf_paths = {'lo_res':'gauss_simple_sig-0.15/res_[0.2,0.2,0.2]', 'hi_res':'gauss_simple_sig-0.15/res_[0.1,0.1,0.1]'}
    psf_paths = {k:util.ptjoin(util.SIM_PSF, pt) for k,pt in psf_paths.items()}    
    psf_fident = 'psf_gauss_simple_{:0>3}.tif'    
    
    out_paths = {'lo_res': util.ptjoin(util.SIM_CONVOLUTED, 'cyto_gauss_simple_sig-0.15', 'res_[0.2,0.2,0.2]'), 'hi_res': util.ptjoin(util.SIM_CONVOLUTED, 'cyto_gauss_simple_sig-0.15', 'res_[0.1,0.1,0.1]')}
    out_fident = 'cyto_conv_{:0>3}.tif'
    
    inp_res = util.stack_loader.read_image_stack(inp_paths[conv_resolution], inp_fident, 0, 256, meta=True) #Including Meta Data
    psf_res = util.stack_loader.read_image_stack(psf_paths[conv_resolution], psf_fident, 0, 33, meta=True)
    

    conv = convoluter.Convoluter(inp_res[0], psf_res[0], isPsfCentered=True, conv_method='real_fft', noise_type='no_noise', noise_params=None, debug_int=3)
    conv.initSaveParameters(out_paths[conv_resolution], out_fident, orig_img_path=[inp_paths[conv_resolution], inp_fident], orig_psf_path=[psf_paths[conv_resolution], psf_fident])
    conv.convolute()

    conv.saveSolution()

        
    #[fig,ani] = v_util.imshow3D_ani(psf_res[0])
    [fig,ani] = v_util.imshow3D_ani(conv.out)
    plt.show()
    
#----------------------------------------------------------------------
def conv_defocus():
    """"""
    
    conv_resolution = 'hi_res'
    
    inp_fident = 'cyto_sim_{:0>3}.tif'
    inp_paths = {'lo_res':'cyto_sim_resol-[0.2,0.2,0.2]', 'hi_res':'cyto_sim_resol-[0.1,0.1,0.1]'}
    inp_paths = {k:util.ptjoin(util.SIM_DATA, pt) for k,pt in inp_paths.items()}    
    
    psf_paths = {'lo_res':'defocussing_sig-0.05_zi-800_K-100/res_[0.2,0.2,0.2]', 'hi_res':'defocussing_sig-0.05_zi-800_K-100/res_[0.1,0.1,0.1]'}
    psf_paths = {k:util.ptjoin(util.SIM_PSF, pt) for k,pt in psf_paths.items()}    
    psf_fident = 'psf_defocus_{:0>3}.tif'
    
    #psf_paths = {'lo_res':'gauss_sig0-0.15_sig1-0.00_sig2-0.22/res_[0.2,0.2,0.2]', 'hi_res':'gauss_sig0-0.15_sig1-0.00_sig2-0.220/res_[0.1,0.1,0.1]'}
    #psf_paths = {k:util.ptjoin(util.SIM_PSF, pt) for k,pt in psf_paths.items()}    
    #psf_fident = 'psf_gauss_{:0>3}.tif'
    
    #psf_paths = {'lo_res':'gauss_simple_sig-0.30/res_[0.2,0.2,0.2]', 'hi_res':'gauss_simple_sig-0.30/res_[0.1,0.1,0.1]'}
    #psf_paths = {k:util.ptjoin(util.SIM_PSF, pt) for k,pt in psf_paths.items()}    
    #psf_fident = 'psf_gauss_simple_{:0>3}.tif'    
    
    out_paths = {'lo_res': util.ptjoin(util.SIM_CONVOLUTED, 'cyto_defocus_sig-0.05_zi-800_K-100', 'res_[0.2,0.2,0.2]'), 'hi_res': util.ptjoin(util.SIM_CONVOLUTED, 'cyto_defocus_sig-0.05_zi-800_K-100', 'res_[0.1,0.1,0.1]')}
    out_fident = 'cyto_conv_{:0>3}.tif'
    
    inp_res = util.stack_loader.read_image_stack(inp_paths[conv_resolution], inp_fident, 0, 256, meta=True) #Including Meta Data
    psf_res = util.stack_loader.read_image_stack(psf_paths[conv_resolution], psf_fident, 0, 129, meta=True)
    

    conv = convoluter.Convoluter(inp_res[0], psf_res[0], isPsfCentered=True, conv_method='real_fft', noise_type='no_noise', noise_params=None, debug_int=3)
    conv.initSaveParameters(out_paths[conv_resolution], out_fident, orig_img_path=[inp_paths[conv_resolution], inp_fident], orig_psf_path=[psf_paths[conv_resolution], psf_fident])
    conv.convolute()

    conv.saveSolution()

        
    #[fig,ani] = v_util.imshow3D_ani(psf_res[0])
    [fig,ani] = v_util.imshow3D_ani(conv.out)
    plt.show()
    
#----------------------------------------------------------------------
def conv_gaussian():
    """"""
    conv_resolution = 'hi_res'
    
    inp_fident = 'cyto_sim_{:0>3}.tif'
    inp_paths = {'lo_res':'cyto_sim_resol-[0.2,0.2,0.2]', 'hi_res':'cyto_sim_resol-[0.1,0.1,0.1]'}
    inp_paths = {k:util.ptjoin(util.SIM_DATA, pt) for k,pt in inp_paths.items()}    
    
    psf_paths = {'lo_res':'gauss_sig0-0.15_sig1-0.00_sig2-0.22/res_[0.2,0.2,0.2]', 'hi_res':'gauss_sig0-0.15_sig1-0.00_sig2-0.22/res_[0.1,0.1,0.1]'}
    psf_paths = {k:util.ptjoin(util.SIM_PSF, pt) for k,pt in psf_paths.items()}    
    psf_fident = 'psf_gauss_{:0>3}.tif'
    
    out_paths = {'lo_res': util.ptjoin(util.SIM_CONVOLUTED, 'cyto_gauss_sig0-0.15_sig1-0.00_sig2-0.22', 'res_[0.2,0.2,0.2]'), 'hi_res': util.ptjoin(util.SIM_CONVOLUTED, 'cyto_gauss_sig0-0.15_sig1-0.00_sig2-0.22', 'res_[0.1,0.1,0.1]')}
    out_fident = 'cyto_conv_{:0>3}.tif'
    
    inp_res = util.stack_loader.read_image_stack(inp_paths[conv_resolution], inp_fident, 0, 256, meta=True) #Including Meta Data
    psf_res = util.stack_loader.read_image_stack(psf_paths[conv_resolution], psf_fident, 0, 65, meta=True)
    

    conv = convoluter.Convoluter(inp_res[0], psf_res[0], isPsfCentered=True, conv_method='real_fft', noise_type='no_noise', noise_params=None, debug_int=3)
    conv.initSaveParameters(out_paths[conv_resolution], out_fident, orig_img_path=[inp_paths[conv_resolution], inp_fident], orig_psf_path=[psf_paths[conv_resolution], psf_fident])
    conv.convolute()

    conv.saveSolution()

        
    #[fig,ani] = v_util.imshow3D_ani(psf_res[0])
    [fig,ani] = v_util.imshow3D_ani(conv.out)
    plt.show()    
    
    

#----------------------------------------------------------------------
def conv_gaussian_simple_batch(used_res):
    """"""
    
    #used_res = [0.1,0.1,0.1]
    psf_oddity = 'odd'    
    convolution_method = 'real_fft'
    noise_type = 'no_noise'
    noise_params = None
    
    #path_res_add = 'res_[{0[0]:.1f},{0[1]:.1f},{0[2]:.1f}]'    
    
    #sample_path_in = util.ptjoin(util.SIM_DATA, 'cyto_sim_full', path_res_add.format(used_res))
    #psf_path_in = util.ptjoin(util.SIM_PSF,psf_oddity, 'gauss_simple_sig-{:.2f}', path_res_add.format(used_res))
    #conv_path_out = util.ptjoin(util.SIM_CONVOLUTED, 'cyto_gauss_simple_sig-{:.2f}', path_res_add.format(used_res))
    
    path_res_add = 'full_res_[{0[0]:.1f},{0[1]:.1f},{0[2]:.1f}]'
    path_res_add_psf = 'res_[{0[0]:.1f},{0[1]:.1f},{0[2]:.1f}]'    
    
    #sample_path_in = util.ptjoin(util.SIM_DATA, 'cyto_sim_full', path_res_add.format(used_res))
    sample_path_in = util.ptjoin(util.SIM_DATA, 'psf_recon_full', path_res_add.format(used_res))
    psf_path_in = util.ptjoin(util.SIM_PSF,psf_oddity, 'gauss_simple_sig-{:.2f}', path_res_add_psf.format(used_res))
    #conv_path_out = util.ptjoin(util.SIM_CONVOLUTED, 'cyto_gauss_simple_sig-{:.2f}', path_res_add.format(used_res))
    conv_path_out = util.ptjoin(util.SIM_CONVOLUTED, 'psf_recon_gauss_simple_sig-{:.2f}', path_res_add.format(used_res))     
    
    
    sigma_list_full = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    
    sigma_list = [0.05, 0.15, 0.25, 0.5]
    
    psf_fident = 'psf_gauss_simple_{:0>3}.tif'
    sample_fident = 'cyto_sim_{:0>3}.tif'
    conv_fident = 'cyto_conv_gauss_simple_{:0>3}.tif'
    
    
    sample_stack, sample_meta = util.stack_loader.read_image_stack(sample_path_in, sample_fident, meta= True)
    if psf_oddity == 'odd':
        print(sample_stack.shape)
        shape_odd = [sh-1 if sh%2 == 0 else sh for sh in sample_stack.shape]
        index = [slice(None, sh, 1) for sh in shape_odd]
        sample_stack = sample_stack[index]
        print(sample_stack.shape)
    
    
    for sig in sigma_list:
        
        t_psf_path_in = psf_path_in.format(sig)
        print('Processing: {}'.format(t_psf_path_in))
        t_conv_path_out = conv_path_out.format(sig)
        t_psf_stack, t_psf_meta = util.stack_loader.read_image_stack(t_psf_path_in, psf_fident, meta=True)
        
        t_conv = convoluter.Convoluter(sample_stack, t_psf_stack, isPsfCentered= True, conv_method= convolution_method, noise_type= noise_type, noise_params= noise_params, debug_int=3, comment = 'With Fourth iteration PSFs')
        t_conv.initSaveParameters(t_conv_path_out, conv_fident, orig_img_path=[sample_path_in, sample_fident], orig_psf_path=[t_psf_path_in, psf_fident])
        t_conv.convolute()
        t_conv.saveSolution()
    
    
    
#----------------------------------------------------------------------
def conv_gaussian_batch(used_res):
    """"""
    #used_res = [0.2,0.2,0.2]
    psf_oddity = 'odd'    
    convolution_method = 'real_fft'
    noise_type = 'no_noise'
    noise_params = None
    
    #path_res_add = 'res_[{0[0]:.1f},{0[1]:.1f},{0[2]:.1f}]'    
    
    #sample_path_in = util.ptjoin(util.SIM_DATA, 'cyto_sim_full', path_res_add.format(used_res))
    #psf_path_in = util.ptjoin(util.SIM_PSF,psf_oddity, 'gauss_sig0-{:.2f}_sig1-{:.2f}_sig2-{:.2f}', path_res_add.format(used_res))
    #conv_path_out = util.ptjoin(util.SIM_CONVOLUTED, 'cyto_gauss_sig0-{:.2f}_sig1-{:.2f}_sig2-{:.2f}', path_res_add.format(used_res))
    
    path_res_add = 'full_res_[{0[0]:.1f},{0[1]:.1f},{0[2]:.1f}]'
    path_res_add_psf = 'res_[{0[0]:.1f},{0[1]:.1f},{0[2]:.1f}]'    
    
    #sample_path_in = util.ptjoin(util.SIM_DATA, 'cyto_sim_full', path_res_add.format(used_res))
    sample_path_in = util.ptjoin(util.SIM_DATA, 'psf_recon_full', path_res_add.format(used_res))
    psf_path_in = util.ptjoin(util.SIM_PSF,psf_oddity, 'gauss_sig0-{:.2f}_sig1-{:.2f}_sig2-{:.2f}', path_res_add_psf.format(used_res))
    #conv_path_out = util.ptjoin(util.SIM_CONVOLUTED, 'cyto_gauss_sig0-{:.2f}_sig1-{:.2f}_sig2-{:.2f}', path_res_add.format(used_res))
    conv_path_out = util.ptjoin(util.SIM_CONVOLUTED, 'psf_recon_gauss_sig0-{:.2f}_sig1-{:.2f}_sig2-{:.2f}', path_res_add.format(used_res))

    
    sig0_list_full = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    sig1_list_full = [0.]
    sig2_list_full = [0.15, 0.20, 0.25, 0.30, 0.40]
    
    sig0_list = [0.05, 0.1, 0.2]
    sig1_list = [0.]
    sig2_list = [0.15, 0.20, 0.25, 0.40]
    
    psf_fident = 'psf_gauss_{:0>3}.tif'
    sample_fident = 'cyto_sim_{:0>3}.tif'
    conv_fident = 'cyto_conv_gauss_{:0>3}.tif'
    
    
    sample_stack, sample_meta = util.stack_loader.read_image_stack(sample_path_in, sample_fident, meta= True)
    if psf_oddity == 'odd':
        print(sample_stack.shape)
        shape_odd = [sh-1 if sh%2 == 0 else sh for sh in sample_stack.shape]
        index = [slice(None, sh, 1) for sh in shape_odd]
        sample_stack = sample_stack[index]
        print(sample_stack.shape)
    
    for sig0 in sig0_list:
        for sig1 in sig1_list:
            for sig2 in sig2_list:
    
                t_psf_path_in = psf_path_in.format(sig0, sig1, sig2)
                print('Processing: {}'.format(t_psf_path_in))
                t_conv_path_out = conv_path_out.format(sig0, sig1, sig2)
                t_psf_stack, t_psf_meta = util.stack_loader.read_image_stack(t_psf_path_in, psf_fident, meta=True)
                
                t_conv = convoluter.Convoluter(sample_stack, t_psf_stack, isPsfCentered= True, conv_method= convolution_method, noise_type= noise_type, noise_params= noise_params, debug_int=3, comment = 'With Fourth iteration PSFs')
                t_conv.initSaveParameters(t_conv_path_out, conv_fident, orig_img_path=[sample_path_in, sample_fident], orig_psf_path=[t_psf_path_in, psf_fident])
                t_conv.convolute()
                t_conv.saveSolution()
    
#----------------------------------------------------------------------
def conv_defocus_batch(used_res):
    """"""
    #used_res = [0.2,0.2,0.2]
    psf_oddity = 'odd'    
    convolution_method = 'real_fft'
    noise_type = 'no_noise'
    noise_params = None
    
    #path_res_add = 'res_[{0[0]:.1f},{0[1]:.1f},{0[2]:.1f}]'    
    path_res_add = 'full_res_[{0[0]:.1f},{0[1]:.1f},{0[2]:.1f}]'
    path_res_add_psf = 'res_[{0[0]:.1f},{0[1]:.1f},{0[2]:.1f}]'    
    
    #sample_path_in = util.ptjoin(util.SIM_DATA, 'cyto_sim_full', path_res_add.format(used_res))
    sample_path_in = util.ptjoin(util.SIM_DATA, 'psf_recon_full', path_res_add.format(used_res))
    
    psf_path_in = util.ptjoin(util.SIM_PSF,psf_oddity, 'defocussing_sig-{:.2f}_zi-{:}_K-{:}', path_res_add_psf.format(used_res))
    #conv_path_out = util.ptjoin(util.SIM_CONVOLUTED, 'cyto_defocus_sig-{:.2f}_zi-{:}_K-{:}', path_res_add.format(used_res))
    conv_path_out = util.ptjoin(util.SIM_CONVOLUTED, 'psf_recon_defocus_sig-{:.2f}_zi-{:}_K-{:}', path_res_add.format(used_res))
    
    
    
    sig_list_full = [.05,0.1,0.2,0.3]
    zi_list_full = [800, 1000, 2000]
    K_list_full = [100,200,300,400]
    
    sig_list = [.05,0.1,0.3]
    zi_list = [800, 2000]
    K_list = [100,200,400]    

    #sig_list = [.3]
    #zi_list = [2000]
    #K_list = [400]
    
    psf_fident = 'psf_defocus_{:0>3}.tif'
    sample_fident = 'cyto_sim_{:0>3}.tif'
    conv_fident = 'cyto_conv_defocus_{:0>3}.tif'
    
    
    sample_stack, sample_meta = util.stack_loader.read_image_stack(sample_path_in, sample_fident, meta= True)
    if psf_oddity == 'odd':
        print(sample_stack.shape)
        shape_odd = [sh-1 if sh%2 == 0 else sh for sh in sample_stack.shape]
        index = [slice(None, sh, 1) for sh in shape_odd]
        sample_stack = sample_stack[index]
        print(sample_stack.shape)
    
    for sig in sig_list:
        for zi in zi_list:
            for K in K_list:
    
                t_psf_path_in = psf_path_in.format(sig, zi, K)
                print('Processing: {}'.format(t_psf_path_in))
                t_conv_path_out = conv_path_out.format(sig, zi, K)
                t_psf_stack, t_psf_meta = util.stack_loader.read_image_stack(t_psf_path_in, psf_fident, meta=True)
                
                t_conv = convoluter.Convoluter(sample_stack, t_psf_stack, isPsfCentered= True, conv_method= convolution_method, noise_type= noise_type, noise_params= noise_params, debug_int=3, comment = 'With Fourth iteration PSFs')
                t_conv.initSaveParameters(t_conv_path_out, conv_fident, orig_img_path=[sample_path_in, sample_fident], orig_psf_path=[t_psf_path_in, psf_fident])
                t_conv.convolute()
                t_conv.saveSolution()    
    

#----------------------------------------------------------------------
def crop_convoluted_stacks():
    """"""
    
    root_path = util.SIM_CONVOLUTED
    
    #Simple Gaussian
    #----------------------------------------------------------------------
    
    sig = [0.5, 0.25, 0.15, 0.05]
    #sig = []
    
    path_ident = 'cyto_gauss_simple_sig-{:.2f}'
    f_ident = 'cyto_conv_gauss_simple_{:0>3}.tif'
    
    crop_diviser = 2
    
    for s in sig:
        cur_path = util.ptjoin(root_path, path_ident.format(s))
        print('Cropping path: {}'.format(cur_path))

        #cur_path_full = util.ptjoin(cur_path, 'full_res_[0.2,0.2,0.2]')
        #cur_path_down = util.ptjoin(cur_path, 'downsampled_res_[0.2,0.2,0.2]')
        
        #crop_paths_full = ['full_crop_{}_{}_res_[0.2,0.2,0.2]'.format(i/crop_diviser, i%crop_diviser) for i in range(crop_diviser**2)]
        #crop_paths_down = ['downsampled_crop_{}_{}_res_[0.2,0.2,0.2]'.format(i/crop_diviser, i%crop_diviser) for i in range(crop_diviser**2)]

        cur_path_full = util.ptjoin(cur_path, 'full_res_[0.4,0.4,0.4]')
        cur_path_down = util.ptjoin(cur_path, 'downsampled_res_[0.4,0.4,0.4]')
        
        crop_paths_full = ['full_crop_{}_{}_res_[0.4,0.4,0.4]'.format(i/crop_diviser, i%crop_diviser) for i in range(crop_diviser**2)]
        crop_paths_down = ['downsampled_crop_{}_{}_res_[0.4,0.4,0.4]'.format(i/crop_diviser, i%crop_diviser) for i in range(crop_diviser**2)]
        
        crop_paths_full = [util.ptjoin(cur_path, t_pt) for t_pt in crop_paths_full]
        crop_paths_down = [util.ptjoin(cur_path, t_pt) for t_pt in crop_paths_down]
        
        util.createAllPaths(crop_paths_full)
        util.createAllPaths(crop_paths_down)
        
        full_stack = util.stack_loader.read_image_stack(cur_path_full, f_ident, meta= False)
        full_meta = util.stack_loader.read_meta_data_only(cur_path_full, f_ident)
        
        t_reduced_shape = [sh/crop_diviser if sh%2 == 0 else 1+sh/crop_diviser for sh in full_stack.shape]
        meta_comment = full_meta.comment        
        
        for i,crop_path in enumerate(crop_paths_full):
            x_ind = i/crop_diviser
            y_ind = i%crop_diviser
            sh_bool2 = [sh%2 == 0 for sh in t_reduced_shape]
            ind_slice = [slice(x_ind*t_reduced_shape[0] if sh_bool2[0] else x_ind*(t_reduced_shape[0]-1),
                               (x_ind+1)*t_reduced_shape[0],1), 
                         slice(y_ind*t_reduced_shape[1] if sh_bool2[1] else y_ind*(t_reduced_shape[0]-1),
                               (y_ind+1)*t_reduced_shape[1],1), 
                         slice(None,None,1)]
            cropped_stack = full_stack[ind_slice].copy()
            full_meta.comment = meta_comment+'; Cropped; Crop_index=[{},{}]'.format(x_ind,y_ind)
            util.stack_loader.write_image_stack(cropped_stack, crop_path, f_ident, 0, meta_data= full_meta.toList())
        
        down_stack = util.stack_loader.read_image_stack(cur_path_down, f_ident, meta= False)
        down_meta = util.stack_loader.read_meta_data_only(cur_path_down, f_ident)
        
        t_reduced_shape = [sh/crop_diviser if sh%2 == 0 else 1+sh/crop_diviser for sh in down_stack.shape]
        meta_comment = down_meta.comment              
        
        for crop_path in crop_paths_down:
            x_ind = i/crop_diviser
            y_ind = i%crop_diviser
            sh_bool2 = [sh%2 == 0 for sh in t_reduced_shape]
            ind_slice = [slice(x_ind*t_reduced_shape[0] if sh_bool2[0] else x_ind*(t_reduced_shape[0]-1),
                               (x_ind+1)*t_reduced_shape[0],1), 
                         slice(y_ind*t_reduced_shape[1] if sh_bool2[1] else y_ind*(t_reduced_shape[0]-1),
                               (y_ind+1)*t_reduced_shape[1],1), 
                         slice(None,None,1)]
            cropped_stack = down_stack[ind_slice].copy()
            down_meta.comment = meta_comment+'; Cropped; Crop_index=[{},{}]'.format(x_ind,y_ind)
            util.stack_loader.write_image_stack(cropped_stack, crop_path, f_ident, 0, meta_data= down_meta.toList())        
        
        
        
        
        
        
        
        
    
    #Gaussian Defocus
    #----------------------------------------------------------------------
    
    sig0 = [0.2, 0.1, 0.05]
    sig1 = [0.]
    sig2 = [0.4, 0.25, 0.20, 0.15]
    
    path_ident = 'cyto_gauss_sig0-{:.2f}_sig1-{:.2f}_sig2-{:.2f}'
    f_ident = 'cyto_conv_gauss_{:0>3}.tif'
    
    for s0 in sig0:
        for s1 in sig1:
            for s2 in sig2:
                cur_path = util.ptjoin(root_path, path_ident.format(s0, s1, s2))
                print('Downsampling path: {}'.format(cur_path))
                
                #cur_path_full = util.ptjoin(cur_path, 'full_res_[0.2,0.2,0.2]')
                #cur_path_down = util.ptjoin(cur_path, 'downsampled_res_[0.2,0.2,0.2]')
                
                #crop_paths_full = ['full_crop_{}_{}_res_[0.2,0.2,0.2]'.format(i/crop_diviser, i%crop_diviser) for i in range(crop_diviser**2)]
                #crop_paths_down = ['downsampled_crop_{}_{}_res_[0.2,0.2,0.2]'.format(i/crop_diviser, i%crop_diviser) for i in range(crop_diviser**2)]


                cur_path_full = util.ptjoin(cur_path, 'full_res_[0.4,0.4,0.4]')
                cur_path_down = util.ptjoin(cur_path, 'downsampled_res_[0.4,0.4,0.4]')
                
                crop_paths_full = ['full_crop_{}_{}_res_[0.4,0.4,0.4]'.format(i/crop_diviser, i%crop_diviser) for i in range(crop_diviser**2)]
                crop_paths_down = ['downsampled_crop_{}_{}_res_[0.4,0.4,0.4]'.format(i/crop_diviser, i%crop_diviser) for i in range(crop_diviser**2)]
                
                crop_paths_full = [util.ptjoin(cur_path, t_pt) for t_pt in crop_paths_full]
                crop_paths_down = [util.ptjoin(cur_path, t_pt) for t_pt in crop_paths_down]
                
                util.createAllPaths(crop_paths_full)
                util.createAllPaths(crop_paths_down)
                
                full_stack = util.stack_loader.read_image_stack(cur_path_full, f_ident, meta= False)
                full_meta = util.stack_loader.read_meta_data_only(cur_path_full, f_ident)
                
                t_reduced_shape = [sh/crop_diviser if sh%2 == 0 else 1+sh/crop_diviser for sh in full_stack.shape]
                meta_comment = full_meta.comment        
                
                for i,crop_path in enumerate(crop_paths_full):
                    x_ind = i/crop_diviser
                    y_ind = i%crop_diviser
                    sh_bool2 = [sh%2 == 0 for sh in t_reduced_shape]
                    ind_slice = [slice(x_ind*t_reduced_shape[0] if sh_bool2[0] else x_ind*(t_reduced_shape[0]-1),
                                       (x_ind+1)*t_reduced_shape[0],1), 
                                 slice(y_ind*t_reduced_shape[1] if sh_bool2[1] else y_ind*(t_reduced_shape[0]-1),
                                       (y_ind+1)*t_reduced_shape[1],1), 
                                 slice(None,None,1)]
                    cropped_stack = full_stack[ind_slice].copy()
                    full_meta.comment = meta_comment+'; Cropped; Crop_index=[{},{}]'.format(x_ind,y_ind)
                    util.stack_loader.write_image_stack(cropped_stack, crop_path, f_ident, 0, meta_data= full_meta.toList())
                
                down_stack = util.stack_loader.read_image_stack(cur_path_down, f_ident, meta= False)
                down_meta = util.stack_loader.read_meta_data_only(cur_path_down, f_ident)
                
                t_reduced_shape = [sh/crop_diviser if sh%2 == 0 else 1+sh/crop_diviser for sh in down_stack.shape]
                meta_comment = down_meta.comment              
                
                for crop_path in crop_paths_down:
                    x_ind = i/crop_diviser
                    y_ind = i%crop_diviser
                    sh_bool2 = [sh%2 == 0 for sh in t_reduced_shape]
                    ind_slice = [slice(x_ind*t_reduced_shape[0] if sh_bool2[0] else x_ind*(t_reduced_shape[0]-1),
                                       (x_ind+1)*t_reduced_shape[0],1), 
                                 slice(y_ind*t_reduced_shape[1] if sh_bool2[1] else y_ind*(t_reduced_shape[0]-1),
                                       (y_ind+1)*t_reduced_shape[1],1), 
                                 slice(None,None,1)]
                    cropped_stack = down_stack[ind_slice].copy()
                    down_meta.comment = meta_comment+'; Cropped; Crop_index=[{},{}]'.format(x_ind,y_ind)
                    util.stack_loader.write_image_stack(cropped_stack, crop_path, f_ident, 0, meta_data= down_meta.toList())                       
                
                             
                
    
    #Fourier Defocus
    #----------------------------------------------------------------------
    
    sigF = [0.3, 0.1, 0.05]
    zi = [2000, 800]
    K = [400, 200, 100]    

    f_ident = 'cyto_conv_defocus_{:0>3}.tif'
    path_ident = 'cyto_defocus_sig-{:.2f}_zi-{:}_K-{:}'

    for sf in sigF:
        for z in zi:
            for k in K:
                
                cur_path = util.ptjoin(root_path, path_ident.format(sf, z, k))
                print('Downsampling path: {}'.format(cur_path))
                #cur_path_full = util.ptjoin(cur_path, 'full_res_[0.2,0.2,0.2]')
                #cur_path_down = util.ptjoin(cur_path, 'downsampled_res_[0.2,0.2,0.2]')
                
                #crop_paths_full = ['full_crop_{}_{}_res_[0.2,0.2,0.2]'.format(i/crop_diviser, i%crop_diviser) for i in range(crop_diviser**2)]
                #crop_paths_down = ['downsampled_crop_{}_{}_res_[0.2,0.2,0.2]'.format(i/crop_diviser, i%crop_diviser) for i in range(crop_diviser**2)]
                
                cur_path_full = util.ptjoin(cur_path, 'full_res_[0.4,0.4,0.4]')
                cur_path_down = util.ptjoin(cur_path, 'downsampled_res_[0.4,0.4,0.4]')
                
                crop_paths_full = ['full_crop_{}_{}_res_[0.4,0.4,0.4]'.format(i/crop_diviser, i%crop_diviser) for i in range(crop_diviser**2)]
                crop_paths_down = ['downsampled_crop_{}_{}_res_[0.4,0.4,0.4]'.format(i/crop_diviser, i%crop_diviser) for i in range(crop_diviser**2)]                
                
                crop_paths_full = [util.ptjoin(cur_path, t_pt) for t_pt in crop_paths_full]
                crop_paths_down = [util.ptjoin(cur_path, t_pt) for t_pt in crop_paths_down]
                
                util.createAllPaths(crop_paths_full)
                util.createAllPaths(crop_paths_down)
                
                full_stack = util.stack_loader.read_image_stack(cur_path_full, f_ident, meta= False)
                full_meta = util.stack_loader.read_meta_data_only(cur_path_full, f_ident)
                
                t_reduced_shape = [sh/crop_diviser if sh%2 == 0 else 1+sh/crop_diviser for sh in full_stack.shape]
                meta_comment = full_meta.comment        
                
                for i,crop_path in enumerate(crop_paths_full):
                    x_ind = i/crop_diviser
                    y_ind = i%crop_diviser
                    sh_bool2 = [sh%2 == 0 for sh in t_reduced_shape]
                    ind_slice = [slice(x_ind*t_reduced_shape[0] if sh_bool2[0] else x_ind*(t_reduced_shape[0]-1),
                                       (x_ind+1)*t_reduced_shape[0],1), 
                                 slice(y_ind*t_reduced_shape[1] if sh_bool2[1] else y_ind*(t_reduced_shape[0]-1),
                                       (y_ind+1)*t_reduced_shape[1],1), 
                                 slice(None,None,1)]
                    cropped_stack = full_stack[ind_slice].copy()
                    full_meta.comment = meta_comment+'; Cropped; Crop_index=[{},{}]'.format(x_ind,y_ind)
                    util.stack_loader.write_image_stack(cropped_stack, crop_path, f_ident, 0, meta_data= full_meta.toList())
                
                down_stack = util.stack_loader.read_image_stack(cur_path_down, f_ident, meta= False)
                down_meta = util.stack_loader.read_meta_data_only(cur_path_down, f_ident)
                
                t_reduced_shape = [sh/crop_diviser if sh%2 == 0 else 1+sh/crop_diviser for sh in down_stack.shape]
                meta_comment = down_meta.comment              
                
                for crop_path in crop_paths_down:
                    x_ind = i/crop_diviser
                    y_ind = i%crop_diviser
                    sh_bool2 = [sh%2 == 0 for sh in t_reduced_shape]
                    ind_slice = [slice(x_ind*t_reduced_shape[0] if sh_bool2[0] else x_ind*(t_reduced_shape[0]-1),
                                       (x_ind+1)*t_reduced_shape[0],1), 
                                 slice(y_ind*t_reduced_shape[1] if sh_bool2[1] else y_ind*(t_reduced_shape[0]-1),
                                       (y_ind+1)*t_reduced_shape[1],1), 
                                 slice(None,None,1)]
                    cropped_stack = down_stack[ind_slice].copy()
                    down_meta.comment = meta_comment+'; Cropped; Crop_index=[{},{}]'.format(x_ind,y_ind)
                    util.stack_loader.write_image_stack(cropped_stack, crop_path, f_ident, 0, meta_data= down_meta.toList())  
                
      
    
#----------------------------------------------------------------------
def crop_convoluted_stacks_overlap():
    """"""
    root_path = util.SIM_CONVOLUTED
    
    #Simple Gaussian
    #----------------------------------------------------------------------
    
    sig = [0.5, 0.25, 0.15, 0.05]
    #sig = []
    
    path_ident = 'cyto_gauss_simple_sig-{:.2f}'
    f_ident = 'cyto_conv_gauss_simple_{:0>3}.tif'
    
    crop_diviser = 3
    
    for s in sig:
        cur_path = util.ptjoin(root_path, path_ident.format(s))
        print('Cropping path: {}'.format(cur_path))

        cur_path_full = util.ptjoin(cur_path, 'full_res_[0.4,0.4,0.4]')
        cur_path_down = util.ptjoin(cur_path, 'downsampled_res_[0.4,0.4,0.4]')
        
        #cur_path_full = util.ptjoin(cur_path, 'full_res_[0.2,0.2,0.2]')
        #cur_path_down = util.ptjoin(cur_path, 'downsampled_res_[0.2,0.2,0.2]')        
        
        crop_paths_full = ['full_overlap_crop_{}_{}_res_[0.4,0.4,0.4]'.format(i/crop_diviser, i%crop_diviser) for i in range(crop_diviser**2)]
        crop_paths_down = ['downsampled_overlap_crop_{}_{}_res_[0.4,0.4,0.4]'.format(i/crop_diviser, i%crop_diviser) for i in range(crop_diviser**2)]
        
        #crop_paths_full = ['full_overlap_crop_{}_{}_res_[0.2,0.2,0.2]'.format(i/crop_diviser, i%crop_diviser) for i in range(crop_diviser**2)]
        #crop_paths_down = ['downsampled_overlap_crop_{}_{}_res_[0.2,0.2,0.2]'.format(i/crop_diviser, i%crop_diviser) for i in range(crop_diviser**2)]        
        
        crop_paths_full = [util.ptjoin(cur_path, t_pt) for t_pt in crop_paths_full]
        crop_paths_down = [util.ptjoin(cur_path, t_pt) for t_pt in crop_paths_down]
        
        util.createAllPaths(crop_paths_full)
        util.createAllPaths(crop_paths_down)
        
        full_stack = util.stack_loader.read_image_stack(cur_path_full, f_ident, meta= False)
        full_meta = util.stack_loader.read_meta_data_only(cur_path_full, f_ident)
        
        t_reduced_shape = [sh/(crop_diviser-1) if sh%2 == 0 else 1+sh/(crop_diviser-1) for sh in full_stack.shape]
        meta_comment = full_meta.comment        
        
        for i,crop_path in enumerate(crop_paths_full):
            x_ind = i/crop_diviser
            y_ind = i%crop_diviser
            sh_bool2 = [sh%2 == 0 for sh in t_reduced_shape]

            ind_slice = [slice(x_ind*(t_reduced_shape[0]/2) if sh_bool2[0] else x_ind*((t_reduced_shape[0]-1)/2),
                               (x_ind+2)*(t_reduced_shape[0]/2) if sh_bool2[0] else 1+(x_ind+2)*(t_reduced_shape[0]-1)/2,
                               1), 
                         slice(y_ind*(t_reduced_shape[1]/2) if sh_bool2[1] else y_ind*((t_reduced_shape[1]-1)/2),
                               (y_ind+2)*(t_reduced_shape[1]/2) if sh_bool2[1] else 1+(y_ind+2)*(t_reduced_shape[1]-1)/2, 
                               1),
                         slice(None,None,1)]
            
            print(ind_slice)
            cropped_stack = full_stack[ind_slice].copy()
            full_meta.comment = meta_comment+'; Cropped; Crop_index=[{},{}]'.format(x_ind,y_ind)
            util.stack_loader.write_image_stack(cropped_stack, crop_path, f_ident, 0, meta_data= full_meta.toList())
        
        down_stack = util.stack_loader.read_image_stack(cur_path_down, f_ident, meta= False)
        down_meta = util.stack_loader.read_meta_data_only(cur_path_down, f_ident)
        
        t_reduced_shape = [sh/(crop_diviser-1) if sh%2 == 0 else 1+sh/(crop_diviser-1) for sh in full_stack.shape]
        meta_comment = down_meta.comment              
        
        for crop_path in crop_paths_down:
            x_ind = i/crop_diviser
            y_ind = i%crop_diviser
            sh_bool2 = [sh%2 == 0 for sh in t_reduced_shape]
            ind_slice = [slice(x_ind*(t_reduced_shape[0]/2) if sh_bool2[0] else x_ind*((t_reduced_shape[0]-1)/2),
                               (x_ind+2)*(t_reduced_shape[0]/2) if sh_bool2[0] else 1+(x_ind+2)*(t_reduced_shape[0]-1)/2,
                               1), 
                         slice(y_ind*(t_reduced_shape[1]/2) if sh_bool2[1] else y_ind*((t_reduced_shape[1]-1)/2),
                               (y_ind+2)*(t_reduced_shape[1]/2) if sh_bool2[1] else 1+(y_ind+2)*(t_reduced_shape[1]-1)/2, 
                               1),
                         slice(None,None,1)]
            cropped_stack = down_stack[ind_slice].copy()
            down_meta.comment = meta_comment+'; Cropped; Crop_index=[{},{}]'.format(x_ind,y_ind)
            util.stack_loader.write_image_stack(cropped_stack, crop_path, f_ident, 0, meta_data= down_meta.toList())        
        
        
        
        
    
        
        
        
    
    #Gaussian Defocus
    #----------------------------------------------------------------------
    
    sig0 = [0.2, 0.1, 0.05]
    sig1 = [0.]
    sig2 = [0.4, 0.25, 0.20, 0.15]
    
    path_ident = 'cyto_gauss_sig0-{:.2f}_sig1-{:.2f}_sig2-{:.2f}'
    f_ident = 'cyto_conv_gauss_{:0>3}.tif'
    
    for s0 in sig0:
        for s1 in sig1:
            for s2 in sig2:
                cur_path = util.ptjoin(root_path, path_ident.format(s0, s1, s2))
                print('Downsampling path: {}'.format(cur_path))
                #cur_path_full = util.ptjoin(cur_path, 'full_res_[0.2,0.2,0.2]')
                #cur_path_down = util.ptjoin(cur_path, 'downsampled_res_[0.2,0.2,0.2]')
                
                #crop_paths_full = ['full_overlap_crop_{}_{}_res_[0.2,0.2,0.2]'.format(i/crop_diviser, i%crop_diviser) for i in range(crop_diviser**2)]
                #crop_paths_down = ['downsampled_overlap_crop_{}_{}_res_[0.2,0.2,0.2]'.format(i/crop_diviser, i%crop_diviser) for i in range(crop_diviser**2)]

                cur_path_full = util.ptjoin(cur_path, 'full_res_[0.4,0.4,0.4]')
                cur_path_down = util.ptjoin(cur_path, 'downsampled_res_[0.4,0.4,0.4]')
                
                crop_paths_full = ['full_overlap_crop_{}_{}_res_[0.4,0.4,0.4]'.format(i/crop_diviser, i%crop_diviser) for i in range(crop_diviser**2)]
                crop_paths_down = ['downsampled_overlap_crop_{}_{}_res_[0.4,0.4,0.4]'.format(i/crop_diviser, i%crop_diviser) for i in range(crop_diviser**2)]

                
                crop_paths_full = [util.ptjoin(cur_path, t_pt) for t_pt in crop_paths_full]
                crop_paths_down = [util.ptjoin(cur_path, t_pt) for t_pt in crop_paths_down]
                
                util.createAllPaths(crop_paths_full)
                util.createAllPaths(crop_paths_down)
                
                full_stack = util.stack_loader.read_image_stack(cur_path_full, f_ident, meta= False)
                full_meta = util.stack_loader.read_meta_data_only(cur_path_full, f_ident)
                
                t_reduced_shape = [sh/(crop_diviser-1) if sh%2 == 0 else 1+sh/(crop_diviser-1) for sh in full_stack.shape]
                meta_comment = full_meta.comment        
                
                for i,crop_path in enumerate(crop_paths_full):
                    x_ind = i/crop_diviser
                    y_ind = i%crop_diviser
                    sh_bool2 = [sh%2 == 0 for sh in t_reduced_shape]
        
                    ind_slice = [slice(x_ind*(t_reduced_shape[0]/2) if sh_bool2[0] else x_ind*((t_reduced_shape[0]-1)/2),
                                       (x_ind+2)*(t_reduced_shape[0]/2) if sh_bool2[0] else 1+(x_ind+2)*(t_reduced_shape[0]-1)/2,
                                       1), 
                                 slice(y_ind*(t_reduced_shape[1]/2) if sh_bool2[1] else y_ind*((t_reduced_shape[1]-1)/2),
                                       (y_ind+2)*(t_reduced_shape[1]/2) if sh_bool2[1] else 1+(y_ind+2)*(t_reduced_shape[1]-1)/2, 
                                       1),
                                 slice(None,None,1)]
                    
                    print(ind_slice)
                    cropped_stack = full_stack[ind_slice].copy()
                    full_meta.comment = meta_comment+'; Cropped; Crop_index=[{},{}]'.format(x_ind,y_ind)
                    util.stack_loader.write_image_stack(cropped_stack, crop_path, f_ident, 0, meta_data= full_meta.toList())
                
                down_stack = util.stack_loader.read_image_stack(cur_path_down, f_ident, meta= False)
                down_meta = util.stack_loader.read_meta_data_only(cur_path_down, f_ident)
                
                t_reduced_shape = [sh/(crop_diviser-1) if sh%2 == 0 else 1+sh/(crop_diviser-1) for sh in full_stack.shape]
                meta_comment = down_meta.comment              
                
                for crop_path in crop_paths_down:
                    x_ind = i/crop_diviser
                    y_ind = i%crop_diviser
                    sh_bool2 = [sh%2 == 0 for sh in t_reduced_shape]
                    ind_slice = [slice(x_ind*(t_reduced_shape[0]/2) if sh_bool2[0] else x_ind*((t_reduced_shape[0]-1)/2),
                                       (x_ind+2)*(t_reduced_shape[0]/2) if sh_bool2[0] else 1+(x_ind+2)*(t_reduced_shape[0]-1)/2,
                                       1), 
                                 slice(y_ind*(t_reduced_shape[1]/2) if sh_bool2[1] else y_ind*((t_reduced_shape[1]-1)/2),
                                       (y_ind+2)*(t_reduced_shape[1]/2) if sh_bool2[1] else 1+(y_ind+2)*(t_reduced_shape[1]-1)/2, 
                                       1),
                                 slice(None,None,1)]
                    cropped_stack = down_stack[ind_slice].copy()
                    down_meta.comment = meta_comment+'; Cropped; Crop_index=[{},{}]'.format(x_ind,y_ind)
                    util.stack_loader.write_image_stack(cropped_stack, crop_path, f_ident, 0, meta_data= down_meta.toList())                       
                
                             
                
    
    #Fourier Defocus
    #----------------------------------------------------------------------
    
    sigF = [0.3, 0.1, 0.05]
    zi = [2000, 800]
    K = [400, 200, 100]    

    f_ident = 'cyto_conv_defocus_{:0>3}.tif'
    path_ident = 'cyto_defocus_sig-{:.2f}_zi-{:}_K-{:}'

    for sf in sigF:
        for z in zi:
            for k in K:
                
                cur_path = util.ptjoin(root_path, path_ident.format(sf, z, k))
                print('Downsampling path: {}'.format(cur_path))
                #cur_path_full = util.ptjoin(cur_path, 'full_res_[0.2,0.2,0.2]')
                #cur_path_down = util.ptjoin(cur_path, 'downsampled_res_[0.2,0.2,0.2]')
                
                #crop_paths_full = ['full_overlap_crop_{}_{}_res_[0.2,0.2,0.2]'.format(i/crop_diviser, i%crop_diviser) for i in range(crop_diviser**2)]
                #crop_paths_down = ['downsampled_overlap_crop_{}_{}_res_[0.2,0.2,0.2]'.format(i/crop_diviser, i%crop_diviser) for i in range(crop_diviser**2)]
                

                cur_path_full = util.ptjoin(cur_path, 'full_res_[0.4,0.4,0.4]')
                cur_path_down = util.ptjoin(cur_path, 'downsampled_res_[0.4,0.4,0.4]')
                
                crop_paths_full = ['full_overlap_crop_{}_{}_res_[0.4,0.4,0.4]'.format(i/crop_diviser, i%crop_diviser) for i in range(crop_diviser**2)]
                crop_paths_down = ['downsampled_overlap_crop_{}_{}_res_[0.4,0.4,0.4]'.format(i/crop_diviser, i%crop_diviser) for i in range(crop_diviser**2)]

                crop_paths_full = [util.ptjoin(cur_path, t_pt) for t_pt in crop_paths_full]
                crop_paths_down = [util.ptjoin(cur_path, t_pt) for t_pt in crop_paths_down]
                
                util.createAllPaths(crop_paths_full)
                util.createAllPaths(crop_paths_down)
                
                full_stack = util.stack_loader.read_image_stack(cur_path_full, f_ident, meta= False)
                full_meta = util.stack_loader.read_meta_data_only(cur_path_full, f_ident)
                
                t_reduced_shape = [sh/(crop_diviser-1) if sh%2 == 0 else 1+sh/(crop_diviser-1) for sh in full_stack.shape]
                meta_comment = full_meta.comment        
                
                for i,crop_path in enumerate(crop_paths_full):
                    x_ind = i/crop_diviser
                    y_ind = i%crop_diviser
                    sh_bool2 = [sh%2 == 0 for sh in t_reduced_shape]
        
                    ind_slice = [slice(x_ind*(t_reduced_shape[0]/2) if sh_bool2[0] else x_ind*((t_reduced_shape[0]-1)/2),
                                       (x_ind+2)*(t_reduced_shape[0]/2) if sh_bool2[0] else 1+(x_ind+2)*(t_reduced_shape[0]-1)/2,
                                       1), 
                                 slice(y_ind*(t_reduced_shape[1]/2) if sh_bool2[1] else y_ind*((t_reduced_shape[1]-1)/2),
                                       (y_ind+2)*(t_reduced_shape[1]/2) if sh_bool2[1] else 1+(y_ind+2)*(t_reduced_shape[1]-1)/2, 
                                       1),
                                 slice(None,None,1)]
                    
                    print(ind_slice)
                    cropped_stack = full_stack[ind_slice].copy()
                    full_meta.comment = meta_comment+'; Cropped; Crop_index=[{},{}]'.format(x_ind,y_ind)
                    util.stack_loader.write_image_stack(cropped_stack, crop_path, f_ident, 0, meta_data= full_meta.toList())
                
                down_stack = util.stack_loader.read_image_stack(cur_path_down, f_ident, meta= False)
                down_meta = util.stack_loader.read_meta_data_only(cur_path_down, f_ident)
                
                t_reduced_shape = [sh/(crop_diviser-1) if sh%2 == 0 else 1+sh/(crop_diviser-1) for sh in full_stack.shape]
                meta_comment = down_meta.comment              
                
                for crop_path in crop_paths_down:
                    x_ind = i/crop_diviser
                    y_ind = i%crop_diviser
                    sh_bool2 = [sh%2 == 0 for sh in t_reduced_shape]
                    ind_slice = [slice(x_ind*(t_reduced_shape[0]/2) if sh_bool2[0] else x_ind*((t_reduced_shape[0]-1)/2),
                                       (x_ind+2)*(t_reduced_shape[0]/2) if sh_bool2[0] else 1+(x_ind+2)*(t_reduced_shape[0]-1)/2,
                                       1), 
                                 slice(y_ind*(t_reduced_shape[1]/2) if sh_bool2[1] else y_ind*((t_reduced_shape[1]-1)/2),
                                       (y_ind+2)*(t_reduced_shape[1]/2) if sh_bool2[1] else 1+(y_ind+2)*(t_reduced_shape[1]-1)/2, 
                                       1),
                                 slice(None,None,1)]
                    cropped_stack = down_stack[ind_slice].copy()
                    down_meta.comment = meta_comment+'; Cropped; Crop_index=[{},{}]'.format(x_ind,y_ind)
                    util.stack_loader.write_image_stack(cropped_stack, crop_path, f_ident, 0, meta_data= down_meta.toList())                        
                
         
          

#----------------------------------------------------------------------
def downsample_convoluted_stack():
    """"""
    
    root_path = util.SIM_CONVOLUTED
    
    
    #Simple Gaussian
    #----------------------------------------------------------------------
    
    sig = [0.5, 0.25, 0.15, 0.05]
    #sig = []
    path_ident = 'cyto_gauss_simple_sig-{:.2f}'
    f_ident = 'cyto_conv_gauss_simple_{:0>3}.tif'
    
    for s in sig:
        cur_path = util.ptjoin(root_path, path_ident.format(s))
        print('Downsampling path: {}'.format(cur_path))
        #os.rename(util.ptjoin(cur_path, 'res_[0.1,0.1,0.1]'), util.ptjoin(cur_path, 'full_res_[0.1,0.1,0.1]'))
        #os.rename(util.ptjoin(cur_path, 'res_[0.2,0.2,0.2]'), util.ptjoin(cur_path, 'full_res_[0.2,0.2,0.2]'))
        
        
        #cur_path_out = util.ptjoin(cur_path, 'downsampled_res_[0.2,0.2,0.2]')
        #cur_path_in = util.ptjoin(cur_path, 'full_res_[0.1,0.1,0.1]')
        
        #util.createAllPaths(util.ptjoin(cur_path, 'downsampled_res_[0.2,0.2,0.2]'))
        #conv_stack, conv_meta = util.stack_loader.read_image_stack(cur_path_in, f_ident, meta= True)

        cur_path_out = util.ptjoin(cur_path, 'downsampled_res_[0.4,0.4,0.4]')
        cur_path_in = util.ptjoin(cur_path, 'full_res_[0.2,0.2,0.2]')
        
        util.createAllPaths(util.ptjoin(cur_path, 'downsampled_res_[0.4,0.4,0.4]'))
        conv_stack, conv_meta = util.stack_loader.read_image_stack(cur_path_in, f_ident, meta= True)        
        
        
        cur_shape = conv_stack.shape
        new_shape = [1+(csh-1)/2 for csh in cur_shape]
        arr_hyb2 = conv_stack.copy()
        arr_hyb = np.pad(arr_hyb2, ((1,1),(1,1),(1,1)), 'edge')
        for dim in range(3):
            slic_ind1 = [slice(0,-1,2) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
            slic_ind2 = [slice(1,None,2) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
            slic_ind3 = [slice(2,None,2) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
            arr_hyb = 0.25 * arr_hyb[slic_ind1]+ 0.5 *  arr_hyb[slic_ind2] + 0.25 * arr_hyb[slic_ind3]    
        
        conv_meta.max_v = arr_hyb.max()
        out_stack = (255.*arr_hyb/arr_hyb.max()).astype('uint8')
        
        conv_meta.size = new_shape
        conv_meta.path = cur_path_out
        conv_meta.resolution = [0.2,0.2,0.2]
        conv_meta.comment = conv_meta.comment + '; Downsampled'
        
        util.stack_loader.write_image_stack(out_stack, cur_path_out, f_ident, 0, meta_data=conv_meta.toList())
        
        
        
        
        
    
    #Gaussian Defocus
    #----------------------------------------------------------------------
    
    sig0 = [0.2, 0.1, 0.05]
    sig1 = [0.]
    sig2 = [0.4, 0.25, 0.20, 0.15]
    
    path_ident = 'cyto_gauss_sig0-{:.2f}_sig1-{:.2f}_sig2-{:.2f}'
    f_ident = 'cyto_conv_gauss_{:0>3}.tif'
    
    for s0 in sig0:
        for s1 in sig1:
            for s2 in sig2:
                cur_path = util.ptjoin(root_path, path_ident.format(s0, s1, s2))
                print('Downsampling path: {}'.format(cur_path))
                #os.rename(util.ptjoin(cur_path, 'res_[0.1,0.1,0.1]'), util.ptjoin(cur_path, 'full_res_[0.1,0.1,0.1]'))
                #os.rename(util.ptjoin(cur_path, 'res_[0.2,0.2,0.2]'), util.ptjoin(cur_path, 'full_res_[0.2,0.2,0.2]'))
                
                #cur_path_out = util.ptjoin(cur_path, 'downsampled_res_[0.2,0.2,0.2]')
                #cur_path_in = util.ptjoin(cur_path, 'full_res_[0.1,0.1,0.1]')
                
                #util.createAllPaths(util.ptjoin(cur_path, 'downsampled_res_[0.2,0.2,0.2]'))
                #conv_stack, conv_meta = util.stack_loader.read_image_stack(cur_path_in, f_ident, meta= True)
                
                
                cur_path_out = util.ptjoin(cur_path, 'downsampled_res_[0.4,0.4,0.4]')
                cur_path_in = util.ptjoin(cur_path, 'full_res_[0.2,0.2,0.2]')
                
                util.createAllPaths(util.ptjoin(cur_path, 'downsampled_res_[0.4,0.4,0.4]'))
                conv_stack, conv_meta = util.stack_loader.read_image_stack(cur_path_in, f_ident, meta= True)                
                
                cur_shape = conv_stack.shape
                new_shape = [1+(csh-1)/2 for csh in cur_shape]
                arr_hyb2 = conv_stack.copy()
                arr_hyb = np.pad(arr_hyb2, ((1,1),(1,1),(1,1)), 'edge')
                for dim in range(3):
                    slic_ind1 = [slice(0,-1,2) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                    slic_ind2 = [slice(1,None,2) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                    slic_ind3 = [slice(2,None,2) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                    arr_hyb = 0.25 * arr_hyb[slic_ind1]+ 0.5 *  arr_hyb[slic_ind2] + 0.25 * arr_hyb[slic_ind3]    
                
                conv_meta.max_v = arr_hyb.max()
                out_stack = (255.*arr_hyb/arr_hyb.max()).astype('uint8')
                
                conv_meta.size = new_shape
                conv_meta.path = cur_path_out
                conv_meta.resolution = [0.2,0.2,0.2]
                conv_meta.comment = conv_meta.comment + '; Downsampled'
                
                util.stack_loader.write_image_stack(out_stack, cur_path_out, f_ident, 0, meta_data=conv_meta.toList())                
                
    
    #Fourier Defocus
    #----------------------------------------------------------------------
    
    sigF = [0.3, 0.1, 0.05]
    zi = [2000, 800]
    K = [400, 200, 100]    

    f_ident = 'cyto_conv_defocus_{:0>3}.tif'
    path_ident = 'cyto_defocus_sig-{:.2f}_zi-{:}_K-{:}'

    for sf in sigF:
        for z in zi:
            for k in K:
                
                cur_path = util.ptjoin(root_path, path_ident.format(sf, z, k))
                print('Downsampling path: {}'.format(cur_path))
                #os.rename(util.ptjoin(cur_path, 'res_[0.1,0.1,0.1]'), util.ptjoin(cur_path, 'full_res_[0.1,0.1,0.1]'))
                #os.rename(util.ptjoin(cur_path, 'res_[0.2,0.2,0.2]'), util.ptjoin(cur_path, 'full_res_[0.2,0.2,0.2]'))
                
                #cur_path_out = util.ptjoin(cur_path, 'downsampled_res_[0.2,0.2,0.2]')
                #cur_path_in = util.ptjoin(cur_path, 'full_res_[0.1,0.1,0.1]')
                
                #util.createAllPaths(util.ptjoin(cur_path, 'downsampled_res_[0.2,0.2,0.2]'))
                #conv_stack, conv_meta = util.stack_loader.read_image_stack(cur_path_in, f_ident, meta= True)

                cur_path_out = util.ptjoin(cur_path, 'downsampled_res_[0.4,0.4,0.4]')
                cur_path_in = util.ptjoin(cur_path, 'full_res_[0.2,0.2,0.2]')
                
                util.createAllPaths(util.ptjoin(cur_path, 'downsampled_res_[0.4,0.4,0.4]'))
                conv_stack, conv_meta = util.stack_loader.read_image_stack(cur_path_in, f_ident, meta= True)
                
                
                cur_shape = conv_stack.shape
                new_shape = [1+(csh-1)/2 for csh in cur_shape]
                arr_hyb2 = conv_stack.copy()
                arr_hyb = np.pad(arr_hyb2, ((1,1),(1,1),(1,1)), 'edge')
                for dim in range(3):
                    slic_ind1 = [slice(0,-1,2) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                    slic_ind2 = [slice(1,None,2) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                    slic_ind3 = [slice(2,None,2) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
                    arr_hyb = 0.25 * arr_hyb[slic_ind1]+ 0.5 *  arr_hyb[slic_ind2] + 0.25 * arr_hyb[slic_ind3]    
                
                conv_meta.max_v = arr_hyb.max()
                out_stack = (255.*arr_hyb/arr_hyb.max()).astype('uint8')
                
                conv_meta.size = new_shape
                conv_meta.path = cur_path_out
                conv_meta.resolution = [0.2,0.2,0.2]
                conv_meta.comment = conv_meta.comment + '; Downsampled'
                
                util.stack_loader.write_image_stack(out_stack, cur_path_out, f_ident, 0, meta_data=conv_meta.toList())                


#----------------------------------------------------------------------
def psf_conv_gaussian(used_res):
    """"""
    
    #used_res = [0.2,0.2,0.2]
    psf_oddity = 'odd'    
    convolution_method = 'real_fft'
    noise_type = 'no_noise'
    noise_params = None
    
    #path_res_add = 'res_[{0[0]:.1f},{0[1]:.1f},{0[2]:.1f}]'    
    
    #sample_path_in = util.ptjoin(util.SIM_DATA, 'cyto_sim_full', path_res_add.format(used_res))
    #psf_path_in = util.ptjoin(util.SIM_PSF,psf_oddity, 'gauss_sig0-{:.2f}_sig1-{:.2f}_sig2-{:.2f}', path_res_add.format(used_res))
    #conv_path_out = util.ptjoin(util.SIM_CONVOLUTED, 'cyto_gauss_sig0-{:.2f}_sig1-{:.2f}_sig2-{:.2f}', path_res_add.format(used_res))
    
    path_res_add = 'full_res_[{0[0]:.1f},{0[1]:.1f},{0[2]:.1f}]'
    path_res_add_psf = 'res_[{0[0]:.1f},{0[1]:.1f},{0[2]:.1f}]'    
    
    #sample_path_in = util.ptjoin(util.SIM_DATA, 'cyto_sim_full', path_res_add.format(used_res))
    sample_path_in = util.ptjoin(util.SIM_DATA, 'psf_recon_full', path_res_add.format(used_res))
    psf_path_in = util.ptjoin(util.SIM_PSF,psf_oddity, 'gauss_sig0-{:.2f}_sig1-{:.2f}_sig2-{:.2f}', path_res_add_psf.format(used_res))
    #conv_path_out = util.ptjoin(util.SIM_CONVOLUTED, 'cyto_gauss_sig0-{:.2f}_sig1-{:.2f}_sig2-{:.2f}', path_res_add.format(used_res))
    conv_path_out = util.ptjoin(util.SIM_CONVOLUTED, 'psf_recon_gauss_sig0-{:.2f}_sig1-{:.2f}_sig2-{:.2f}', path_res_add.format(used_res))

    
    sig0_list_full = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    sig1_list_full = [0.]
    sig2_list_full = [0.15, 0.20, 0.25, 0.30, 0.40]
    
    sig0_list = [0.05, 0.1, 0.2]
    sig1_list = [0.]
    sig2_list = [0.15, 0.20, 0.25, 0.40]
    
    psf_fident = 'psf_gauss_{:0>3}.tif'
    sample_fident = 'psf_recon_{:0>3}.tif'
    conv_fident = 'psf_recon_gauss_{:0>3}.tif'
    
    
    sample_stack, sample_meta = util.stack_loader.read_image_stack(sample_path_in, sample_fident, meta= True)
    if psf_oddity == 'odd':
        print(sample_stack.shape)
        shape_odd = [sh-1 if sh%2 == 0 else sh for sh in sample_stack.shape]
        index = [slice(None, sh, 1) for sh in shape_odd]
        sample_stack = sample_stack[index]
        print(sample_stack.shape)
        
    samp_max = sample_stack.max()
    samp_min = sample_stack.min()
    sample_stack = (-1)*(sample_stack - samp_min) + (samp_max-samp_min)    
    
    for sig0 in sig0_list:
        for sig1 in sig1_list:
            for sig2 in sig2_list:
    
                t_psf_path_in = psf_path_in.format(sig0, sig1, sig2)
                print('Processing: {}'.format(t_psf_path_in))
                t_conv_path_out = conv_path_out.format(sig0, sig1, sig2)
                t_psf_stack, t_psf_meta = util.stack_loader.read_image_stack(t_psf_path_in, psf_fident, meta=True)
                
                t_conv = convoluter.Convoluter(sample_stack, t_psf_stack, isPsfCentered= True, conv_method= convolution_method, noise_type= noise_type, noise_params= noise_params, debug_int=3, comment = 'With Fourth iteration PSFs')
                t_conv.initSaveParameters(t_conv_path_out, conv_fident, orig_img_path=[sample_path_in, sample_fident], orig_psf_path=[t_psf_path_in, psf_fident])
                t_conv.convolute()
                
                t_conv.out = (t_conv.out - (t_conv.out.max() - t_conv.out.min()))*(-1)
                t_conv.out = (t_conv.out * (samp_max-samp_min)/t_conv.out.max()) + samp_min                
                
                t_conv.saveSolution()    
    
#----------------------------------------------------------------------
def psf_conv_gaussian_simple(used_res):
    """"""
    

    #used_res = [0.1,0.1,0.1]
    psf_oddity = 'odd'    
    convolution_method = 'real_fft'
    noise_type = 'no_noise'
    noise_params = None
    
    #path_res_add = 'res_[{0[0]:.1f},{0[1]:.1f},{0[2]:.1f}]'    
    
    #sample_path_in = util.ptjoin(util.SIM_DATA, 'cyto_sim_full', path_res_add.format(used_res))
    #psf_path_in = util.ptjoin(util.SIM_PSF,psf_oddity, 'gauss_simple_sig-{:.2f}', path_res_add.format(used_res))
    #conv_path_out = util.ptjoin(util.SIM_CONVOLUTED, 'cyto_gauss_simple_sig-{:.2f}', path_res_add.format(used_res))
    
    path_res_add = 'full_res_[{0[0]:.1f},{0[1]:.1f},{0[2]:.1f}]'
    path_res_add_psf = 'res_[{0[0]:.1f},{0[1]:.1f},{0[2]:.1f}]'    
    
    #sample_path_in = util.ptjoin(util.SIM_DATA, 'cyto_sim_full', path_res_add.format(used_res))
    sample_path_in = util.ptjoin(util.SIM_DATA, 'psf_recon_full', path_res_add.format(used_res))
    psf_path_in = util.ptjoin(util.SIM_PSF,psf_oddity, 'gauss_simple_sig-{:.2f}', path_res_add_psf.format(used_res))
    #conv_path_out = util.ptjoin(util.SIM_CONVOLUTED, 'cyto_gauss_simple_sig-{:.2f}', path_res_add.format(used_res))
    conv_path_out = util.ptjoin(util.SIM_CONVOLUTED, 'psf_recon_gauss_simple_sig-{:.2f}', path_res_add.format(used_res))     
    
    
    sigma_list_full = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    
    sigma_list = [0.05, 0.15, 0.25, 0.5]
    
    psf_fident = 'psf_gauss_simple_{:0>3}.tif'
    sample_fident = 'psf_recon_{:0>3}.tif'
    conv_fident = 'psf_recon_gauss_simple_{:0>3}.tif'
    
    
    sample_stack, sample_meta = util.stack_loader.read_image_stack(sample_path_in, sample_fident, meta= True)
    

    
    if psf_oddity == 'odd':
        print(sample_stack.shape)
        shape_odd = [sh-1 if sh%2 == 0 else sh for sh in sample_stack.shape]
        index = [slice(None, sh, 1) for sh in shape_odd]
        sample_stack = sample_stack[index]
        print(sample_stack.shape)
    
    samp_max = sample_stack.max()
    samp_min = sample_stack.min()
    sample_stack = (-1)*(sample_stack - samp_min) + (samp_max-samp_min)    
    
    for sig in sigma_list:
        
        t_psf_path_in = psf_path_in.format(sig)
        print('Processing: {}'.format(t_psf_path_in))
        t_conv_path_out = conv_path_out.format(sig)
        t_psf_stack, t_psf_meta = util.stack_loader.read_image_stack(t_psf_path_in, psf_fident, meta=True)
        
        t_conv = convoluter.Convoluter(sample_stack, t_psf_stack, isPsfCentered= True, conv_method= convolution_method, noise_type= noise_type, noise_params= noise_params, debug_int=3, comment = 'With Fourth iteration PSFs')
        t_conv.initSaveParameters(t_conv_path_out, conv_fident, orig_img_path=[sample_path_in, sample_fident], orig_psf_path=[t_psf_path_in, psf_fident])
        t_conv.convolute()
        
        t_conv.out = (t_conv.out - (t_conv.out.max() - t_conv.out.min()))*(-1)
        t_conv.out = (t_conv.out * (samp_max-samp_min)/t_conv.out.max()) + samp_min
        
        t_conv.saveSolution()    
    

    
if __name__ == '__main__':
    pass
    #used_res = [0.2,0.2,0.2]
    #conv_defocus_batch(used_res)
    #conv_gaussian_simple_batch(used_res)
    #conv_gaussian_batch(used_res)
    
    #conv_defocus_batch(used_res)
    #conv_gaussian_simple_batch(used_res)
    #conv_gaussian_batch(used_res)
    

