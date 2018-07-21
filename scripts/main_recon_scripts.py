# -*- coding: utf-8 -*-
"""
Part of the pyconvolve framework for convolution and deconvolution. 
Author: Lukas Küpper, 2018
License: GPLv3
"""
import pytiff
import util
import util.path_declarations
import util.stack_loader as st_l
import util.visu_util

import decon.abstract_decon as abs_decon
import decon.iterative as iterative
import decon.single_step as sing_st
import decon.wavelet as wave

import sim_creator.convoluter as cn

import os
import numpy as np
import math as mt

import matplotlib.pyplot as plt
import matplotlib.animation as anim

from util.timer import Timer

#----------------------------------------------------------------------
def reconstruct_psf_influence_type_single_step():
    """"""
    
    gauss_single = [0.5, 0.15, 0.25, 0.5]
    gauss_multi = [[0.05, 0., 0.15],
                   [0.05, 0., 0.2],
                   [0.05, 0., 0.25],
                   [0.05, 0., 0.4],
                   [0.1, 0., 0.15],
                   [0.1, 0., 0.2],
                   [0.1, 0., 0.25],
                   [0.1, 0., 0.4],
                   [0.2, 0., 0.15],
                   [0.2, 0., 0.2],
                   [0.2, 0., 0.25],
                   [0.2, 0., 0.4]]
    
    #defocus = [[0.3, 2000, 400],
               #[0.3, 2000, 200],
               #[0.3, 2000, 100],
               #[0.3, 800, 400],
               #[0.3, 800, 200],
               #[0.3, 800, 100],
               #[0.1, 2000, 400],
               #[0.1, 2000, 200],
               #[0.1, 2000, 100],
               #[0.1, 800, 400],
               #[0.1, 800, 200],
               #[0.1, 800, 100],
               #[0.05, 2000, 400],
               #[0.05, 2000, 200],
               #[0.05, 2000, 100],
               #[0.05, 800, 400],
               #[0.05, 800, 200],
               #[0.05, 800, 100]]
    
    
    dict_paths = []
    
    orig_path = util.ptjoin(util.SIM_DATA, 'cyto_sim_full', 'full_res_[0.4,0.4,0.4]')
    orig_fident = 'cyto_sim_{:0>3}.tif'
    
    for fl in gauss_single:
        conv_path = 'cyto_gauss_simple_sig-{:.2f}'.format(fl)
        conv_f_ident = 'cyto_conv_gauss_simple_{:0>3}.tif'
        
        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res_[0.2,0.2,0.2]')
        
        psf_path = 'gauss_simple_sig-{:.2f}'.format(fl)
        psf_f_ident = 'psf_gauss_simple_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')
        
        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})
        
    
    for lis in gauss_multi:
        conv_path = 'cyto_gauss_sig0-{0[0]:.2f}_sig1-{0[1]:.2f}_sig2-{0[2]:.2f}'.format(lis)
        conv_f_ident = 'cyto_conv_gauss_{:0>3}.tif'
        
        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res[0.2,0.2,0.2]')
        
        psf_path = 'gauss_sig0-{0[0]:.2f}_sig1-{0[1]:.2f}_sig2-{0[2]:.2f}'.format(lis)
        psf_f_ident = 'psf_gauss_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')
        
        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})
        
    
    for lis in defocus:
        conv_path = 'cyto_defocus_sig-{0[0]:.2f}_zi-{0[1]}_K-{0[2]}'.format(lis)
        conv_f_ident = 'cyto_conv_defocus_{:0>3}.tif'
        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res[0.2,0.2,0.2]')
        
        psf_path = 'defocussing_sig-{0[0]:.2f}_zi-{0[1]}_K-{0[2]}'.format(lis)
        psf_f_ident = 'psf_defocus_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')
        
        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})
        
        
        
    orig_stack, orig_meta = util.stack_loader.read_image_stack(orig_path, orig_fident, meta=True)
    
    for paths in dict_paths:
        
        conv_path = util.ptjoin(util.SIM_CONVOLUTED, paths['conv_path'], 'full_res_[0.4,0.4,0.4]')
        psf_path = util.ptjoin(util.SIM_PSF, 'odd', paths['psf_path'], 'res_[0.4,0.4,0.4]')
        
        psf_stack, psf_meta = util.stack_loader.read_image_stack(psf_path, paths['psf_fident'], meta=True)
        conv_stack, conv_meta = util.stack_loader.read_image_stack(conv_path, paths['conv_fident'], meta=True)
        
        #-------------------------------------------------------------------------------------------
        recon_path = util.ptjoin(util.SIM_RECON, 'psf_type_influence', 'InverseFilter', paths['conv_path'], 'res_[0.4,0.4,0.4]')
        util.createAllPaths(recon_path)
        recon_fident = 'recon_{:0>3}.tif'        
        
        inverse = sing_st.InverseFilter(conv_stack, psf=psf_stack, groundTruth=orig_stack, isPsfCentered= True, cutoff= 0.0005, 
                             relativeCutoff= True, cutoffInFourierDomain= True, constraints= None, 
                             useCpxFFT= False, debugInt= 3, compareWithTruth= True)
        
        inverse.initSaveParameters(recon_path, recon_fident, 
                                   orig_img_path= [conv_path, paths['conv_fident']], 
                                   orig_psf_path=[psf_path, paths['psf_fident']], 
                                   orig_sample_path=[None, None], 
                                   orig_truth_path=[orig_path, orig_fident], overwrite= True)
        
        inverse.prepare()
        inverse.solve()
        inverse.saveSolution()
        
        #-------------------------------------------------------------------------------------------
        recon_path = util.ptjoin(util.SIM_RECON, 'psf_type_influence', 'WienerFilter', paths['conv_path'], 'res_[0.4,0.4,0.4]')
        util.createAllPaths(recon_path)
        recon_fident = 'recon_{:0>3}.tif'        
        
        wiener = sing_st.WienerFilter(conv_stack, psf=psf_stack, sample= None, groundTruth=orig_stack, solveFor= 'sample', isPsfCentered= True, 
                                     debugInt=3, noise= 0.005, relativeNoise=True, compareWithTruth= True)
        
        
        wiener.initSaveParameters(recon_path, recon_fident, 
                                   orig_img_path= [conv_path, paths['conv_fident']], 
                                   orig_psf_path=[psf_path, paths['psf_fident']], 
                                   orig_sample_path=[None, None], 
                                   orig_truth_path=[orig_path, orig_fident], overwrite= True)
        
        wiener.prepare()
        wiener.solve()        
        wiener.saveSolution()
        
        #-------------------------------------------------------------------------------------------
        recon_path = util.ptjoin(util.SIM_RECON, 'psf_type_influence', 'Tikhonov', paths['conv_path'], 'res_[0.4,0.4,0.4]')
        util.createAllPaths(recon_path)
        recon_fident = 'recon_{:0>3}.tif'        
        
        tikhonov = sing_st.RegularizedTikhonov(conv_stack, psf=psf_stack, groundTruth=orig_stack, lam = 5e-3, tolerance=1e-6, debugInt=3,
                                              compareWithTruth=True)
        
        tikhonov.initSaveParameters(recon_path, recon_fident, 
                                   orig_img_path= [conv_path, paths['conv_fident']], 
                                   orig_psf_path=[psf_path, paths['psf_fident']], 
                                   orig_sample_path=[None, None], 
                                   orig_truth_path=[orig_path, orig_fident], overwrite= True)
        
        tikhonov.prepare()
        tikhonov.solve()        
        tikhonov.saveSolution()
        
    print('Done reconstructing....')
    
        
        
#----------------------------------------------------------------------
def reconstruct_psf_influence_type_iterative():
    """"""
    
    gauss_single = [0.5, 0.15, 0.25, 0.5]
    gauss_multi = [[0.05, 0., 0.15],
                   [0.05, 0., 0.2],
                   [0.05, 0., 0.25],
                   [0.05, 0., 0.4],
                   [0.1, 0., 0.15],
                   [0.1, 0., 0.2],
                   [0.1, 0., 0.25],  
                   [0.1, 0., 0.4],
                   [0.2, 0., 0.15],
                   [0.2, 0., 0.2],
                   [0.2, 0., 0.25],
                   [0.2, 0., 0.4]]
    
    #defocus = [[0.3, 2000, 400],
               #[0.3, 2000, 200],
               #[0.3, 2000, 100],
               #[0.3, 800, 400],
               #[0.3, 800, 200],
               #[0.3, 800, 100],
               #[0.1, 2000, 400],
               #[0.1, 2000, 200],
               #[0.1, 2000, 100],
               #[0.1, 800, 400],
               #[0.1, 800, 200],
               #[0.1, 800, 100],
               #[0.05, 2000, 400],
               #[0.05, 2000, 200],
               #[0.05, 2000, 100],
               #[0.05, 800, 400],
               #[0.05, 800, 200],
               #[0.05, 800, 100]]
    
    
    dict_paths = []
    
    orig_path = util.ptjoin(util.SIM_DATA, 'cyto_sim_full', 'full_res_[0.4,0.4,0.4]')
    orig_fident = 'cyto_sim_{:0>3}.tif'
    
    for fl in gauss_single:
        conv_path = 'cyto_gauss_simple_sig-{:.2f}'.format(fl)
        conv_f_ident = 'cyto_conv_gauss_simple_{:0>3}.tif'
        
        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res_[0.2,0.2,0.2]')
        
        psf_path = 'gauss_simple_sig-{:.2f}'.format(fl)
        psf_f_ident = 'psf_gauss_simple_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')
        
        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})
        
    
    for lis in gauss_multi:
        conv_path = 'cyto_gauss_sig0-{0[0]:.2f}_sig1-{0[1]:.2f}_sig2-{0[2]:.2f}'.format(lis)
        conv_f_ident = 'cyto_conv_gauss_{:0>3}.tif'
        
        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res[0.2,0.2,0.2]')
        
        psf_path = 'gauss_sig0-{0[0]:.2f}_sig1-{0[1]:.2f}_sig2-{0[2]:.2f}'.format(lis)
        psf_f_ident = 'psf_gauss_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')
        
        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})
        
    
    for lis in defocus:
        conv_path = 'cyto_defocus_sig-{0[0]:.2f}_zi-{0[1]}_K-{0[2]}'.format(lis)
        conv_f_ident = 'cyto_conv_defocus_{:0>3}.tif'
        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res[0.2,0.2,0.2]')
        
        psf_path = 'defocussing_sig-{0[0]:.2f}_zi-{0[1]}_K-{0[2]}'.format(lis)
        psf_f_ident = 'psf_defocus_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')
        
        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})
        
        
        
    orig_stack, orig_meta = util.stack_loader.read_image_stack(orig_path, orig_fident, meta=True)
    
    for paths in dict_paths:
        
        conv_path = util.ptjoin(util.SIM_CONVOLUTED, paths['conv_path'], 'full_res_[0.4,0.4,0.4]')
        psf_path = util.ptjoin(util.SIM_PSF, 'odd', paths['psf_path'], 'res_[0.4,0.4,0.4]')
        
        psf_stack, psf_meta = util.stack_loader.read_image_stack(psf_path, paths['psf_fident'], meta=True)
        conv_stack, conv_meta = util.stack_loader.read_image_stack(conv_path, paths['conv_fident'], meta=True)
        
        
        wiener = sing_st.WienerFilter(conv_stack, psf=psf_stack, groundTruth=orig_stack, 
                                      compareWithTruth=True)
        wiener.prepare()
        wiener.solve()        
        
        
        #-------------------------------------------------------------------------------------------
        recon_path = util.ptjoin(util.SIM_RECON, 'psf_type_influence', 'Gold', paths['conv_path'], 'res_[0.4,0.4,0.4]')
        util.createAllPaths(recon_path)
        recon_fident = 'recon_{:0>3}.tif'
        
        gold = iterative.Gold(conv_stack, psf=psf_stack, groundTruth=orig_stack, initialGuess='orig_array', debugInt=3, iterSteps= 100, compareWithTruth=True)
        
        gold.initSaveParameters(recon_path, recon_fident, 
                                orig_img_path=[conv_path, paths['conv_fident']], 
                                orig_psf_path=[psf_path, paths['psf_fident']], 
                                orig_sample_path=[None, None], 
                                orig_truth_path=[orig_path, orig_fident], 
                                overwrite=True)
        
        gold.prepare()
        gold.prevGuess = wiener.out.copy()
        gold.errors[0] = wiener.curError
        
        gold.solve()
        gold.saveSolution()
        
        #-------------------------------------------------------------------------------------------
        recon_path = util.ptjoin(util.SIM_RECON, 'psf_type_influence', 'ICTM', paths['conv_path'], 'res_[0.4,0.4,0.4]')
        util.createAllPaths(recon_path)
        recon_fident = 'recon_{:0>3}.tif'        
        
        ictm = iterative.ICTM(conv_stack, psf=psf_stack, groundTruth=orig_stack, initialGuess='orig_array', iterSteps=100, 
                             errTol= 1e-5, gamma= 1., lamb= 1., constraints= None, isPsfCentered= True, 
                             useCpxFFT= True, debugInt= 3, compareWithTruth= True, saveIntermediateSteps= 0)
        
        ictm.initSaveParameters(recon_path, recon_fident, 
                                orig_img_path=[conv_path, paths['conv_fident']], 
                                orig_psf_path=[psf_path, paths['psf_fident']], 
                                orig_sample_path=[None, None], 
                                orig_truth_path=[orig_path, orig_fident], 
                                overwrite=True)        
        
        
        ictm.prepare()
        ictm.prevGuess = wiener.out.copy()
        ictm.errors[0] = wiener.curError
        
        ictm.solve()
        ictm.saveSolution()
        
        #-------------------------------------------------------------------------------------------
        recon_path = util.ptjoin(util.SIM_RECON, 'psf_type_influence', 'Jannson', paths['conv_path'], 'res_[0.4,0.4,0.4]')
        util.createAllPaths(recon_path)
        recon_fident = 'recon_{:0>3}.tif'        
        
        
        jannson = iterative.JannsonVCittert(conv_stack, psf=psf_stack, groundTruth=orig_stack, initialGuess= 'orig_array',
                                 iterSteps=100, useCpxFFT= False, debugInt= 3, compareWithTruth =True, 
                                 saveIntermediateSteps= 0)
        
        jannson.initSaveParameters(recon_path, recon_fident, 
                                orig_img_path=[conv_path, paths['conv_fident']], 
                                orig_psf_path=[psf_path, paths['psf_fident']], 
                                orig_sample_path=[None, None], 
                                orig_truth_path=[orig_path, orig_fident], 
                                overwrite=True)                
        
        
        jannson.prepare()
        jannson.prevGuess = wiener.out.copy()
        jannson.errors[0] = wiener.curError
        
        jannson.solve()
        jannson.saveSolution()        
        

        #-------------------------------------------------------------------------------------------
        recon_path = util.ptjoin(util.SIM_RECON, 'psf_type_influence', 'Landweber', paths['conv_path'], 'res_[0.4,0.4,0.4]')
        util.createAllPaths(recon_path)
        recon_fident = 'recon_{:0>3}.tif'        
        
        
        landweber = iterative.Landweber(conv_stack, psf=psf_stack, groundTruth=orig_stack, initialGuess='orig_array', iterSteps=100, isPsfCentered= True, useCpxFFT= False, 
                           debugInt=3, compareWithTruth=True)
        
        landweber.initSaveParameters(recon_path, recon_fident, 
                                orig_img_path=[conv_path, paths['conv_fident']], 
                                orig_psf_path=[psf_path, paths['psf_fident']], 
                                orig_sample_path=[None, None], 
                                orig_truth_path=[orig_path, orig_fident], 
                                overwrite=True)                
        
        
        landweber.prepare()
        landweber.prevGuess = wiener.out.copy()
        landweber.errors[0] = wiener.curError
        
        landweber.solve()
        landweber.saveSolution()
        
        #-------------------------------------------------------------------------------------------
        recon_path = util.ptjoin(util.SIM_RECON, 'psf_type_influence', 'Richardson', paths['conv_path'], 'res_[0.4,0.4,0.4]')
        util.createAllPaths(recon_path)
        recon_fident = 'recon_{:0>3}.tif'        
        
        
        richardson = iterative.RichardsonLucy(conv_stack, psf=psf_stack, groundTruth=orig_stack, initialGuess='orig_array', 
                                iterSteps=100, debugInt=2, compareWithTruth= True, 
                                saveIntermediateSteps= 0)
        
        richardson.initSaveParameters(recon_path, recon_fident, 
                                orig_img_path=[conv_path, paths['conv_fident']], 
                                orig_psf_path=[psf_path, paths['psf_fident']], 
                                orig_sample_path=[None, None], 
                                orig_truth_path=[orig_path, orig_fident], 
                                overwrite=True)        
        
        richardson.prepare()
        richardson.prevGuess = wiener.out.copy()
        richardson.errors[0] = wiener.curError
        
        richardson.solve()
        richardson.saveSolution()                
        
        #-------------------------------------------------------------------------------------------
        recon_path = util.ptjoin(util.SIM_RECON, 'psf_type_influence', 'StarkParker', paths['conv_path'], 'res_[0.4,0.4,0.4]')
        util.createAllPaths(recon_path)
        recon_fident = 'recon_{:0>3}.tif'        
        
        starkparker = iterative.StarkParker(conv_stack, psf=psf_stack, groundTruth=orig_stack, initialGuess='orig_array', iterSteps=100, 
                                            debugInt= 3, compareWithTruth= True)
        
        starkparker.initSaveParameters(recon_path, recon_fident, 
                                orig_img_path=[conv_path, paths['conv_fident']], 
                                orig_psf_path=[psf_path, paths['psf_fident']], 
                                orig_sample_path=[None, None], 
                                orig_truth_path=[orig_path, orig_fident], 
                                overwrite=True)                
        
        starkparker.prepare()
        starkparker.prevGuess = wiener.out.copy()
        starkparker.errors[0] = wiener.curError
        
        starkparker.solve()
        starkparker.saveSolution()                
        
        
        #-------------------------------------------------------------------------------------------
        recon_path = util.ptjoin(util.SIM_RECON, 'psf_type_influence', 'TikhonovMiller', paths['conv_path'], 'res_[0.4,0.4,0.4]')
        util.createAllPaths(recon_path)
        recon_fident = 'recon_{:0>3}.tif'        
        
        
        miller = iterative.TikhonovMiller(conv_stack, psf=psf_stack, groundTruth=orig_stack, initialGuess='orig_array', iterSteps=100, debugInt=3, 
                                 compareWithTruth=True, useCpxFFT=True, saveIntermediateSteps= 0)

        miller.initSaveParameters(recon_path, recon_fident, 
                                orig_img_path=[conv_path, paths['conv_fident']], 
                                orig_psf_path=[psf_path, paths['psf_fident']], 
                                orig_sample_path=[None, None], 
                                orig_truth_path=[orig_path, orig_fident], 
                                overwrite=True)                
    
        miller.prepare()
        miller.prevGuess = wiener.out.copy()
        miller.errors[0] = wiener.curError
        
        miller.solve()
        miller.saveSolution()                
  
  
    print('Done reconstructing....')
    
        
        

#----------------------------------------------------------------------
def parameter_study_SingleStep():
    """"""
    
    gauss_single = [0.05, 0.15, 0.25, 0.5]
    gauss_multi = [[0.05, 0., 0.15],
                   [0.05, 0., 0.2],
                   [0.05, 0., 0.25],
                   [0.05, 0., 0.4],
                   [0.1, 0., 0.15],
                   [0.1, 0., 0.2],
                   [0.1, 0., 0.25],  
                   [0.1, 0., 0.4],
                   [0.2, 0., 0.15],
                   [0.2, 0., 0.2],
                   [0.2, 0., 0.25],
                   [0.2, 0., 0.4]]
    
    defocus = [[0.3, 2000, 400],
               [0.3, 2000, 200],
               [0.3, 2000, 100],
               [0.3, 800, 400],
               [0.3, 800, 200],
               [0.3, 800, 100],
               [0.1, 2000, 400],
               [0.1, 2000, 200],
               [0.1, 2000, 100],
               [0.1, 800, 400],
               [0.1, 800, 200],
               [0.1, 800, 100],
               [0.05, 2000, 400],
               [0.05, 2000, 200],
               [0.05, 2000, 100],
               [0.05, 800, 400],
               [0.05, 800, 200],
               [0.05, 800, 100]]    
    
    
    dict_paths = []
    
    orig_path = util.ptjoin(util.SIM_DATA, 'cyto_sim_full', 'full_res_[0.4,0.4,0.4]')
    orig_fident = 'cyto_sim_{:0>3}.tif'
    
    for fl in gauss_single:
        conv_path = 'cyto_gauss_simple_sig-{:.2f}'.format(fl)
        conv_f_ident = 'cyto_conv_gauss_simple_{:0>3}.tif'
        
        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res_[0.2,0.2,0.2]')
        
        psf_path = 'gauss_simple_sig-{:.2f}'.format(fl)
        psf_f_ident = 'psf_gauss_simple_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')
        
        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})
        
    
    for lis in gauss_multi:
        conv_path = 'cyto_gauss_sig0-{0[0]:.2f}_sig1-{0[1]:.2f}_sig2-{0[2]:.2f}'.format(lis)
        conv_f_ident = 'cyto_conv_gauss_{:0>3}.tif'
        
        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res[0.2,0.2,0.2]')
        
        psf_path = 'gauss_sig0-{0[0]:.2f}_sig1-{0[1]:.2f}_sig2-{0[2]:.2f}'.format(lis)
        psf_f_ident = 'psf_gauss_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')
        
        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})
        
    
    for lis in defocus:
        conv_path = 'cyto_defocus_sig-{0[0]:.2f}_zi-{0[1]}_K-{0[2]}'.format(lis)
        conv_f_ident = 'cyto_conv_defocus_{:0>3}.tif'
        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res[0.2,0.2,0.2]')
        
        psf_path = 'defocussing_sig-{0[0]:.2f}_zi-{0[1]}_K-{0[2]}'.format(lis)
        psf_f_ident = 'psf_defocus_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')
        
        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})
        
        
        
    orig_stack, orig_meta = util.stack_loader.read_image_stack(orig_path, orig_fident, meta=True)
    
    inverse_cutoff = [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1]
    
    wiener_param = [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1]
    
    tikhonov_lambda = [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1]
    
    for paths in dict_paths:
        
        conv_path = util.ptjoin(util.SIM_CONVOLUTED, paths['conv_path'], 'full_res_[0.4,0.4,0.4]')
        psf_path = util.ptjoin(util.SIM_PSF, 'odd', paths['psf_path'], 'res_[0.4,0.4,0.4]')
        
        psf_stack, psf_meta = util.stack_loader.read_image_stack(psf_path, paths['psf_fident'], meta=True)
        conv_stack, conv_meta = util.stack_loader.read_image_stack(conv_path, paths['conv_fident'], meta=True)
        
        
        #Inverse Filter
        #-------------------------------------------------------------------------------------------
        
        for inv_cut in inverse_cutoff:
            
            recon_path = util.ptjoin(util.SIM_RECON, 'parameter_study', 'InverseFilter', paths['conv_path'], 'res_[0.4,0.4,0.4]', 'inv_cutoff-{}'.format(inv_cut))
            util.createAllPaths(recon_path)
            recon_fident = 'recon_{:0>3}.tif'   
            
            inverse = sing_st.InverseFilter(conv_stack, psf=psf_stack, groundTruth=orig_stack, isPsfCentered= True, cutoff= inv_cut, 
                                  relativeCutoff= True, cutoffInFourierDomain= True, constraints= None, 
                                  useCpxFFT= False, debugInt= 3, compareWithTruth= True)
             
            inverse.initSaveParameters(recon_path, recon_fident, 
                                       orig_img_path= [conv_path, paths['conv_fident']], 
                                       orig_psf_path=[psf_path, paths['psf_fident']], 
                                       orig_sample_path=[None, None], 
                                       orig_truth_path=[orig_path, orig_fident], overwrite= True)
            
            inverse.prepare()
            inverse.solve()
            inverse.saveSolution()
            
        
        #Wiener Filter
        #-------------------------------------------------------------------------------------------        
        for wien_para in wiener_param:
            recon_path = util.ptjoin(util.SIM_RECON, 'parameter_study', 'WienerFilter', paths['conv_path'], 'res_[0.4,0.4,0.4]', 'wien_noise-{}'.format(wien_para))
            util.createAllPaths(recon_path)
            recon_fident = 'recon_{:0>3}.tif'     
            
            wiener = sing_st.WienerFilter(conv_stack, psf=psf_stack, sample= None, groundTruth=orig_stack, solveFor= 'sample', isPsfCentered= True, 
                                          debugInt=3, noise= wien_para, relativeNoise=True, compareWithTruth= True)
        
        
            wiener.initSaveParameters(recon_path, recon_fident, 
                                      orig_img_path= [conv_path, paths['conv_fident']], 
                                      orig_psf_path=[psf_path, paths['psf_fident']], 
                                      orig_sample_path=[None, None], 
                                      orig_truth_path=[orig_path, orig_fident], overwrite= True)     
            
            wiener.prepare()
            wiener.solve()        
            wiener.saveSolution()            
        
        
        #Tikhonov        
        #------------------------------------------------------------------------------------------        
        for lam in tikhonov_lambda:
            recon_path = util.ptjoin(util.SIM_RECON, 'parameter_study', 'Tikhonov', paths['conv_path'], 'res_[0.4,0.4,0.4]', 'tikhonov_lambda-{}'.format(lam))
            util.createAllPaths(recon_path)
            recon_fident = 'recon_{:0>3}.tif'           

     
        
            tikhonov = sing_st.RegularizedTikhonov(conv_stack, psf=psf_stack, groundTruth=orig_stack, lam = lam, tolerance=1e-6, debugInt=3,
                                                   compareWithTruth=True)
        
            tikhonov.initSaveParameters(recon_path, recon_fident, 
                                        orig_img_path= [conv_path, paths['conv_fident']], 
                                        orig_psf_path=[psf_path, paths['psf_fident']], 
                                        orig_sample_path=[None, None], 
                                        orig_truth_path=[orig_path, orig_fident], overwrite= True)
        
            tikhonov.prepare()
            tikhonov.solve()        
            tikhonov.saveSolution()
         
    
         
#----------------------------------------------------------------------
def parameterStudyStarkParker():
    """"""
    
    #gauss_single = [0.5, 0.15, 0.25, 0.5]
    gauss_single = [0.15, 0.25]
    
    gauss_multi = [[ 0.05, 0., 0.15],
                   [0.05, 0., 0.2],
                   #[0.05, 0., 0.25],
                   [0.05, 0., 0.4],
                   #[0.1, 0., 0.15],
                   [0.1, 0., 0.2],
                   #[0.1, 0., 0.25],  
                   [0.1, 0., 0.4],
                   #[0.2, 0., 0.15],
                   [0.2, 0., 0.2],
                   #[0.2, 0., 0.25],
                   [0.2, 0., 0.4]]
    
    defocus = []
    
    """
    defocus = [[0.3, 2000, 400],
               #[0.3, 2000, 200],
               [0.3, 2000, 100],
               #[0.3, 800, 400],
               [0.3, 800, 200],
               #[0.3, 800, 100],
               [0.1, 2000, 400],
               #[0.1, 2000, 200],
               #[0.1, 2000, 100],
               #[0.1, 800, 400],
               [0.1, 800, 200],
               #[0.1, 800, 100],
               [0.05, 2000, 400],
               #[0.05, 2000, 200],
               [0.05, 2000, 100],
               #[0.05, 800, 400],
               #[0.05, 800, 200],
               [0.05, 800, 100]]    
    """    
    
    dict_paths = []
    
    orig_path = util.ptjoin(util.SIM_DATA, 'cyto_sim_full', 'full_res_[0.4,0.4,0.4]')
    orig_fident = 'cyto_sim_{:0>3}.tif'
    
    for fl in gauss_single:
        conv_path = 'cyto_gauss_simple_sig-{:.2f}'.format(fl)
        conv_f_ident = 'cyto_conv_gauss_simple_{:0>3}.tif'
        
        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res_[0.2,0.2,0.2]')
        
        psf_path = 'gauss_simple_sig-{:.2f}'.format(fl)
        psf_f_ident = 'psf_gauss_simple_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')
        
        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})
        
    
    for lis in gauss_multi:
        conv_path = 'cyto_gauss_sig0-{0[0]:.2f}_sig1-{0[1]:.2f}_sig2-{0[2]:.2f}'.format(lis)
        conv_f_ident = 'cyto_conv_gauss_{:0>3}.tif'
        
        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res[0.2,0.2,0.2]')
        
        psf_path = 'gauss_sig0-{0[0]:.2f}_sig1-{0[1]:.2f}_sig2-{0[2]:.2f}'.format(lis)
        psf_f_ident = 'psf_gauss_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')
        
        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})
        
    
    for lis in defocus:
        conv_path = 'cyto_defocus_sig-{0[0]:.2f}_zi-{0[1]}_K-{0[2]}'.format(lis)
        conv_f_ident = 'cyto_conv_defocus_{:0>3}.tif'
        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res[0.2,0.2,0.2]')
        
        psf_path = 'defocussing_sig-{0[0]:.2f}_zi-{0[1]}_K-{0[2]}'.format(lis)
        psf_f_ident = 'psf_defocus_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')
        
        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})
        
        
        
    orig_stack, orig_meta = util.stack_loader.read_image_stack(orig_path, orig_fident, meta=True)    
    
    #gamma = [0.8, 0.85, 0.87, 0.9, 0.92, 0.94, 0.96, 0.98, 1., 1.02, 1.04, 1.06, 1.08, 1.1, 1.13, 1.15, 1.2]
    gamma= [0.87,0.94, 0.98, 1., 1.02, 1.06, 1.13]

    for paths in dict_paths:
        
        conv_path = util.ptjoin(util.SIM_CONVOLUTED, paths['conv_path'], 'full_res_[0.4,0.4,0.4]')
        psf_path = util.ptjoin(util.SIM_PSF, 'odd', paths['psf_path'], 'res_[0.4,0.4,0.4]')
        
        psf_stack, psf_meta = util.stack_loader.read_image_stack(psf_path, paths['psf_fident'], meta=True)
        conv_stack, conv_meta = util.stack_loader.read_image_stack(conv_path, paths['conv_fident'], meta=True)
        
        for gam in gamma:
            
            recon_path = util.ptjoin(util.SIM_RECON, 'parameter_study', 'StarkParker', paths['conv_path'], 'res_[0.4,0.4,0.4]', 'stark_gamma-{}'.format(gam))
            util.createAllPaths(recon_path)
            recon_fident = 'recon_{:0>3}.tif'    
    
            stark = iterative.StarkParker(conv_stack, psf=psf_stack, groundTruth=orig_stack, 
                                         iterSteps=300, gamma=gam, 
                                         debugInt=3, compareWithTruth=True, 
                                         saveIntermediateSteps=50)
            stark.initSaveParameters(recon_path, recon_fident, 
                   orig_img_path=[conv_path, paths['conv_fident']], 
                   orig_psf_path=[psf_path, paths['psf_fident']], 
                   orig_sample_path=[None, None], 
                   orig_truth_path=[orig_path, orig_fident], 
                   overwrite=True)   
            stark.prepare()
            stark.solve()
            stark.saveSolution()
        
         
#----------------------------------------------------------------------
def parameterStudyLandweber():
    """"""
    
    #gauss_single = [0.05, 0.15, 0.25, 0.5]
    gauss_single = [0.15,0.25]
    gauss_multi = [[0.05, 0., 0.15],
                   [0.05, 0., 0.2],
                   #[0.05, 0., 0.25],
                   [0.05, 0., 0.4],
                   #[0.1, 0., 0.15],
                   [0.1, 0., 0.2],
                   #[0.1, 0., 0.25],  
                   [0.1, 0., 0.4],
                   #[0.2, 0., 0.15],
                   [0.2, 0., 0.2],
                   #[0.2, 0., 0.25],
                   [0.2, 0., 0.4]]
    
    defocus = []
    
    """
    defocus = [[0.3, 2000, 400],
               #[0.3, 2000, 200],
               [0.3, 2000, 100],
               #[0.3, 800, 400],
               [0.3, 800, 200],
               #[0.3, 800, 100],
               [0.1, 2000, 400],
               #[0.1, 2000, 200],
               [0.1, 2000, 100],
               #[0.1, 800, 400],
               [0.1, 800, 200],
               #[0.1, 800, 100],
               [0.05, 2000, 400],
               #[0.05, 2000, 200],
               [0.05, 2000, 100],
               #[0.05, 800, 400],
               [0.05, 800, 200],
               [0.05, 800, 100]]    
    """
    
    
    dict_paths = []
    
    orig_path = util.ptjoin(util.SIM_DATA, 'cyto_sim_full', 'full_res_[0.4,0.4,0.4]')
    orig_fident = 'cyto_sim_{:0>3}.tif'
    
    for fl in gauss_single:
        conv_path = 'cyto_gauss_simple_sig-{:.2f}'.format(fl)
        conv_f_ident = 'cyto_conv_gauss_simple_{:0>3}.tif'
        
        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res_[0.2,0.2,0.2]')
        
        psf_path = 'gauss_simple_sig-{:.2f}'.format(fl)
        psf_f_ident = 'psf_gauss_simple_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')
        
        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})
        
    
    for lis in gauss_multi:
        conv_path = 'cyto_gauss_sig0-{0[0]:.2f}_sig1-{0[1]:.2f}_sig2-{0[2]:.2f}'.format(lis)
        conv_f_ident = 'cyto_conv_gauss_{:0>3}.tif'
        
        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res[0.2,0.2,0.2]')
        
        psf_path = 'gauss_sig0-{0[0]:.2f}_sig1-{0[1]:.2f}_sig2-{0[2]:.2f}'.format(lis)
        psf_f_ident = 'psf_gauss_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')
        
        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})
        
    
    for lis in defocus:
        conv_path = 'cyto_defocus_sig-{0[0]:.2f}_zi-{0[1]}_K-{0[2]}'.format(lis)
        conv_f_ident = 'cyto_conv_defocus_{:0>3}.tif'
        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res[0.2,0.2,0.2]')
        
        psf_path = 'defocussing_sig-{0[0]:.2f}_zi-{0[1]}_K-{0[2]}'.format(lis)
        psf_f_ident = 'psf_defocus_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')
        
        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})
        
        
        
    orig_stack, orig_meta = util.stack_loader.read_image_stack(orig_path, orig_fident, meta=True)    
    
    gamma = [0.8, 0.85, 0.87, 0.9, 0.92, 0.94, 0.96, 0.98, 1., 1.02, 1.04, 1.06, 1.08, 1.1, 1.13, 1.15, 1.2]
    
    for paths in dict_paths:
        
        conv_path = util.ptjoin(util.SIM_CONVOLUTED, paths['conv_path'], 'full_res_[0.4,0.4,0.4]')
        psf_path = util.ptjoin(util.SIM_PSF, 'odd', paths['psf_path'], 'res_[0.4,0.4,0.4]')
        
        psf_stack, psf_meta = util.stack_loader.read_image_stack(psf_path, paths['psf_fident'], meta=True)
        conv_stack, conv_meta = util.stack_loader.read_image_stack(conv_path, paths['conv_fident'], meta=True)
        
        for gam in gamma:
            
            recon_path = util.ptjoin(util.SIM_RECON, 'parameter_study', 'Landweber', paths['conv_path'], 'res_[0.4,0.4,0.4]', 'landweber_gamma-{}'.format(gam))
            util.createAllPaths(recon_path)
            recon_fident = 'recon_{:0>3}.tif'
            
            landweber = iterative.Landweber(conv_stack, psf=psf_stack, groundTruth=orig_stack, 
                                           iterSteps=300, gamma=gam, 
                                           debugInt=3, compareWithTruth=True, 
                                           saveIntermediateSteps=50)
            
            landweber.initSaveParameters(recon_path, recon_fident, 
                                         orig_img_path=[conv_path, paths['conv_fident']], 
                                         orig_psf_path=[psf_path, paths['psf_fident']], 
                                         orig_sample_path=[None, None], 
                                         orig_truth_path=[orig_path, orig_fident], 
                                         overwrite=True)       
            landweber.prepare()
            landweber.solve()
            landweber.saveSolution()            
    
#----------------------------------------------------------------------
def parameterStudyICTM():
    """"""
    
    #gauss_single = [0.05, 0.15, 0.25, 0.5]
    gauss_single = [0.15, 0.25]
    gauss_multi = [[0.05, 0., 0.15],
                   [0.05, 0., 0.2],
                   #[0.05, 0., 0.25],
                   [0.05, 0., 0.4],
                   #[0.1, 0., 0.15],
                   [0.1, 0., 0.2],
                   [0.1, 0., 0.25],  
                   [0.1, 0., 0.4],
                   #[0.2, 0., 0.15],
                   [0.2, 0., 0.2],
                   #[0.2, 0., 0.25],
                   [0.2, 0., 0.4]]
    
    defocus = []
    
    """
    defocus = [[0.3, 2000, 400],
               #[0.3, 2000, 200],
               [0.3, 2000, 100],
               #[0.3, 800, 400],
               [0.3, 800, 200],
               #[0.3, 800, 100],
               [0.1, 2000, 400],
               #[0.1, 2000, 200],
               #[0.1, 2000, 100],
               #[0.1, 800, 400],
               [0.1, 800, 200],
               #[0.1, 800, 100],
               [0.05, 2000, 400],
               #[0.05, 2000, 200],
               [0.05, 2000, 100],
               #[0.05, 800, 400],
               #[0.05, 800, 200],
               [0.05, 800, 100]]    
    """
    
    
    dict_paths = []
    
    orig_path = util.ptjoin(util.SIM_DATA, 'cyto_sim_full', 'full_res_[0.4,0.4,0.4]')
    orig_fident = 'cyto_sim_{:0>3}.tif'
    
    for fl in gauss_single:
        conv_path = 'cyto_gauss_simple_sig-{:.2f}'.format(fl)
        conv_f_ident = 'cyto_conv_gauss_simple_{:0>3}.tif'
        
        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res_[0.2,0.2,0.2]')
        
        psf_path = 'gauss_simple_sig-{:.2f}'.format(fl)
        psf_f_ident = 'psf_gauss_simple_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')
        
        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})
        
    
    for lis in gauss_multi:
        conv_path = 'cyto_gauss_sig0-{0[0]:.2f}_sig1-{0[1]:.2f}_sig2-{0[2]:.2f}'.format(lis)
        conv_f_ident = 'cyto_conv_gauss_{:0>3}.tif'
        
        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res[0.2,0.2,0.2]')
        
        psf_path = 'gauss_sig0-{0[0]:.2f}_sig1-{0[1]:.2f}_sig2-{0[2]:.2f}'.format(lis)
        psf_f_ident = 'psf_gauss_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')
        
        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})
        
    
    for lis in defocus:
        conv_path = 'cyto_defocus_sig-{0[0]:.2f}_zi-{0[1]}_K-{0[2]}'.format(lis)
        conv_f_ident = 'cyto_conv_defocus_{:0>3}.tif'
        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res[0.2,0.2,0.2]')
        
        psf_path = 'defocussing_sig-{0[0]:.2f}_zi-{0[1]}_K-{0[2]}'.format(lis)
        psf_f_ident = 'psf_defocus_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')
        
        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})
        
        
        
    orig_stack, orig_meta = util.stack_loader.read_image_stack(orig_path, orig_fident, meta=True)
    
    #lambdas = [0.8, 0.85, 0.87, 0.9, 0.92, 0.94, 0.96, 0.98, 1., 1.02, 1.04, 1.06, 1.08, 1.1, 1.13, 1.15, 1.2]

    lambdas = [0.87,0.94, 0.98, 1., 1.02, 1.06, 1.13]

    for paths in dict_paths:
        
        conv_path = util.ptjoin(util.SIM_CONVOLUTED, paths['conv_path'], 'full_res_[0.4,0.4,0.4]')
        psf_path = util.ptjoin(util.SIM_PSF, 'odd', paths['psf_path'], 'res_[0.4,0.4,0.4]')
        
        psf_stack, psf_meta = util.stack_loader.read_image_stack(psf_path, paths['psf_fident'], meta=True)
        conv_stack, conv_meta = util.stack_loader.read_image_stack(conv_path, paths['conv_fident'], meta=True)
               
        
        for lam in lambdas:
            
            recon_path = util.ptjoin(util.SIM_RECON, 'parameterStudy', 'RichardsonLucy', paths['conv_path'], 'res_[0.4,0.4,0.4]', 'richard_lamb-{}'.format(lam))
            util.createAllPaths(recon_path)
            recon_fident = 'recon_{:0>3}.tif'             
            
            ictm = iterative.ICTM(conv_stack, psf=psf_stack, groundTruth=orig_stack, initialGuess='orig_array', iterSteps=300, 
                      errTol= 1e-5, gamma= 1., lamb= lam, constraints= None, isPsfCentered= True, 
                      useCpxFFT= True, debugInt= 3, compareWithTruth= True, saveIntermediateSteps= 50)

            ictm.initSaveParameters(recon_path, recon_fident, 
                            orig_img_path=[conv_path, paths['conv_fident']], 
                            orig_psf_path=[psf_path, paths['psf_fident']], 
                            orig_sample_path=[None, None], 
                            orig_truth_path=[orig_path, orig_fident], 
                            overwrite=True)        


            ictm.prepare()    
            ictm.solve()
            ictm.saveSolution()            

#----------------------------------------------------------------------
def timeTests_detailed():
    """"""
    
    n_xy_start = 5
    n_z_start = 5
    
    n_xy_end = 256
    n_z_end = 96
    
    n_xy_list = range(n_xy_start, n_xy_end, 3)
    n_z_list = range(n_z_start, n_z_end, 3)
    
    fident = 'time_test_{:0>3}.tif'
    
    
    for n_xy in n_xy_list:
        for n_z in n_z_list:
            
            fident = 'time_nxy-{:0>3}_nz-{:0>3}_{}.tif'.format(n_xy, n_z, '{:0>3}')
            
            
            conv = 10*np.random.random((n_xy, n_xy, n_z))
            psf = np.random.random((n_xy, n_xy, n_z))
            
            out_path = util.ptjoin(util.SIM_RECON, 'time_scale_detailed')
            
            #wiener
            
            wiener = sing_st.WienerFilter(conv, psf=psf, debugInt=3)
            wiener.prepare()
            wiener.solve()
            wiener.initSaveParameters(util.ptjoin(out_path, 'Wiener'), fident, overwrite=True)
            wiener.saveSolution(meta_only=True)
            
            #landweber

            landweber = iterative.Landweber(conv, psf=psf, debugInt=3, iterSteps= 10, errTol=1e-10, initialGuess='orig_array')
            landweber.prepare()
            landweber.solve()
            landweber.initSaveParameters(util.ptjoin(out_path, 'Landweber'), fident, overwrite=True)
            landweber.saveSolution(meta_only=True)

#----------------------------------------------------------------------
def timeTests_algorithms():
    """"""
    
    n_xy_list = [4, 8, 12, 16, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260]
    n_z_list = [4, 8, 12, 16, 20, 25, 30, 35, 40, 45, 50, 60, 70]
    
    fident = 'time_test_{:0>3}.tif'
    out_path = util.ptjoin(util.SIM_RECON, 'time_scale_study')
    for n_xy in n_xy_list:
        
        for n_z in n_z_list:
            
            conv = np.random.random((n_xy, n_xy, n_z))
            psf = np.random.random((n_xy, n_xy, n_z))
            
            
            
            fident = 'time_nxy-{:0>3}_nz-{:0>3}_{}.tif'.format(n_xy, n_z, '{:0>3}')
            
            #simple
            
            simple = sing_st.InverseFilter(conv, psf=psf, debugInt=3)
            simple.prepare()
            simple.solve()
            simple.initSaveParameters(util.ptjoin(out_path, 'Inverse'), fident, overwrite=True)
            simple.saveSolution(meta_only=True)
            
            #wiener
            
            wiener = sing_st.WienerFilter(conv, psf=psf, debugInt=3)
            wiener.prepare()
            wiener.solve()
            wiener.initSaveParameters(util.ptjoin(out_path, 'Wiener'), fident, overwrite=True)
            wiener.saveSolution(meta_only=True)
            
            #tikhonov
            
            tikhonov = sing_st.RegularizedTikhonov(conv, psf=psf, debugInt=3)
            tikhonov.prepare()
            tikhonov.solve()
            tikhonov.initSaveParameters(util.ptjoin(out_path, 'RegTikhonov'), fident, overwrite=True)
            tikhonov.saveSolution(meta_only=True)
    
            #gold
            
            gold = iterative.Gold(conv, psf=psf, iterSteps=10, debugInt=3, initialGuess='orig_array', errTol=1e-10, useCpxFFT=True)
            gold.prepare()
            gold.solve()
            gold.initSaveParameters(util.ptjoin(out_path, 'Gold'), fident, overwrite=True)
            gold.saveSolution(meta_only=True)
            
            #ictm
            
            ictm = iterative.ICTM(conv, psf=psf, iterSteps=10, debugInt=3, initialGuess='orig_array', errTol=1e-10)
            ictm.prepare()
            ictm.solve()
            ictm.initSaveParameters(util.ptjoin(out_path, 'ICTM'), fident, overwrite=True)
            ictm.saveSolution(meta_only=True)
            
            #jannson
            
            jannson = iterative.JannsonVCittert(conv, psf=psf, iterSteps=10, debugInt=3, initialGuess='orig_array', errTol=1e-10)
            jannson.prepare()
            jannson.solve()
            jannson.initSaveParameters(util.ptjoin(out_path, 'Jannson'), fident, overwrite=True)
            jannson.saveSolution(meta_only=True)
            
            #landweber
            
            landweber = iterative.Landweber(conv, psf=psf, iterSteps=10, debugInt=3, initialGuess='orig_array', errTol=1e-10)
            landweber.prepare()
            landweber.solve()
            landweber.initSaveParameters(util.ptjoin(out_path, 'Landweber'), fident, overwrite=True)
            landweber.saveSolution(meta_only=True)
            
            #richardson
            
            richardson = iterative.RichardsonLucy(conv, psf=psf, iterSteps=10, debugInt=3, initialGuess='orig_array', errTol=1e-10)
            richardson.prepare()
            richardson.solve()
            richardson.initSaveParameters(util.ptjoin(out_path, 'Richardson'), fident, overwrite=True)
            richardson.saveSolution(meta_only=True)
            
            #starkparker
            
            stark = iterative.StarkParker(conv, psf=psf, iterSteps=10, debugInt=3, initialGuess='orig_array', errTol=1e-10)
            stark.prepare()
            stark.solve()
            stark.initSaveParameters(util.ptjoin(out_path, 'StarkParker'), fident, overwrite=True)
            stark.saveSolution(meta_only=True)
            
            #tikhonov    

            tikhonov = iterative.TikhonovMiller(conv, psf=psf, iterSteps=10, debugInt=3, initialGuess='orig_array', errTol=1e-10)
            tikhonov.prepare()
            tikhonov.solve()
            tikhonov.initSaveParameters(util.ptjoin(out_path, 'TikhonovMiller'), fident, overwrite=True)
            tikhonov.saveSolution(meta_only=True)
            

def parameter_ISTA():
    """"""
    

    #gauss_single = [0.05, 0.15, 0.25, 0.5]
    gauss_single = [0.05 , 0.15, 0.25]
    gauss_multi = [[0.05, 0., 0.15],
                   #[0.05, 0., 0.2],
                   [0.05, 0., 0.25],
                   #[0.05, 0., 0.4],
                   [0.1, 0., 0.15],
                   #[0.1, 0., 0.2],
                   [0.1, 0., 0.25],  
                   #[0.1, 0., 0.4],
                   [0.2, 0., 0.15],
                   #[0.2, 0., 0.2],
                   [0.2, 0., 0.25],
                   [0.2, 0., 0.4]]

    #defocus = []

    
    defocus = [[0.3, 2000, 400],
               #[0.3, 2000, 200],
               [0.3, 2000, 100],
               #[0.3, 800, 400],
               [0.3, 800, 200],
               #[0.3, 800, 100],
               [0.1, 2000, 400],
               #[0.1, 2000, 200],
               [0.1, 2000, 100],
               #[0.1, 800, 400],
               [0.1, 800, 200],
               #[0.1, 800, 100],
               [0.05, 2000, 400],
               #[0.05, 2000, 200],
               [0.05, 2000, 100],
               #[0.05, 800, 400],
               [0.05, 800, 200],
               [0.05, 800, 100]]    
    


    dict_paths = []

    orig_path = util.ptjoin(util.SIM_DATA, 'cyto_sim_full', 'full_res_[0.4,0.4,0.4]')
    orig_fident = 'cyto_sim_{:0>3}.tif'

    for fl in gauss_single:
        conv_path = 'cyto_gauss_simple_sig-{:.2f}'.format(fl)
        conv_f_ident = 'cyto_conv_gauss_simple_{:0>3}.tif'

        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res_[0.2,0.2,0.2]')

        psf_path = 'gauss_simple_sig-{:.2f}'.format(fl)
        psf_f_ident = 'psf_gauss_simple_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')

        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})


    for lis in gauss_multi:
        conv_path = 'cyto_gauss_sig0-{0[0]:.2f}_sig1-{0[1]:.2f}_sig2-{0[2]:.2f}'.format(lis)
        conv_f_ident = 'cyto_conv_gauss_{:0>3}.tif'

        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res[0.2,0.2,0.2]')

        psf_path = 'gauss_sig0-{0[0]:.2f}_sig1-{0[1]:.2f}_sig2-{0[2]:.2f}'.format(lis)
        psf_f_ident = 'psf_gauss_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')

        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})


    for lis in defocus:
        conv_path = 'cyto_defocus_sig-{0[0]:.2f}_zi-{0[1]}_K-{0[2]}'.format(lis)
        conv_f_ident = 'cyto_conv_defocus_{:0>3}.tif'
        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res[0.2,0.2,0.2]')

        psf_path = 'defocussing_sig-{0[0]:.2f}_zi-{0[1]}_K-{0[2]}'.format(lis)
        psf_f_ident = 'psf_defocus_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')

        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})



    orig_stack, orig_meta = util.stack_loader.read_image_stack(orig_path, orig_fident, meta=True)

    #lambdas = [0.8, 0.85, 0.87, 0.9, 0.92, 0.94, 0.96, 0.98, 1., 1.02, 1.04, 1.06, 1.08, 1.1, 1.13, 1.15, 1.2]

    lambdas = [0.87,0.94, 0.98, 1., 1.02, 1.06, 1.13]

    for paths in dict_paths:

        for lam in lambdas:
            conv_path = util.ptjoin(util.SIM_CONVOLUTED, paths['conv_path'], 'full_res_[0.4,0.4,0.4]')
            psf_path = util.ptjoin(util.SIM_PSF, 'odd', paths['psf_path'], 'res_[0.4,0.4,0.4]')
    
            psf_stack, psf_meta = util.stack_loader.read_image_stack(psf_path, paths['psf_fident'], meta=True)
            conv_stack, conv_meta = util.stack_loader.read_image_stack(conv_path, paths['conv_fident'], meta=True)
        
            recon_path = util.ptjoin(util.SIM_RECON, 'parameter_study_ISTA', 'lambda-{}'.format(lam), paths['conv_path'], 'res_[0.4,0.4,0.4]')
            util.createAllPaths(recon_path)
            recon_fident = 'recon_{:0>3}.tif'
            
            ista = wave.ISTA(conv_stack, psf=psf_stack, iterSteps=300, groundTruth=orig_stack, 
                            depth= 3, gamma= 1., lamb= lam, errTol= 1e-5, constraints= None, 
                            isPsfCentered= True, useCpxFFT= False, debugInt= 3, compareWithTruth= True, 
                            saveIntermediateSteps= 50)
            ista.initSaveParameters(recon_path, recon_fident)
            
            ista.prepare()
            ista.solve()
            ista.saveSolution()            



def parameter_FISTA():
    """"""
    

    #gauss_single = [0.05, 0.15, 0.25, 0.5]
    gauss_single = [0.05 , 0.15, 0.25, 0.5]
    gauss_multi = [[0.05, 0., 0.15],
                   [0.05, 0., 0.2],
                   [0.05, 0., 0.25],
                   #[0.05, 0., 0.4],
                   [0.1, 0., 0.15],
                   [0.1, 0., 0.2],
                   [0.1, 0., 0.25],  
                   #[0.1, 0., 0.4],
                   [0.2, 0., 0.15],
                   [0.2, 0., 0.2],
                   [0.2, 0., 0.25],
                   [0.2, 0., 0.4]]

    defocus = []

    '''
    defocus = [[0.3, 2000, 400],
               #[0.3, 2000, 200],
               [0.3, 2000, 100],
               #[0.3, 800, 400],
               [0.3, 800, 200],
               #[0.3, 800, 100],
               [0.1, 2000, 400],
               #[0.1, 2000, 200],
               [0.1, 2000, 100],
               #[0.1, 800, 400],
               [0.1, 800, 200],
               #[0.1, 800, 100],
               [0.05, 2000, 400],
               #[0.05, 2000, 200],
               [0.05, 2000, 100],
               #[0.05, 800, 400],
               [0.05, 800, 200],
               [0.05, 800, 100]]    
    
               '''

    dict_paths = []

    orig_path = util.ptjoin(util.SIM_DATA, 'cyto_sim_full', 'full_res_[0.4,0.4,0.4]')
    orig_fident = 'cyto_sim_{:0>3}.tif'

    for fl in gauss_single:
        conv_path = 'cyto_gauss_simple_sig-{:.2f}'.format(fl)
        conv_f_ident = 'cyto_conv_gauss_simple_{:0>3}.tif'

        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res_[0.2,0.2,0.2]')

        psf_path = 'gauss_simple_sig-{:.2f}'.format(fl)
        psf_f_ident = 'psf_gauss_simple_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')

        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})


    for lis in gauss_multi:
        conv_path = 'cyto_gauss_sig0-{0[0]:.2f}_sig1-{0[1]:.2f}_sig2-{0[2]:.2f}'.format(lis)
        conv_f_ident = 'cyto_conv_gauss_{:0>3}.tif'

        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res[0.2,0.2,0.2]')

        psf_path = 'gauss_sig0-{0[0]:.2f}_sig1-{0[1]:.2f}_sig2-{0[2]:.2f}'.format(lis)
        psf_f_ident = 'psf_gauss_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')

        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})


    for lis in defocus:
        conv_path = 'cyto_defocus_sig-{0[0]:.2f}_zi-{0[1]}_K-{0[2]}'.format(lis)
        conv_f_ident = 'cyto_conv_defocus_{:0>3}.tif'
        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res[0.2,0.2,0.2]')

        psf_path = 'defocussing_sig-{0[0]:.2f}_zi-{0[1]}_K-{0[2]}'.format(lis)
        psf_f_ident = 'psf_defocus_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')

        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})



    orig_stack, orig_meta = util.stack_loader.read_image_stack(orig_path, orig_fident, meta=True)

    #lambdas = [0.8, 0.85, 0.87, 0.9, 0.92, 0.94, 0.96, 0.98, 1., 1.02, 1.04, 1.06, 1.08, 1.1, 1.13, 1.15, 1.2]

    lambdas = [0.87,0.94, 0.98, 1., 1.02, 1.06, 1.13]

    for paths in dict_paths:

        for lam in lambdas:
            conv_path = util.ptjoin(util.SIM_CONVOLUTED, paths['conv_path'], 'full_res_[0.4,0.4,0.4]')
            psf_path = util.ptjoin(util.SIM_PSF, 'odd', paths['psf_path'], 'res_[0.4,0.4,0.4]')
    
            psf_stack, psf_meta = util.stack_loader.read_image_stack(psf_path, paths['psf_fident'], meta=True)
            conv_stack, conv_meta = util.stack_loader.read_image_stack(conv_path, paths['conv_fident'], meta=True)
        
            recon_path = util.ptjoin(util.SIM_RECON, 'parameter_study_FISTA', 'lambda-{}'.format(lam), paths['conv_path'], 'res_[0.4,0.4,0.4]')
            util.createAllPaths(recon_path)
            recon_fident = 'recon_{:0>3}.tif'
            
            
            fista = wave.FISTA(conv_stack, psf=psf_stack, groundTruth=orig_stack, iterSteps=200, 
                               lamb=lam, debugInt=3, compareWithTruth=True, saveIntermediateSteps=50)
            
            
            fista.initSaveParameters(recon_path, recon_fident)
            
            fista.prepare()
            fista.solve()
            fista.saveSolution()      

#----------------------------------------------------------------------
def psf_recon_study():
    """"""
    
    #gauss_single = [0.05 , 0.15, 0.25, 0.5]
    gauss_single = [0.15, 0.25]
    gauss_multi = [[0.05, 0., 0.15],
                   #[0.05, 0., 0.2],
                   [0.05, 0., 0.25],
                   #[0.05, 0., 0.4],
                   [0.1, 0., 0.15],
                   #[0.1, 0., 0.2],
                   [0.1, 0., 0.25],  
                   #[0.1, 0., 0.4],
                   [0.2, 0., 0.15],
                   #[0.2, 0., 0.2],
                   [0.2, 0., 0.25],
                   [0.2, 0., 0.4]]

    defocus = []    
    
    dict_paths = []
    
    
    orig_path = util.ptjoin(util.SIM_DATA, 'psf_recon_full', 'full_res_[0.4,0.4,0.4]')
    orig_fident = 'psf_recon_{:0>3}.tif'

    for fl in gauss_single:
        conv_path = 'psf_recon_gauss_simple_sig-{:.2f}'.format(fl)
        conv_f_ident = 'psf_recon_gauss_simple_{:0>3}.tif'

        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res_[0.2,0.2,0.2]')

        psf_path = 'gauss_simple_sig-{:.2f}'.format(fl)
        psf_f_ident = 'psf_gauss_simple_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')

        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})


    for lis in gauss_multi:
        conv_path = 'psf_recon_gauss_sig0-{0[0]:.2f}_sig1-{0[1]:.2f}_sig2-{0[2]:.2f}'.format(lis)
        conv_f_ident = 'psf_recon_gauss_{:0>3}.tif'

        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res[0.2,0.2,0.2]')

        psf_path = 'gauss_sig0-{0[0]:.2f}_sig1-{0[1]:.2f}_sig2-{0[2]:.2f}'.format(lis)
        psf_f_ident = 'psf_gauss_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')

        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})


    for lis in defocus:
        conv_path = 'psf_recon_sig-{0[0]:.2f}_zi-{0[1]}_K-{0[2]}'.format(lis)
        conv_f_ident = 'psf_recon_defocus_{:0>3}.tif'
        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res[0.2,0.2,0.2]')

        psf_path = 'defocussing_sig-{0[0]:.2f}_zi-{0[1]}_K-{0[2]}'.format(lis)
        psf_f_ident = 'psf_defocus_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')

        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})



    orig_stack, orig_meta = util.stack_loader.read_image_stack(orig_path, orig_fident, meta=True)
    
    for paths in dict_paths:
    
        conv_path = util.ptjoin(util.SIM_CONVOLUTED, paths['conv_path'], 'full_res_[0.4,0.4,0.4]')
        psf_path = util.ptjoin(util.SIM_PSF, 'odd', paths['psf_path'], 'res_[0.4,0.4,0.4]')
    
        psf_stack, psf_meta = util.stack_loader.read_image_stack(psf_path, paths['psf_fident'], meta=True)
        conv_stack, conv_meta = util.stack_loader.read_image_stack(conv_path, paths['conv_fident'], meta=True)
    
        recon_fident = 'psf_recon_{:0>3}.tif'                
    

        #Wiener
        #--------------------------------------------------------------
        recon_path = util.ptjoin(util.SIM_RECON, 'psf_reconst', 'Wiener', paths['conv_path'], 'res_[0.4,0.4,0.4]')
        util.createAllPaths(recon_path)
        
        wiener = sing_st.WienerFilter(conv_stack, sample=orig_stack, groundTruth=psf_stack, solveFor='psf', debugInt=3, 
                                     compareWithTruth=False)
        
        wiener.initSaveParameters(recon_path, recon_fident)
        wiener.prepare()
        wiener.solve()
        wiener.saveSolution()

        
        #Gold
        #--------------------------------------------------------------        
        recon_path = util.ptjoin(util.SIM_RECON, 'psf_reconst', 'Gold', paths['conv_path'], 'res_[0.4,0.4,0.4]')
        util.createAllPaths(recon_path)
        
        gold = iterative.Gold(conv_stack, sample=orig_stack, groundTruth=psf_stack, solveFor='psf', iterSteps=200, useCpxFFT=True, debugInt=3, 
                              compareWithTruth=False, initialGuess='orig_array')
        gold.initSaveParameters(recon_path, recon_fident)
        gold.prepare()
        gold.solve()
        gold.saveSolution()
        
        #Landweber
        #--------------------------------------------------------------
        
        recon_path = util.ptjoin(util.SIM_RECON, 'psf_reconst', 'Landweber', paths['conv_path'], 'res_[0.4,0.4,0.4]')
        util.createAllPaths(recon_path)
        
        landweber = iterative.Landweber(conv_stack, sample=orig_stack, groundTruth=psf_stack, solveFor='psf', initialGuess='orig_array', 
                                       iterSteps=200, debugInt=3, 
                                       compareWithTruth=False)
        landweber.initSaveParameters(recon_path, recon_fident)
        landweber.prepare()
        landweber.solve()
        landweber.saveSolution()
        
        #StarkParker
        #--------------------------------------------------------------
        
        recon_path = util.ptjoin(util.SIM_RECON, 'psf_reconst', 'StarkParker', paths['conv_path'], 'res_[0.4,0.4,0.4]')
        util.createAllPaths(recon_path)
        
        stark = iterative.StarkParker(conv_stack, sample=orig_stack, groundTruth=psf_stack, solveFor='psf', initialGuess='orig_array', 
                                     iterSteps=200, vmin=-1., vmax=-1., debugInt=3, 
                                     compareWithTruth=False)
        stark.initSaveParameters(recon_path, recon_fident)
        stark.prepare()
        stark.solve()
        stark.saveSolution()
        
        #Richardson
        #--------------------------------------------------------------
        
        recon_path = util.ptjoin(util.SIM_RECON, 'psf_reconst', 'Richardson', paths['conv_path'], 'res_[0.4,0.4,0.4]')
        util.createAllPaths(recon_path)        
        
        richardson = iterative.RichardsonLucy(conv_stack, sample=orig_stack, groundTruth=psf_stack, solveFor='psf', initialGuess='orig_array', 
                                             iterSteps=200, debugInt=3, 
                                             compareWithTruth=False)
        richardson.initSaveParameters(recon_path, recon_fident)
        richardson.prepare()
        richardson.solve()
        richardson.saveSolution()


        #Tikhonov
        #--------------------------------------------------------------
        
        recon_path = util.ptjoin(util.SIM_RECON, 'psf_reconst', 'Tikhonov', paths['conv_path'], 'res_[0.4,0.4,0.4]')
        util.createAllPaths(recon_path)        
        
        tikhonov = iterative.TikhonovMiller(conv_stack, sample=orig_stack, groundTruth=psf_stack, solveFor='psf', initialGuess='orig_array', 
                                           iterSteps=200, useCpxFFT=True,
                                           debugInt=3, compareWithTruth=False)
        tikhonov.initSaveParameters(recon_path, recon_fident)
        tikhonov.prepare()
        tikhonov.solve()
        tikhonov.saveSolution()


        #Jannson
        #--------------------------------------------------------------
        
        recon_path = util.ptjoin(util.SIM_RECON, 'psf_reconst', 'Jannson', paths['conv_path'], 'res_[0.4,0.4,0.4]')
        util.createAllPaths(recon_path)        
        
        jannson = iterative.JannsonVCittert(conv_stack, sample=orig_stack, groundTruth=psf_stack, solveFor='psf', 
                                           iterSteps=200, 
                                           debugInt=3, compareWithTruth=False)
        jannson.initSaveParameters(recon_path, recon_fident)
        jannson.prepare()
        jannson.solve()
        jannson.saveSolution()


        #ISTA
        #--------------------------------------------------------------
        
        recon_path = util.ptjoin(util.SIM_RECON, 'psf_reconst', 'ISTA', paths['conv_path'], 'res_[0.4,0.4,0.4]')
        util.createAllPaths(recon_path)
        
        ista = wave.ISTA(conv_stack, sample=orig_stack, groundTruth=psf_stack, solveFor='psf', iterSteps=200, debugInt=3, compareWithTruth=False)
        
        ista.initSaveParameters(recon_path, recon_fident)
        ista.prepare()
        ista.solve()
        ista.saveSolution()
    


if __name__ == '__main__':
    #reconstruct_psf_influence_type_single_step()
    #reconstruct_psf_influence_type_iterative()
    #parameter_study_SingleStep()
    #parameterStudyLandweber()
    #parameterStudyICTM()
    #parameterStudyStarkParker()
    #parameter_study_SingleStep()
    #timeTests_algorithms()
    #timeTests_detailed()
    #parameter_FISTA()
    psf_recon_study()