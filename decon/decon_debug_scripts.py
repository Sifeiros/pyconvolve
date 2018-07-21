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



#----------------------------------------------------------------------
def test_abstract_decon(test_conv_path, test_conv_fident, test_psf_path, test_psf_fident):
    """"""
    
    test_conv = st_l.read_image_stack(test_conv_path, test_conv_fident)
    test_psf = st_l.read_image_stack(test_psf_path, test_psf_fident)
    
    
    test_dec = abs_decon.AbstractDecon(test_conv, psf = test_psf, sample = None, groundTruth = None, algoName = 'Abstr', solveFor = 'sample', constraints = None, 
                           isPsfCentered = True, useCpxFFT = False, debugInt=3)
    
    test_dec.print_dbg('Test0' ,0)
    test_dec.print_dbg('Test1' ,1)
    test_dec.print_dbg('Test2' ,2)
    test_dec.print_dbg('Test3' ,3)
    


#----------------------------------------------------------------------
def test_iterative(test_gr_truth_path, test_gr_truth_fident, test_conv_path, test_conv_fident, test_psf_path, test_psf_fident):
    """"""
    
    test_gr_truth, t_meta = st_l.read_image_stack(test_gr_truth_path, test_gr_truth_fident, meta=True)
    test_gr_truth = test_gr_truth[0:256,0:256,0:128]
    test_conv, t_meta = st_l.read_image_stack(test_conv_path, test_conv_fident, meta=True)
    test_psf, t_meta = st_l.read_image_stack(test_psf_path, test_psf_fident, meta=True)    

    #print(test_psf)
    
    print('Ground Truth Shape: {}'.format(test_gr_truth.shape))
    print('Convoluted Import Shape: {}'.format(test_conv.shape))
    print('PSF shape: {}'.format(test_psf.shape))
    
    test_after_conv = cn.Convoluter(test_gr_truth, test_psf, conv_method='cpx_fft', debug_int= 3)
    test_after_conv.convolute()

    #test_dec = iterative.AbstractIterative(test_after_conv.out, psf=test_psf, sample= None, groundTruth= None, solveFor= 'sample', initialGuess= 'WienerFilter', 
                                          #algoName= 'AbstractIterative', iterSteps= 1000, 
                                          #errTol= 1e-5, constraints= None, isPsfCentered= True, 
                                          #useCpxFFT= False, debugInt= 3, compareWithTruth= False)
    
    #test_dec = iterative.Gold(test_after_conv.out, psf=test_psf, initialGuess='WienerFilter', 
                             #experimentalMode=False, iterSteps=10, debugInt = 3)


    test_dec = iterative.Gold(test_after_conv.out, psf=test_psf, groundTruth=test_gr_truth, initialGuess='WienerFilter', iterSteps=10, debugInt=3, 
                              compareWithTruth=True, saveIntermediateSteps= 2)
    
    #test_dec = iterative.JannsonVCittert(test_after_conv.out, psf=test_psf, initialGuess='WienerFilter', 
                                        #iterSteps=10, errTol= 1e-5, gamma= 1., constraints= None, 
                                        #isPsfCentered =True, useCpxFFT =False, debugInt=3, 
                                        #compareWithTruth= False, saveIntermediateSteps= 0)    

    #test_dec = iterative.Landweber(test_after_conv.out, psf=test_psf, initialGuess= 'WienerFilter',
                                   #iterSteps=10, errTol= 1e-5, gamma=1., isPsfCentered=True, useCpxFFT=False,
                                   #debugInt=3, compareWithTruth=False, saveIntermediateSteps=0)
    
    #test_dec = iterative.RichardsonLucy(test_after_conv.out, psf=test_psf, sample =None, groundTruth =None, solveFor ='sample', initialGuess ='orig-array', 
                                       #iterSteps =10, errTol =1e-5, p =1., constraints =None, 
                                       #isPsfCentered =True, useCpxFFT =False, debugInt =3, compareWithTruth= False, saveIntermediateSteps= 0) 
    

    #test_dec = iterative.StarkParker(test_after_conv.out, psf=test_psf, sample= None, groundTruth= None, solveFor= 'sample', initialGuess= 'WienerFilter', 
                                    #iterSteps=10, errTol= 1e-5, gamma= 1.0, vmin= -1, vmax= -1, 
                                    #constraints= None, isPsfCentered= True, useCpxFFT= False, debugInt= 3, 
                                    #compareWithTruth= False, saveIntermediateSteps= 0)
                                    
    #test_dec = iterative.TikhonovMiller(test_after_conv.out, psf=test_psf, sample= None, groundTruth= None, solveFor= 'sample', initialGuess= 'WienerFilter', 
                                       #iterSteps=10, errTol= 1e-5, gamma= 1., lamb= 1., 
                                       #constraints= None, isPsfCentered= True, useCpxFFT= False, 
                                       #debugInt= 3, compareWithTruth= False, saveIntermediateSteps= 0)
                                       
    #test_dec = iterative.ICTMi (test_after_conv.out, ps Int= 3, compareWithTruth= False, saveIntermediateSteps= 0)
    
    #test_dec.initSaveParameters(save_path, save_fident, intermediate_paths=['inter[IND]', None], None], None], None], None])
    save_path = util.ptjoin(util.SIM_RECON, 'test_decon_iter', 'Gold')
    f_ident = 'recon_wiener_{:0>3}.tif'
    
    util.path_declarations.createAllPaths(save_path)
    
    test_dec.initSaveParameters(save_path, f_ident, intermediate_path= 'inter_[IND]', orig_img_path=[test_conv_path, test_conv_fident], orig_psf_path=[test_psf_path, test_psf_fident], 
                                orig_sample_path=[None, None], orig_truth_path=[test_gr_truth_path, test_gr_truth_fident], overwrite=False)    
    
    test_dec.prepare()
    test_dec.solve()
    
    test_dec.saveSolution()
    
    
    
#----------------------------------------------------------------------
def test_single_step(test_gr_truth_path, test_gr_truth_fident, test_conv_path, test_conv_fident, test_psf_path, test_psf_fident):
    """"""
    
    test_gr_truth, t_meta = st_l.read_image_stack(test_gr_truth_path, test_gr_truth_fident, meta=True)
    test_gr_truth = test_gr_truth[0:256,0:256,0:128]
    test_conv, t_meta = st_l.read_image_stack(test_conv_path, test_conv_fident, meta=True)
    test_psf, t_meta = st_l.read_image_stack(test_psf_path, test_psf_fident, meta=True)    

    #print(test_psf)
    
    print('Ground Truth Shape: {}'.format(test_gr_truth.shape))
    print('Convoluted Import Shape: {}'.format(test_conv.shape))
    print('PSF shape: {}'.format(test_psf.shape))
    
    test_after_conv = cn.Convoluter(test_gr_truth, test_psf, conv_method='cpx_fft', debug_int= 3)
    test_after_conv.convolute()
    

    #test_dec = sing_st.InverseFilter(test_after_conv.out, groundTruth= test_gr_truth, psf=test_psf, solveFor= 'sample', isPsfCentered= True, cutoff= 1e-3, 
                                    #relativeCutoff= True, cutoffInFourierDomain= True, constraints= None, 
                                    #useCpxFFT= False, debugInt=3, compareWithTruth= True)
    
    #test_dec = sing_st.WienerFilter(test_after_conv.out, groundTruth= test_gr_truth, psf=test_psf, noise = 1e-2,
                                   #relativeNoise= True, constraints= None, useCpxFFT= False, debugInt= 3, 
                                   #compareWithTruth= True)
                                   
    test_dec = sing_st.RegularizedTikhonov(test_after_conv.out, psf=test_psf, sample= None, groundTruth= None, 
                                           solveFor= 'sample', isPsfCentered= True, lam= 1e-3, 
                                           tolerance= 1e-6, constraints= None, useCpxFFT= False, 
                                           debugInt=3, compareWithTruth= False)

    test_dec.prepare()
    test_dec.solve()
    
    save_path = util.ptjoin(util.SIM_RECON, 'test_decon_single', 'Tikhonov')
    f_ident = 'recon_wiener_{:0>3}.tif'
    
    util.path_declarations.createAllPaths(save_path)
    
    test_dec.initSaveParameters(save_path, f_ident, orig_img_path=[test_conv_path, test_conv_fident], orig_psf_path=[test_psf_path, test_psf_fident], 
                                orig_sample_path=[None, None], orig_truth_path=[test_gr_truth_path, test_gr_truth_fident], overwrite=False)
    
    test_dec.saveSolution()
    
    #conv = np.pad(test_after_conv.out, [[8,8],[8,8],[8,8]], mode='constant', constant_values = 0)
    #conv = test_after_conv.out.copy()
    #conv = test_after_conv.out.copy()
    #psf = np.pad(test_psf, [[120,120],[120,120],[56,56]], mode='constant', constant_values = 0)
    #psf = np.pad(test_psf, [[0,240],[0,240],[0,112]], mode='constant', constant_values = 0)
    
    
    #psf = np.pad(test_psf, [[128,128],[128,128],[64,64]], mode='constant', constant_values = 0)
    #psf = np.fft.ifftshift(psf)
    
    #fig, ani = util.visu_util.imshow3D_ani(psf)
    #plt.show()
    
    #f_conv = np.fft.fftshift(np.fft.fftn(conv))
    #f_psf = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(psf)))
    #f_psf = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(psf)))
    
    #mag = f_psf*f_psf.conj()
    #mag[mag < 1e-6] = 1e-6

    
    #f_recon = f_recon / mag
    

    #recon = np.fft.ifftn(np.fft.ifftshift(f_recon)).real
    
    #fig, ani = util.visu_util.imshow3D_ani(recon[8:256+9,8:256+9,8:128+9])
    #fig, ani = util.visu_util.imshow3D_ani(test_dec.out)
    plt.show()

    
    
#----------------------------------------------------------------------
def main():
    """"""
    test_gr_truth_path = util.ptjoin(util.SIM_DATA, 'cyto_sim_full', 'full_crop_0_0_res_[0.2,0.2,0.2]')
    test_gr_truth_fident = 'cyto_sim_{:0>3}.tif'
    test_conv_path = util.ptjoin(util.SIM_CONVOLUTED, 'cyto_gauss_simple_sig-0.50', 'full_crop_0_0_res_[0.2,0.2,0.2]')
    test_conv_fident = 'cyto_conv_gauss_simple_{:0>3}.tif'
    #test_psf_path = util.ptjoin(util.SIM_PSF, 'odd', 'gauss_simple_sig-0.50', 'res_[0.2,0.2,0.2]')
    test_psf_path = util.ptjoin(util.SIM_PSF, 'even', 'gauss_simple_sig-0.50', 'res_[0.2,0.2,0.2]')
    test_psf_fident = 'psf_gauss_simple_{:0>3}.tif'
    
    #test_abstract_decon(test_conv_path, test_conv_fident, test_psf_path, test_psf_fident)
    #test_single_step(test_gr_truth_path, test_gr_truth_fident, test_conv_path, test_conv_fident, test_psf_path, test_psf_fident)
    test_iterative(test_gr_truth_path, test_gr_truth_fident, test_conv_path, test_conv_fident, test_psf_path, test_psf_fident)
    
    
    
    
    
    

    
if __name__ == '__main__':
    main()
    


