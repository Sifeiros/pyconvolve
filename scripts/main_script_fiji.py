#!/usr/bin/env python
#coding:utf-8
"""
  Author:  Lukas Küpper - Sifeiros --<>
  Purpose: 
  Created: 01.05.2017
"""

import os
from os.path import join as ptjoin
import numpy as np
import numpy.random as np_ra
import numpy.linalg as np_la
import scipy as sp
import scipy.misc as sp_mi
import scipy.linalg as sp_lin
import matplotlib.pyplot as plt
import create_simulation as crs
import psf_generator as psf_g
import decon_algorithms_rework as da
import utilities as util
import tiff_loader as tiff_l
import tiff_stack as tiff_s
from libtiff import TIFF

RECON_ALGOS = [['gold', False],
               ['inverse', True],
               #'iter_tikhonov', 
               ['jannson', True],
               #'log_likelihood', 
               #['one_step_late', ]
               ['richardson', True],
               #['tikhonov', ],
               #['split_gradient', ],
               ['wiener', True],
               ['blind_richardson', True]
               ]

if os.environ['COMPUTERNAME'] == 'SEVERUS':
    DIR_ROOT = 'C:\\Users\\lukas\\Master FZJ\\TestData\\'
    print('Using Surface Book Path specifications.')
else:
    DIR_ROOT = 'K:\\Masterarbeit FZJ\\TestData'
    print('Using Desktop Path specifications.')



DIR_RECON_IMG = os.path.join(DIR_ROOT, 'reconstructed')
#DIR_RECON_PSF = os.path.join(DIR_ROOT, 'reconstructed\\psf')

DIR_DICT = {'bars_p15': {'orig':['Bars', 'Z{:0>3}.tif', 128], 'psf':['PSF-Bars', 'PSF-Bars-Z{:0>3}.tif', 128], 'conv':['Bars-G10-P15', 'Z{:0>3}.tif', 128]},
            'bars_p30': {'orig':['Bars', 'Z{:0>3}.tif', 128], 'psf':['PSF-Bars', 'PSF-Bars-Z{:0>3}.tif', 128], 'conv':['Bars-G10-P30', 'Z{:0>3}.tif', 128]},
            'elegans-cy3': {'psf':['PSF-CElegans-CY3', 'PSF-CY3-Z{:0>3}.tif', 104], 'conv':['CElegans-CY3', 'Data-CY3-Z{:0>3}.tif', 104]},
            'elegans-dapi': {'psf':['PSF-CElegans-DAPI', 'PSF-DAPI-Z{:0>3}.tif', 104], 'conv':['CElegans-DAPI', 'Data-DAPI-Z{:0>3}.tif',104]},
            'elegans-fitc': {'psf':['PSF-CElegans-FITC', 'PSF-FITC-Z{:0>3}.tif', 104], 'conv':['CElegans-FITC', 'Data-FITC-Z{:0>3}.tif',104]},
            }


def deconFiji():
    
    print('Starting Deconvolution of DeconvolutionLab data sets...')
    print('--------------------------------------------------------')
    print('Used algorithms for deconvolution:')
    for algo in [alg[0] for alg in RECON_ALGOS if alg[1]]:
        print(algo)
    
    for k in DIR_DICT.keys():
        
        cur_out_dir = ptjoin(DIR_RECON_IMG, k)
        print('Starting with {}-dataset.'.format(k))
        
        if not os.path.isdir(cur_out_dir):
            os.mkdir(cur_out_dir)
            
        print('Loading PSF and Convoluted Stacks...')
        psf_stack = tiff_l.readTiffStack(ptjoin(DIR_ROOT, DIR_DICT[k]['psf'][0]), DIR_DICT[k]['psf'][1], 0, DIR_DICT[k]['psf'][2])
        conv_stack = tiff_l.readTiffStack(ptjoin(DIR_ROOT, DIR_DICT[k]['conv'][0]), DIR_DICT[k]['conv'][1], 0, DIR_DICT[k]['conv'][2])
        print('Done')
        
        
        orig_exists = 'orig' in DIR_DICT[k]
        
        if orig_exists:
            print('Loading original Stack...')
            orig_stack = tiff_l.readTiffStack(ptjoin(DIR_ROOT, DIR_DICT[k]['orig'][0]), DIR_DICT[k]['orig'][1], 0, DIR_DICT[k]['orig'][2])
            print('Done')
        
        print('Starting decon:')
        print('-----')
        for algo in [alg[0] for alg in RECON_ALGOS if alg[1]]:
            
            cur_dir = ptjoin(cur_out_dir, algo)
            if not os.path.isdir(cur_dir):
                os.mkdir(cur_dir)
                
            
            if algo == 'gold':
                print('Using Gold Algorithm')
                decon = da.Gold(conv_stack, 
                                psf_stack, 
                                useCpxFFT=False, 
                                iniGuessAlgo=GLD_ORIGARRAY, 
                                correctZeros=True, 
                                maxIter=1e3, 
                                errTol=1e-5, 
                                p=1)
                
            elif algo == 'inverse':
                print('Using Inverse Filter')
                decon = da.InverseFilter(conv_stack, 
                                         psf_stack, 
                                         psf_cutoff=0.001, 
                                         isPSFCutoffRelative=True, 
                                         useCpxFFT=False)
                
            elif algo == 'jannson':
                print('Using Jannson-van-Cittert Algorithm')
                decon = da.JannsonVCittert(conv_stack, 
                                           psf_stack, 
                                           useCpxFFT=False, 
                                           correctZeros=True, 
                                           maxIter=1e3, 
                                           errTol=1e-5, 
                                           p=1)
            elif algo == 'richardson':
                print('Using Richardson-Lucy Algorithm')
                decon = da.RichardsonLucy(conv_stack, 
                                          psf_stack, 
                                          useCpxFFT=False, 
                                          correctZeros=False, 
                                          maxIter=1e3, 
                                          errTol=1e-5, 
                                          p=1)
            elif algo == 'wiener':
                print('Using Wiener Inverse Filter')
                decon = da.WienerFilter(conv_stack, 
                                        psf_stack, 
                                        useCpxFFT=False, 
                                        cutoff_noise=0.001, 
                                        noiseRelative=True)
                
            elif algo == 'blind_richardson':
                pass
            
            decon.debug = True
            print('Starting deconvolution...')
            decon.solve()
            print('Done.')
            print('Saving results to {}'.format(cur_dir))
            t_stack = tiff_s.TiffStack(decon.out, 
                                       cur_dir,
                                       'recon_'+algo+'_'+DIR_DICT[k]['conv'][1])
            t_stack.saveTiffStack(overwrite=True)
            print('Done')
            print('-----')
            
deconFiji()