# -*- coding: utf-8 -*-
"""
Part of the pyconvolve framework for convolution and deconvolution. 
Author: Lukas Küpper, 2018
License: GPLv3
"""
import os
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

RECON_ALGOS = ['gold', 'inverse', 'iter_tikhonov', 'cittert', 'log_likelihood', 'one_step_late', 'richardson', 'tikhonov', 'split_gradient', 'wiener']

if os.environ['COMPUTERNAME'] == 'SEVERUS':
    DIR_ROOT = 'C:\\Users\\lukas\\Master FZJ\\Data\\'
    print('Using Surface Book Path specifications.')
else:
    DIR_ROOT = 'K:\\Masterarbeit FZJ\\Data'
    print('Using Desktop Path specifications.')
    
    
DIR_DATA = os.path.join(DIR_ROOT, 'cropped\\')
DIR_RECON_IMG = os.path.join(DIR_ROOT, 'reconstructed\\img')
DIR_RECON_PSF = os.path.join(DIR_ROOT, 'reconstructed\\psf')


def cropHuronData():
    
    huron_path = os.path.join(DIR_ROOT, 'Huron', 'aligned')
    huron_ident = 'BB01_3747_R01_Slice{:0>2}_crop.tif'
    
    huron_stack = tiff_l.readTiffStack(huron_path, huron_ident, 1, 31)
    
    cropped_size = [512,512,30]
    cropped_corner = [9000,2000,0]
    
    
    cropped_stack = huron_stack.astype('float')[cropped_corner[0]:cropped_corner[0]+cropped_size[0],cropped_corner[1]:cropped_corner[1]+cropped_size[1],:]
    
    
    cropped_stack = tiff_s.TiffStack(cropped_stack)
    cropped_stack.cropped = True
    cropped_stack.origShape = huron_stack.shape
    cropped_stack.subCropArea
    cropped_stack.path = DIR_DATA
    cropped_stack.f_ident = 'huron_cropped_{:0>3}.tif'
    cropped_stack.saveTiffStack(overwrite=True)
    
    util.imshow3D(cropped_stack, cmap='gray')
    
    print('Showing Plots')
    plt.show()
    
def blindReconCrop():
    
    cropPath = os.path.join(DIR_DATA, 'crop3')
    f_ident = 'huron_cropped_{:0>3}.tif'
    
    cropped_stack = tiff_l.readTiffStack(cropPath, f_ident, 0, 30)
    
    psf = psf_g.createPSF((512,512,30),
                          1., 
                          min_sig=(.05,.05), 
                          max_sig=(.9,.9), 
                          supp=(0.,-7.), 
                          normalize=True)
    blind = da.BlindRichardsonLucy(cropped_stack.view(np.ndarray), 
                                   psf, 
                                   useCpxFFT=False, 
                                   pad_psf=True,   
                                   correctZeros=True, 
                                   maxIter=10, 
                                   errTol=1e-5, 
                                   p=1, 
                                   norm_psf=2500.)
    blind.debug = True
    blind.solve()
    
    t_stack = tiff_s.TiffStack(blind.out, 
                           DIR_RECON_IMG,
                           'recon_img_'+f_ident)

    t_meta = ['Type: Reconstruction', 
              'Reconstruction Algorithm: Blind Richardson Lucy Algorithm', 
              'Iterations: {}'.format(blind.curIter),
              'Error: {}'.format(blind.err[0]),
              '---', 
              'Properties of Convoluted Stack:']
    

    t_stack.meta = t_meta
    t_stack.saveTiffStack(overwrite=True)
    
    t_stack = tiff_s.TiffStack(blind.out_psf, 
                               DIR_RECON_PSF,
                               'recon_psf_'+f_ident)
    
    t_meta = ['Type: PSF Reconstruction', 
              'Reconstruction Algorithm: Blind Richardson Lucy Algorithm', 
              'Iterations: {}'.format(blind.curIter),
              'Error: {}'.format(blind.err[1]),
              '---', 
              'Properties of Convoluted Stack:']
    

    t_stack.meta = t_meta
    t_stack.saveTiffStack(overwrite=True)
        
#cropHuronData()
blindReconCrop()