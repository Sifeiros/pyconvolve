# -*- coding: utf-8 -*-
"""
Part of the pyconvolve framework for convolution and deconvolution. 
Author: Lukas Küpper, 2018
License: GPLv3
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

import util
import util.stack_loader as sl


def analyze_recon():
    
    recon_path = util.MICRO_RECON
    
    sub_paths = ['4082', '4083', '4084', '4085']
    
    f_ident = 'Richardson-Lucy_{:0>4}_{:0>3}.tif'
    
    
    #recon_array = np.array((1350, 1600, 120))
    
    recon_raw = []
    
    for sub in sub_paths:
        
        cur_path = util.ptjoin(recon_path, sub)
        tmp_stack = sl.read_image_stack(cur_path, f_ident.format(sub, '{:0>3}'), 0, 30)
        print(tmp_stack.shape)
        recon_raw.append(tmp_stack)
        
    recon_array = np.zeros((recon_raw[0].shape[0], recon_raw[0].shape[1], 120))
    
    recon_array[:,:,0:30] = recon_raw[0][:,:,:]
    recon_array[:,:,30:60] = recon_raw[1][:,:,:]
    recon_array[:,:,60:90] = recon_raw[2][:,:,:]
    recon_array[:,:,90:120] = recon_raw[3][:,:,:]
    
    util.imshow3D_slice(recon_array[256:512,256:512,:], center_coord=[90,90,80], show_slice_lines=True, cmap='gray')
    #return util.imshow3D_ani(recon_array)
    
    
    
    
if __name__ == '__main__':
    #[fig, ani] = analyze_recon()
    analyze_recon()
    plt.show()