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
import psf_recon.image_registration as im_reg




def main():
    
    test_path = util.ptjoin(util.MICRO_DATA, 'test_crops_17.08.2017')
    f_ident = 'B21_3525_test_psf_mode_LE01_R01_Slice{:0>2}_00_30_crop.tif'
    
    reg = im_reg.Linear_Registrator(test_path, f_ident, n_min=1, max_translation= [44,44], output_shape=[256,256])
    reg.calcTranslation()
    reg.translate()
    
    reg.writeImageStack(util.ptjoin(test_path, 'aligned'), f_ident)




if __name__ == '__main__':
    main()
    
    #plt.show()
    
    #raise StandardError