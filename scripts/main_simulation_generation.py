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
import sim_creator.convolution_creation_scripts as cn_scripts
import sim_creator.psf_generator as psfg
import sim_creator.psf_generation_scripts as psfg_scripts
import sim_creator.noise_addition_scripts as noi_add_scripts

import os
import numpy as np
import math as mt


#----------------------------------------------------------------------
def main_generation():
    """"""
    
    #psfg_scripts.createDefocusBatch()
    #psfg_scripts.createGaussBatch()
    #psfg_scripts.createGaussSimpleBatch()
    
    #noi_add_scripts.PSF_noise_additions_batch()
    
    #used_res = [0.1,0.1,0.1]    
    #cn_scripts.conv_defocus_batch(used_res)
    #cn_scripts.conv_gaussian_simple_batch(used_res)
    #cn_scripts.conv_gaussian_batch(used_res)    
    
    used_res = [0.2,0.2,0.2]
    cn_scripts.conv_defocus_batch(used_res)
    cn_scripts.conv_gaussian_simple_batch(used_res)
    cn_scripts.conv_gaussian_batch(used_res)
    
    used_res = [0.4,0.4,0.4]
    cn_scripts.conv_defocus_batch(used_res)
    cn_scripts.conv_gaussian_simple_batch(used_res)
    cn_scripts.conv_gaussian_batch(used_res)

    
    cn_scripts.downsample_convoluted_stack()
    cn_scripts.crop_convoluted_stacks()
    cn_scripts.crop_convoluted_stacks_overlap()    
    
    
#----------------------------------------------------------------------
def main_gen_psf_recon():
    
    used_res = [0.4,0.4,0.4]
    cn_scripts.psf_conv_gaussian_simple(used_res)
    cn_scripts.psf_conv_gaussian(used_res)
    """"""
    
    
if __name__ == '__main__':
    #main_generation()
    main_gen_psf_recon()