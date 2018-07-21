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
import psf_recon.psf_data_cropping as psf_crop
import decon 
import util.stack_loader as sl

os.chdir(util.CODE_PATH)

BG_VAL = 244

def recon_from256():
    
    psf_raw_path = util.ptjoin(util.PSF_DATA_CROP, 'LE01_Sample_4_100nm_aligned_256')
    psf_out_path = util.ptjoin(util.PSF_RECON, 'LE01_Sample_4_100nm_aligned_256')
    
    sub_path = 'crop_256x256_{0:0>2}_{1}' # 0: 01,02,   1: DL DR TL TR
    
    crop_indexes = range(1,8)
    area_indexes = ['DL','DR','TL','TR']
    
    f_ident = 'PSF_LE01_crop_256x256_{:0>2}_Slice{:0>3}.tif'
    
    for cr_ind in crop_indexes:
        
        cur_fident = f_ident.format(cr_ind, '{:0>3}')
        
        for area in area_indexes:
            cur_path = util.ptjoin(psf_raw_path, sub_path.format(cr_ind, area))
            cur_img = sl.read_image_stack(cur_path, cur_fident, 0, 160)
            #cur_img = cur_img - cur_img.min()
            cur_img = np.array(cur_img, dtype='float')
            
            cur_psf = np.ones(cur_img.shape, dtype=cur_img.dtype)*BG_VAL
            #cur_psf = np.zeros(cur_img.shape, dtype= 'float')
            cur_psf[:,:,79] = cur_img[:,:,79]
            
            print('Recon with Wiener Inverse Filter.')
            psf_rec = decon.WienerFilter(cur_img, cur_psf, useCpxFFT=False, cutoff_noise=0.005, noiseRelative=True)
            psf_rec.solve()
            
            print('Normalizing')
            temp_img = psf_rec.out
            #temp_img = temp_img/temp_img.sum()
            
            psf_rec = decon.RichardsonLucy(cur_img, cur_psf, useCpxFFT=False,correctZeros=False, maxIter=600, errTol=0, p=2, normalize=False)
            psf_rec.curGuess = temp_img
            psf_rec.debug = True
            psf_rec.solve()
            
            sl.write_image_stack(psf_rec.out.real, util.ptjoin(psf_out_path, sub_path.format(cr_ind, area)) , cur_fident, 0)
            
            
def statistic_psf_recon():
    psf_in_path = util.ptjoin(util.PSF_RECON, 'LE01_Sample_4_100nm_aligned_256')
    
    psf_out_path = util.ptjoin(util.PSF_RECON, 'LE01_Sample_4_100nm_aligned_256')
    
    sub_in_path = 'crop_256x256_{0:0>2}_{1}' # 0: 01,02,   1: DL DR TL TR
    sub_out_path = 'crop_256x256_stat'
    
    crop_indexes = range(1,8)
    area_indexes = ['DL','DR','TL','TR']
    
    f_ident = 'PSF_LE01_crop_256x256_{:0>2}_Slice{:0>3}.tif'    
    
    n_stack = 0
    psf_stack = np.zeros((256,256,160), dtype='float')
    
    for cr_ind in crop_indexes:
        
        cur_fident = f_ident.format(cr_ind, '{:0>3}')
        
        for area in area_indexes:    
    
            cur_path = util.ptjoin(psf_in_path, sub_in_path.format(cr_ind, area))
            
            tmp_stack = sl.read_image_stack(cur_path, cur_fident, 0, 160)
            
            if tmp_stack.shape == (256,256,160):
                print('Adding to psf_stack. {} {}'.format(cr_ind, area))
                psf_stack += tmp_stack
                n_stack += 1
                
            
    print('Using {} stacks for statistics...'.format(n_stack))
    psf_stack = psf_stack/n_stack
    psf_stack = psf_stack/psf_stack.sum()
    
    sl.write_image_stack(psf_stack, util.ptjoin(psf_out_path, sub_out_path), f_ident.format('STAT', '{:0>3}'), 0)
    
    
def psf_test_recon():
    
    test_data_path = util.ptjoin(util.MICRO_DATA, 'nucleus_test')
    data_f_ident = 'B21_{:0>4}_z_stacks_LE01_S01_R01_Slice{:0>2}_crop.tif'
    
    recon_image_data = np.zeros((4, 512, 512, 30), dtype='uint8')
    
    
    #psf_path = util.ptjoin(util.PSF_RECON, 'LE01_Sample_4_100nm_aligned_256','crop_256x256_stat')
    #psf_ident = 'PSF_LE01_crop_256x256_STAT_Slice{:0>3}.tif'    
    
    #psf_path = 'O:\\Master FZJ\\PSF_recon\\fiji\\RegInv_TikMiller'
    #psf_ident = 'TMiller_crop_256x256_01_TL{:0>3}.tif'
    
    psf_path = 'C:\\Users\\lukas\\Master FZJ\\PSF_recon\\Sept_2017\\fiji\\Wiener'
    psf_ident = 'Wiener_PSF_recon_{:0>3}.tif'
    
    psf_data = sl.read_image_stack(psf_path, psf_ident, 0, 160)
    
    
    compr = [4,4,4]
    base_shape = [256,256,160]
    compr_shape = [64,64,40]
    psf_data_compr = np.zeros(compr_shape, dtype='float')
    
    print('calculating compressed PSF')
    
    for z in range(compr_shape[2]):
        for x in range(compr_shape[0]):
            for y in range(compr_shape[1]):
                temp = 0.
                
                for indz in range(compr[2]):
                    for indx in range(compr[0]):
                        for indy in range(compr[1]):
                            temp += psf_data[compr[0]*x+indx, compr[1]*y+indy, compr[2]*z+indz]
                psf_data_compr[x,y,z] = temp

    print('{} {}'.format(psf_data_compr.max(), psf_data_compr.min()))
    psf_data_compr = psf_data_compr-psf_data_compr.min()
    psf_data_compr = (psf_data_compr/psf_data_compr.max())*255
    print('{} {}'.format(psf_data_compr.max(), psf_data_compr.min()))
    
    
    psf_data_out = np.zeros(compr_shape, dtype = 'uint8')        
    psf_data_out[:,:,:] = psf_data_compr[:,:,:]
    print('{} {}'.format(psf_data_out.max(), psf_data_out.min()))
    
    sl.write_image_stack(psf_data_out, util.ptjoin(psf_path, 'compressed'), psf_ident, 0)
    
    #psf_data_compr = psf_data_compr[:,:, 5:35]
    #psf_data_compr = psf_data_compr/psf_data_compr.sum()
    
    return util.imshow3D_ani(psf_data_compr)
    
    
    #print('Calculating reconstruction')
    
    #for sample_index in range(4082, 4086):
        
        #tmp_img = sl.read_image_stack(test_data_path, data_f_ident.format(sample_index, '{:0>2}'), 1, 30)
        #tmp_img = tmp_img[0:512, 0:512, :]
        #print(tmp_img.shape)
        
        ##recon = decon.WienerFilter(tmp_img, psf_data_compr, useCpxFFT=False, cutoff_noise=1., noiseRelative=True)
        #recon = decon.RichardsonLucy(tmp_img, psf_data_compr, maxIter=13, errTol=0, normalize=True, correctZeros=True)
        #recon.solve()
        
        #recon_image_data[sample_index-4082,:,:,:] = recon.out[:,:,:]*255
        
        
    #recon_path = util.ptjoin(util.MICRO_RECON, 'nucleus_test')
    #f_ident = 'B21_{:0>2}_z_stacks_LE01_S01_R01_Slice{:0>2}_recon.tif'
    
    #print('Saving reconstruction')
    #for ind in range(4):
        #tmp_stack = recon_image_data[ind,:,:,:]
        #sl.write_image_stack(tmp_stack, recon_path, f_ident.format(ind, '{:0>2}'), 1)
    
        
    
            
def show_psf_animation():
    
    psf_path = util.ptjoin(util.PSF_RECON, 'LE01_Sample_4_100nm_aligned_256')
    
    
    sub_path = 'crop_256x256_{0:0>2}_{1}' # 0: 01,02,   1: DL DR TL TR
    
    crop_indexes = range(1,8)
    area_indexes = ['DL','DR','TL','TR']
    
    f_ident = 'PSF_LE01_crop_256x256_{:0>2}_Slice{:0>3}.tif'
    

    cr_ind = 1
    area = 'DL'
    
    cur_path = util.ptjoin(psf_path, sub_path.format(cr_ind, area))
    cur_fident = f_ident.format(cr_ind, '{:0>3}')
    
    #cur_path = util.ptjoin(psf_path, 'crop_256x256_stat')
    #cur_fident = 'PSF_LE01_crop_256x256_STAT_Slice{:0>3}.tif'
    
    cur_img = sl.read_image_stack(cur_path, cur_fident, 0, 160)    
    
    #cur_img[128,128,:] = cur_img.max()
    
    return util.imshow3D_ani(cur_img, cmap='gray', useArrayWideMax=False)
    
                
            

if __name__ == '__main__':
    #recon_from256()
    #statistic_psf_recon()
    #[fig, ani] = show_psf_animation()
    [fig,ani] = psf_test_recon()
    
    plt.show()
