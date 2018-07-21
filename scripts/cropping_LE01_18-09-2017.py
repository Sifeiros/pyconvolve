import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

import util
import psf_recon.psf_data_cropping as psf_crop


os.chdir(util.CODE_PATH)

def first_crops_middle_section():
    path_psf_data_LE01 = util.ptjoin(util.PSF_DATA, 'LE01_Sample_4_100nm')
    fname_psf_data_LE01 = 'PSF_Data_100nm_1_LE01_R01_Slice{:0>3}.tif'

    

    out_path = util.ptjoin(util.PSF_DATA_CROP, '02_LE01_Sample_4_100nm')
    sub_path = 'crop_[X_CRD]_[XY]_[CR_IND]'
    fname_out = 'PSF_LE01_crop_[XY]_[CR_IND]_Slice[IMG_IND].tif'
    
    #Middle slice top-down
    x_slice = [slice(1000,1600,1), slice(1600, 2200,1), slice(2200, 2800, 1), slice(2800, 3400, 1), slice(3400, 4000, 1), slice(4000, 4600, 1), slice(4600,5200,1), slice(5200, 5800,1)] 
    #x_slice = [slice(1000,1200,1), slice(1600, 1800,1), slice(2200, 2400, 1), slice(2800, 3000, 1), slice(3400, 3600, 1), slice(4000, 4200, 1), slice(4600,4800,1), slice(5200, 5400,1)]
    y_slice = slice(4700,5300,1)
    #y_slice = slice(4900,5100,1)
    z_slice = slice(None, None, 1)
    
    
    cropper = psf_crop.PSF_cropper(path_psf_data_LE01, fname_psf_data_LE01, out_path, fname_out, 
                                  out_sub_path=sub_path, 
                                  x_slice=x_slice, 
                                  y_slice=y_slice,
                                  z_slice=z_slice)
    
    cropper.EXTRA_SIZE = 0
    cropper.crop()

def aligned_crops_to_256x256():
    path_psf_data_input = util.ptjoin(util.PSF_DATA_CROP, 'LE01_Sample_4_100nm')
    
    sub_path_in = 'crop_{0}_600x600_{1:0>2}' # 0: 1000, 1600,... , 1: 01, 02, 03, ...
    add_path = 'aligned'
    
    f_ident_in = 'PSF_LE01_crop_600x600_{:0>2}_Slice{:0>3}.tif'
    
    out_path_root = util.ptjoin(util.PSF_DATA_CROP, 'LE01_Sample_4_100nm_aligned_256')
    sub_path_out = 'crop_256x256_{0:0>2}_{1}' #0: 01, 02, 03, 1: TR, TL, DR, DL
    
    crop_indexes = list(range(8))
    areas_slices = {'TR':[slice(0,256,1), slice(256,512,1)],'TL':[slice(0,256,1),slice(0,256,1)],'DR':[slice(256,512,1),slice(256,512,1)],'DL':[slice(256,512,1),slice(0,256,1)]}
    
    f_ident_out = 'PSF_LE01_crop_256x256_{:0>2}_Slice{:0>3}.tif'
    
    for cr_ind in crop_indexes:
        
        cur_in_path = util.ptjoin(path_psf_data_input, sub_path_in.format(1000+cr_ind*600, cr_ind+1), add_path)
        cur_f_ident_in = f_ident_in.format(cr_ind+1, '{:0>3}')
        cur_f_ident_out = f_ident_out.format(cr_ind+1, '{2:0>3}')
        
        for area in areas_slices.keys():
            cur_out_path = util.ptjoin(out_path_root, sub_path_out.format(cr_ind+1, area))
            
            t_crop = psf_crop.PSF_cropper(cur_in_path, cur_f_ident_in, cur_out_path, cur_f_ident_out, 
                                x_slice=areas_slices[area][0], 
                                y_slice=areas_slices[area][1])
    
            t_crop.crop()
    


if __name__ == '__main__':
    aligned_crops_to_256x256()