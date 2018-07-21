import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

import util
import util.stack_loader as sl

def stacks_for_fiji():
    
    
    BG_VAL = 244    
    
    psf_raw_path = util.ptjoin(util.PSF_DATA_CROP, 'LE01_Sample_4_100nm_aligned_256')
    psf_out_path = util.ptjoin(util.PSF_DATA_CROP, 'fiji', 'LE01_Sample_4_100nm_aligned_256')
    
    sub_path = 'crop_256x256_{0:0>2}_{1}' # 0: 01,02,   1: DL DR TL TR
    
    crop_indexes = range(1,8)
    area_indexes = ['DL','DR','TL','TR']
    
    f_ident = 'PSF_LE01_crop_256x256_{:0>2}_Slice{:0>3}.tif'
    
    for cr_ind in crop_indexes:
        
        cur_fident = f_ident.format(cr_ind, '{:0>3}')
        
        for area in area_indexes:
            cur_path = util.ptjoin(psf_raw_path, sub_path.format(cr_ind, area))
            cur_out_path = util.ptjoin(psf_out_path, sub_path.format(cr_ind, area))
            cur_img = sl.read_image_stack(cur_path, cur_fident, 0, 160)    
            
            cur_psf = np.ones(cur_img.shape, dtype=cur_img.dtype)*BG_VAL
            #cur_psf = np.zeros(cur_img.shape, dtype= 'float')
            cur_psf[:,:,79] = cur_img[:,:,79]
            
            sl.write_image_stack(cur_psf, cur_out_path, f_ident.format(cr_ind, '{:0>3}'),0)
    


def main():
    
    arr1d = np.arange(101, dtype = 'float') % 30
    
    arr2d = np.zeros((101,81), dtype = 'float')    
    arr2d[20:40, 30:40] = 20.
    arr2d[50:90, 50:65] = 10.
    
    arr3d = np.zeros((100,80,40), dtype = 'float')
    arr3d[10:20, 10:20, :] = 10.
    arr3d[20:40, 30:60, 20:30] = 25.
    arr3d[30:50, 30:550, 30:35] = 30.
    arr3d[50:90, 60:75, 10:25] = 37.
    
    
    f_arr1d = np.fft.fft(arr1d)
    f_arr1d = f_arr1d.conj()
    arr1d_rec = np.fft.ifft(f_arr1d).real
    
    
    f_arr2d = np.fft.fft2(arr2d)
    f_arr2d = f_arr2d.conj()
    arr2d_rec = np.fft.ifft2(f_arr2d).real
    
    
    plt.figure()
    plt.plot(arr1d)
    plt.plot(arr1d_rec)
    plt.plot(arr1d[::-1] - arr1d_rec)
    
    plt.figure()
    plt.subplot(221)
    plt.imshow(arr2d)
    plt.subplot(222)
    plt.imshow(arr2d_rec)
    plt.subplot(223)
    plt.imshow(arr2d[::-1,::-1]-arr2d_rec)
    
    
    [fig, ani] = util.imshow3D_ani(arr3d)
    
    plt.show()

if __name__ == '__main__':
    #main()
    stacks_for_fiji()