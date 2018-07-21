#!/usr/bin/env python
#coding:utf-8
"""
  Author:  Lukas Küpper - Sifeiros --<>
  Purpose: 
  Created: 27.03.2017
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
    DIR_ROOT = 'C:\\Users\\lukas\\Master FZJ\\SimData\\'
    print('Using Surface Book Path specifications.')
else:
    DIR_ROOT = 'K:\\Masterarbeit FZJ\\SimData'
    print('Using Desktop Path specifications.')

DIR_DATA = os.path.join(DIR_ROOT, 'data\\')
DIR_PSF = os.path.join(DIR_ROOT, 'psf\\')
DIR_CONV_OUT = os.path.join(DIR_ROOT, 'convoluted\\')
DIR_RECON_OUT = os.path.join(DIR_ROOT, 'reconstructed\\')
DIR_RECON_ALGO = {algo: os.path.join(DIR_RECON_OUT, algo) for algo in RECON_ALGOS }
SUB_DIRS = ['noise_free', 'gauss_noise',  'poisson_noise']

def createConvolutedTestImage_2D():
    
    psf_selection = [['psf_16x16_1.0', (16,16), 1.], 
                     ['psf_16x16_1.5', (16,16), 1.5], 
                     ['psf_16x16_2.0', (16,16), 2.], 
                     ['psf_16x16_2.5', (16,16), 2.5], 
                     ['psf_16x16_3.0', (16,16), 3.],
                     ['psf_32x32_1.0', (32,32), 1.],
                     ['psf_32x32_1.5', (32,32), 1.5],
                     ['psf_32x32_2.0', (32,32), 2.],
                     ['psf_32x32_2.5', (32,32), 2.5],
                     ['psf_32x32_3.0', (32,32), 3.],
                     ['psf_64x64_1.0', (64,64), 1.],
                     ['psf_64x64_1.5', (64,64), 1.5],
                     ['psf_64x64_2.0', (64,64), 2.],
                     ['psf_64x64_2.5', (64,64), 2.5],
                     ['psf_64x64_3.0', (64,64), 3.],
                     ['psf_128x128_1.0', (128,128), 1.],
                     ['psf_128x128_1.5', (128,128), 1.5],
                     ['psf_128x128_2.0', (128,128), 2.],
                     ['psf_128x128_2.5', (128,128), 2.5],
                     ['psf_128x128_3.0', (128,128), 3.],
                     ]
    
    stack_name = 'sim_cyto_noRnd_9x9_256x256x32'
    file_identifier = 'cyto_stack_256x256_{:0>3}.tif'
    img_no = 19
    
    img_tiff = TIFF.open(os.path.join(DIR_DATA,stack_name, file_identifier.format(img_no)))
    image = img_tiff.read_image()
    img_tiff.close()
    
    image = image.astype('float')
    
    for psf_def in psf_selection:
        
        psf = psf_g.createPSF(psf_def[1], psf_def[2], typ=psf_g.GAUSS_2D)
        
        psf_tif = TIFF.open(os.path.join(DIR_PSF, '2D', psf_def[0]+'.tif'), 'w')
        t_psf = np.zeros(psf.shape, dtype='uint16')
        t_psf[:,:] = 65535*psf[:,:] 
        psf_tif.write_image(t_psf)
        psf_tif.close()
        
        
        conv = crs.ConvoluteImgPsf(image, psf)
        conv.convolute()
        conv = conv.out
        
        conv_tif = TIFF.open(os.path.join(DIR_CONV_OUT, '2D', stack_name+'_'+psf_def[0]+'.tif'), 'w')
        conv_tif.write_image((65535*conv/conv.max()).astype('uint16'))
        conv_tif.close()
    
def createPSFs_3D():
    ch_interp = psf_g.INTERP_QUADR
    
    meta_dir = { psf_g.GAUSS_2D+ch_interp: 'PSF 2D Gaussian',
                 psf_g.GAUSS_3D_LOCAL+ch_interp: 'PSF 3D Gaussian (localized)',
                 psf_g.GAUSS_3D_SPREAD+psf_g.INTERP_LINEAR: 'PSF 3D Gaussian (fanned out), linear Interpolation',
                 psf_g.GAUSS_3D_SPREAD+psf_g.INTERP_QUADR: 'PSF 3D Gaussian (fanned out), quadratic Interpolation',
                 psf_g.LORENTZ_2D+ch_interp: 'PSF 2D Lorentz',
                 psf_g.LORENTZ_3D_LOCAL+ch_interp: 'PSF 3D Lorentz (localized)',
                 psf_g.LORENTZ_3D_SPREAD+psf_g.INTERP_LINEAR: 'PSF 3D Lorentz (fanned out), linear Interpolation',
                 psf_g.LORENTZ_3D_SPREAD+psf_g.INTERP_QUADR: 'PSF 3D Lorentz (fanned out), quadratic interpolation'
                 }   
    
    #psf_selection = [['psf_16x16x16_3Dspread_NoSup_05-40', 'psf_16x16x16_{:0>3}.tif', (16,16,16), (0.5,0.5), (4.,4.), (0.,0.)],
                     #['psf_32x32x16_3Dspread_NoSup_05-40', 'psf_32x32x16_{:0>3}.tif', (32,32,16), (0.5,0.5), (4.,4.), (0.,0.)],
                     #['psf_64x64x16_3Dspread_NoSup_05-40', 'psf_64x64x16_{:0>3}.tif', (64,64,16), (0.5,0.5), (4.,4.), (0.,0.)],
                     #['psf_128x128x16_3Dspread_NoSup_05-40', 'psf_128x128x16_{:0>3}.tif', (128,128,16), (0.5,0.5), (4.,4.), (0.,0.)],
                     #['psf_32x32x32_3Dspread_NoSup_05-40', 'psf_32x32x32_{:0>3}.tif', (32,32,32), (0.5,0.5), (4.,4.), (0.,0.)],
                     #['psf_64x64x32_3Dspread_NoSup_05-40', 'psf_64x64x32_{:0>3}.tif', (64,64,32), (0.5,0.5), (4.,4.), (0.,0.)],
                     #['psf_128x128x32_3Dspread_NoSup_05-40', 'psf_128x128x32_{:0>3}.tif', (128,128,32), (0.5,0.5), (4.,4.), (0.,0.)],
                     #['psf_256x256x32_3Dspread_NoSup_05-40', 'psf_256x256x32_{:0>3}.tif', (256,256,32), (0.5,0.5), (4.,4.), (0.,0.)],
                     #['psf_512x512x32_3Dspread_NoSup_05-40', 'psf_512x512x32_{:0>3}.tif', (512,512,32), (0.5,0.5), (4.,4.), (0.,0.)],
                     #['psf_64x64x64_3Dspread_NoSup_05-40', 'psf_64x64x64_{:0>3}.tif', (64,64,64), (0.5,0.5), (4.,4.), (0.,0.)],
                     #['psf_128x128x64_3Dspread_NoSup_05-40', 'psf_128x128x64_{:0>3}.tif', (128,128,64), (0.5,0.5), (4.,4.), (0.,0.)],
                     #['psf_256x256x64_3Dspread_NoSup_05-40', 'psf_256x256x64_{:0>3}.tif', (256,256,64), (0.5,0.5), (4.,4.), (0.,0.)],
                     #['psf_512x512x64_3Dspread_NoSup_05-40', 'psf_512x512x64_{:0>3}.tif', (512,512,64), (0.5,0.5), (4.,4.), (0.,0.)],
                     #]
    
    psf_selection = [['psf_gauss_64x64x32_3Dlocal', 'psf_64x64x32_{:0>3}.tif', (64,64,32), .5, None, None, None],
                     ['psf_gauss_64x64x32_3Dlocal', 'psf_64x64x32_{:0>3}.tif', (64,64,32), (.5,.5,2.), None, None, None],
                     ['psf_gauss_64x64x32_3Dspread_NoSup_05-40', 'psf_64x64x32_{:0>3}.tif', (64,64,32), None, (.5,.5,), (4.,4.), (0.,0.)],
                     ['psf_lorentz_64x64x32_3Dlocal', 'psf_64x64x32_{:0>3}.tif', (64,64,32), 1., None, None, None],
                     ['psf_lorentz_64x64x32_3Dspread_NoSup_05-40', 'psf_64x64x32_{:0>3}.tif', (64,64,32), (.5, 4.), None, None, (0.,0.)],
                     ]
    
    
                     
    #psf_selection = [['psf_128x128x16_3Dspread_NoSup_05-30', 'psf_128x128x16_{:0>3}.tif', (128,128,16), (0.5,0.5), (3.,3.), (0.,0.)],
                     #['psf_128x128x16_3Dspread_NoSup_05-40', 'psf_128x128x16_{:0>3}.tif', (128,128,16), (0.5,0.5), (4.,4.), (0.,0.)],
                     #['psf_128x128x16_3Dspread_NoSup_05-50', 'psf_128x128x16_{:0>3}.tif', (128,128,16), (0.5,0.5), (5.,5.), (0.,0.)],
                     #['psf_128x128x16_3Dspread_NoSup_10-40', 'psf_128x128x16_{:0>3}.tif', (128,128,16), (1.,1.), (4.,4.), (0.,0.)],
                     #['psf_128x128x16_3Dspread_NoSup_10-50', 'psf_128x128x16_{:0>3}.tif', (128,128,16), (1.,1.), (5.,5.), (0.,0.)],
                     #['psf_128x128x16_3Dspread_Sup_00_-30_05-50', 'psf_128x128x16_{:0>3}.tif', (128,128,16), (0.5,0.5), (5.,5.), (0.,3.)],
                     #['psf_128x128x16_3Dspread_Sup_00_-40_05-50', 'psf_128x128x16_{:0>3}.tif', (128,128,16), (0.5,0.5), (5.,5.), (0.,4.)],
                     #['psf_128x128x16_3Dspread_Sup_00_-50_05-50', 'psf_128x128x16_{:0>3}.tif', (128,128,16), (0.5,0.5), (5.,5.), (0.,5.)],
                     #['psf_128x128x16_3Dspread_Sup_00_-30_10-50', 'psf_128x128x16_{:0>3}.tif', (128,128,16), (1.,1.), (5.,5.), (0.,3.)],
                     #['psf_128x128x16_3Dspread_Sup_00_-40_10-50', 'psf_128x128x16_{:0>3}.tif', (128,128,16), (1.,1.), (5.,5.), (0.,4.)],
                     #['psf_128x128x16_3Dspread_Sup_00_-50_10-50', 'psf_128x128x16_{:0>3}.tif', (128,128,16), (1.,1.), (5.,5.), (0.,5.)],
                     #]    
    
    
    
    
    for psf_def in psf_selection:
    
        if len(psf_def[2]) == 2:
            if 'lorentz' in psf_def[0]:
                psf_typ = psf_g.LORENTZ_2D
            else:
                psf_typ = psf_g.GAUSS_2D
        elif len(psf_def[2]) == 3:
            if 'lorentz' in psf_def[0]:
                if 'local' in psf_def[0]:
                    psf_typ = psf_g.LORENTZ_3D_LOCAL
                else:
                    psf_typ = psf_g.LORENTZ_3D_SPREAD
            else:
                if 'local' in psf_def[0]:
                    psf_typ = psf_g.GAUSS_3D_LOCAL
                else:
                    psf_typ = psf_g.GAUSS_3D_SPREAD
            
        
                
                
    
        psf = psf_g.createPSF(psf_def[2], 
                              sigma = psf_def[3], 
                              typ=psf_typ, 
                              interp=ch_interp, 
                              min_sig= psf_def[4], 
                              max_sig= psf_def[5], 
                              supp= psf_def[6])
        
        
        meta = ['Type: {}'.format(meta_dir[psf_typ+ch_interp]),
                'Parameters:']
        
        if psf_def[3] is not None:
            meta.append('Standard Deviation: {}'.format(psf_def[3]))
        if psf_def[4] is not None:
            meta.append('Minimum St. Deviation: x={0[0]}, y={0[1]}'.format(psf_def[4]))
        if psf_def[5] is not None:
            meta.append('Maximum St. Deviation: x={0[0]}, y={0[1]}'.format(psf_def[5]))
        if psf_def[6] is not None:
            meta.append('Additional Suppression: min={0[0]}, max{0[1]}'.format(psf_def[6]))
        
        psf = tiff_s.TiffStack(psf, path = os.path.join(DIR_PSF, psf_def[0]), fidentifier = psf_def[1])
        psf.meta = meta
        psf.saveTiffStack(overwrite=True)
    
    


def convoluteIMG_PSF_3D():
    
    stack_name = 'sim_cyto_noRnd_9x9_256x256x32_inv'
    file_identifier = 'cyto_stack_256x256_{:0>3}.tif'
    
    sim_data = tiff_l.readTiffStack(os.path.join(DIR_DATA,stack_name), file_identifier, 0, 32)
    
    psf_paths = ['param_analysis', 'size_analysis', 'type_analysis']
    
    
    psf_selection_param = [['psf_128x128x16_3Dspread_NoSup_05-30', 'psf_128x128x16_{:0>3}.tif', (128,128,16), (0.5,0.5), (3.,3.), (0.,0.)],
                           ['psf_128x128x16_3Dspread_NoSup_05-40', 'psf_128x128x16_{:0>3}.tif', (128,128,16), (0.5,0.5), (4.,4.), (0.,0.)],
                           ['psf_128x128x16_3Dspread_NoSup_05-50', 'psf_128x128x16_{:0>3}.tif', (128,128,16), (0.5,0.5), (5.,5.), (0.,0.)],
                           ['psf_128x128x16_3Dspread_NoSup_10-40', 'psf_128x128x16_{:0>3}.tif', (128,128,16), (1.,1.), (4.,4.), (0.,0.)],
                           ['psf_128x128x16_3Dspread_NoSup_10-50', 'psf_128x128x16_{:0>3}.tif', (128,128,16), (1.,1.), (5.,5.), (0.,0.)],
                           ['psf_128x128x16_3Dspread_Sup_00_-30_05-50', 'psf_128x128x16_{:0>3}.tif', (128,128,16), (0.5,0.5), (5.,5.), (0.,3.)],
                           ['psf_128x128x16_3Dspread_Sup_00_-40_05-50', 'psf_128x128x16_{:0>3}.tif', (128,128,16), (0.5,0.5), (5.,5.), (0.,4.)],
                           ['psf_128x128x16_3Dspread_Sup_00_-50_05-50', 'psf_128x128x16_{:0>3}.tif', (128,128,16), (0.5,0.5), (5.,5.), (0.,5.)],
                           ['psf_128x128x16_3Dspread_Sup_00_-30_10-50', 'psf_128x128x16_{:0>3}.tif', (128,128,16), (1.,1.), (5.,5.), (0.,3.)],
                           ['psf_128x128x16_3Dspread_Sup_00_-40_10-50', 'psf_128x128x16_{:0>3}.tif', (128,128,16), (1.,1.), (5.,5.), (0.,4.)],
                           ['psf_128x128x16_3Dspread_Sup_00_-50_10-50', 'psf_128x128x16_{:0>3}.tif', (128,128,16), (1.,1.), (5.,5.), (0.,5.)],
                           ] 
    
    psf_selection_size = [['psf_16x16x16_3Dspread_NoSup_05-40', 'psf_16x16x16_{:0>3}.tif', (16,16,16), (0.5,0.5), (4.,4.), (0.,0.)],
                          ['psf_32x32x16_3Dspread_NoSup_05-40', 'psf_32x32x16_{:0>3}.tif', (32,32,16), (0.5,0.5), (4.,4.), (0.,0.)],
                          ['psf_64x64x16_3Dspread_NoSup_05-40', 'psf_64x64x16_{:0>3}.tif', (64,64,16), (0.5,0.5), (4.,4.), (0.,0.)],
                          ['psf_128x128x16_3Dspread_NoSup_05-40', 'psf_128x128x16_{:0>3}.tif', (128,128,16), (0.5,0.5), (4.,4.), (0.,0.)],
                          ['psf_32x32x32_3Dspread_NoSup_05-40', 'psf_32x32x32_{:0>3}.tif', (32,32,32), (0.5,0.5), (4.,4.), (0.,0.)],
                          ['psf_64x64x32_3Dspread_NoSup_05-40', 'psf_64x64x32_{:0>3}.tif', (64,64,32), (0.5,0.5), (4.,4.), (0.,0.)],
                          ['psf_128x128x32_3Dspread_NoSup_05-40', 'psf_128x128x32_{:0>3}.tif', (128,128,32), (0.5,0.5), (4.,4.), (0.,0.)],
                          ['psf_256x256x32_3Dspread_NoSup_05-40', 'psf_256x256x32_{:0>3}.tif', (256,256,32), (0.5,0.5), (4.,4.), (0.,0.)],
                         # ['psf_512x512x32_3Dspread_NoSup_05-40', 'psf_512x512x32_{:0>3}.tif', (512,512,32), (0.5,0.5), (4.,4.), (0.,0.)],
                         # ['psf_64x64x64_3Dspread_NoSup_05-40', 'psf_64x64x64_{:0>3}.tif', (64,64,64), (0.5,0.5), (4.,4.), (0.,0.)],
                         # ['psf_128x128x64_3Dspread_NoSup_05-40', 'psf_128x128x64_{:0>3}.tif', (128,128,64), (0.5,0.5), (4.,4.), (0.,0.)],
                         # ['psf_256x256x64_3Dspread_NoSup_05-40', 'psf_256x256x64_{:0>3}.tif', (256,256,64), (0.5,0.5), (4.,4.), (0.,0.)],
                         # ['psf_512x512x64_3Dspread_NoSup_05-40', 'psf_512x512x64_{:0>3}.tif', (512,512,64), (0.5,0.5), (4.,4.), (0.,0.)],
                          ]
    
    psf_selection_type = [['psf_gauss_64x64x32_3Dlocal', 'psf_64x64x32_{:0>3}.tif', (64,64,32), .5, None, None, None],
                          ['psf_gauss_64x64x32_3Dlocal', 'psf_64x64x32_{:0>3}.tif', (64,64,32), (.5,.5,2.), None, None, None],
                          ['psf_gauss_64x64x32_3Dspread_NoSup_05-40', 'psf_64x64x32_{:0>3}.tif', (64,64,32), None, (.5,.5,), (4.,4.), (0.,0.)],
                          ['psf_lorentz_64x64x32_3Dlocal', 'psf_64x64x32_{:0>3}.tif', (64,64,32), 1., None, None, None],
                          ['psf_lorentz_64x64x32_3Dspread_NoSup_05-40', 'psf_64x64x32_{:0>3}.tif', (64,64,32), (.5, 4.), None, None, (0.,0.)],
                          ]
    
    SUB_DIRS = ['noise_free', 'gauss_noise',  'poisson_noise', 'lorentz_noise']
    
    #for psf_def in psf_selection_param:
        
        #t_psf = tiff_l.readTiffStack(os.path.join(DIR_PSF,psf_paths[0],psf_def[0]), psf_def[1], 0, psf_def[2][2])
        
        
        #conv_meta = ['Type: Convolution', 'Parameters:', 'Used Method: Fourier Convolution']
        #t_conv = crs.ConvoluteImgPsf(sim_data, t_psf)
        #t_conv.convolute()
        #temp_arr = t_conv.out.copy()
        
        ##Save Noise Free Image
        #t_stack = tiff_s.TiffStack(t_conv.out, 
                                   #path=os.path.join(DIR_CONV_OUT,stack_name, 'param_analysis', SUB_DIRS[0], psf_def[0]), 
                                   #fidentifier = 'conv_{0[0]}x{0[1]}x{0[2]}_{1}.tif'.format(t_conv.out.shape, '{:0>3}'), 
                                   #meta = conv_meta+['Added Noise: No Noise', 'Properties of Convoluted Images:']+sim_data.meta+['---']+t_psf.meta)
        #t_stack.saveTiffStack(overwrite=True)
        ##Add Gauss Noise
        #t_conv.addNoise(mean=8., std=4., typ='gauss')
        ##Save Gauss Noise
        #t_stack = tiff_s.TiffStack(t_conv.out, 
                                   #path=os.path.join(DIR_CONV_OUT,stack_name, 'param_analysis', SUB_DIRS[1], psf_def[0]), 
                                   #fidentifier = 'conv_{0[0]}x{0[1]}x{0[2]}_{1}.tif'.format(t_conv.out.shape, '{:0>3}'), 
                                   #meta = conv_meta+['Added Noise: Gauss Noise, STD=4., MEAN=0.', 'Properties of Convoluted Images:']+sim_data.meta+['---']+t_psf.meta)
        #t_stack.saveTiffStack(overwrite=True)        
        ##Revert to Noise Free
        #t_conv.out = temp_arr
        ##Add Poisson Noise
        #t_conv.addNoise(mean=8., std=4., typ='poisson')        
        ##Save Poisson Noise
        #t_stack = tiff_s.TiffStack(t_conv.out, 
                                   #path=os.path.join(DIR_CONV_OUT,stack_name, 'param_analysis', SUB_DIRS[2], psf_def[0]), 
                                   #fidentifier = 'conv_{0[0]}x{0[1]}x{0[2]}_{1}.tif'.format(t_conv.out.shape, '{:0>3}'), 
                                   #meta = conv_meta+['Added Noise: Poisson Noise, STD=4., MEAN=0.', 'Properties of Convoluted Images:']+sim_data.meta+['---']+t_psf.meta)
        #t_stack.saveTiffStack(overwrite=True)                
    
    
    
    #for psf_def in psf_selection_size:
    
        #t_psf = tiff_l.readTiffStack(os.path.join(DIR_PSF,psf_paths[1],psf_def[0]), psf_def[1], 0, psf_def[2][2])
        
        
        #conv_meta = ['Type: Convolution', 'Parameters:', 'Used Method: Fourier Convolution']
        #t_conv = crs.ConvoluteImgPsf(sim_data, t_psf)
        #t_conv.convolute()
        #temp_arr = t_conv.out.copy()
        
        ##Save Noise Free Image
        #t_stack = tiff_s.TiffStack(t_conv.out, 
                                   #path=os.path.join(DIR_CONV_OUT,stack_name, 'size_analysis', SUB_DIRS[0], psf_def[0]), 
                                   #fidentifier = 'conv_{0[0]}x{0[1]}x{0[2]}_{1}.tif'.format(t_conv.out.shape, '{:0>3}'), 
                                   #meta = conv_meta+['Added Noise: No Noise', 'Properties of Convoluted Images:']+sim_data.meta+['---']+t_psf.meta)
        #t_stack.saveTiffStack(overwrite=True)
        ##Add Gauss Noise
        #t_conv.addNoise(mean=0., std=5., typ='gauss')
        ##Save Gauss Noise
        #t_stack = tiff_s.TiffStack(t_conv.out, 
                                   #path=os.path.join(DIR_CONV_OUT,stack_name, 'size_analysis', SUB_DIRS[1], psf_def[0]), 
                                   #fidentifier = 'conv_{0[0]}x{0[1]}x{0[2]}_{1}.tif'.format(t_conv.out.shape, '{:0>3}'), 
                                   #meta = conv_meta+['Added Noise: Gauss Noise, STD=4., MEAN=0.', 'Properties of Convoluted Images:']+sim_data.meta+['---']+t_psf.meta)
        #t_stack.saveTiffStack(overwrite=True)        
        ##Revert to Noise Free
        #t_conv.out = temp_arr
        ##Add Poisson Noise
        #t_conv.addNoise(mean=0., std=5., typ='poisson')        
        ##Save Poisson Noise
        #t_stack = tiff_s.TiffStack(t_conv.out, 
                                   #path=os.path.join(DIR_CONV_OUT,stack_name, 'size_analysis', SUB_DIRS[2], psf_def[0]), 
                                   #fidentifier = 'conv_{0[0]}x{0[1]}x{0[2]}_{1}.tif'.format(t_conv.out.shape, '{:0>3}'), 
                                   #meta = conv_meta+['Added Noise: Poisson Noise, STD=4., MEAN=0.', 'Properties of Convoluted Images:']+sim_data.meta+['---']+t_psf.meta)
        #t_stack.saveTiffStack(overwrite=True)     
        
    for psf_def in psf_selection_type:
        
        
        gauss_noise = [.1,.25, 1.,3., 6., 10.]
        poisson_noise = [.1,.25, 1., 3., 6., 10.]
        lorentz_noise = [.1,.25, 1., 3., 6., 10.]
        
        t_psf = tiff_l.readTiffStack(os.path.join(DIR_PSF, psf_paths[2], psf_def[0]), psf_def[1], 0, psf_def[2][2])
        
        conv_meta = ['Type: Image Convoluted with PSF', 'Used Method: Fourier Convolution']
        t_conv = crs.ConvoluteImgPsf(sim_data, t_psf)
        t_conv.convolute()
        temp_arr = t_conv.out.copy()


        if not os.path.isdir(os.path.join(DIR_CONV_OUT, stack_name, 'type_analysis',SUB_DIRS[0])):
                os.mkdir(os.path.join(DIR_CONV_OUT, stack_name, 'type_analysis',SUB_DIRS[0]))


        t_stack = tiff_s.TiffStack(t_conv.out,
                                   path = os.path.join(DIR_CONV_OUT, stack_name, 'type_analysis',  SUB_DIRS[0], psf_def[0]),
                                   fidentifier = 'conv_{0[0]}x{0[1]}x{0[2]}_{1}.tif'.format(t_conv.out.shape, '{:0>3}'), 
                                   meta = conv_meta+['Added Noise: No Noise', 'Properties of Convoluted Images:']+sim_data.meta+['---']+t_psf.meta
                                   )
        t_stack.saveTiffStack(overwrite=True)
        
        for noi in gauss_noise:
            t_conv.out = temp_arr.copy()
            t_conv.addNoise(std=noi)
            
            if not os.path.isdir(os.path.join(DIR_CONV_OUT, stack_name, 'type_analysis',SUB_DIRS[1]+'_{}'.format(noi))):
                os.mkdir(os.path.join(DIR_CONV_OUT, stack_name, 'type_analysis',SUB_DIRS[1]+'_{}'.format(noi)))
            
            t_stack = tiff_s.TiffStack(t_conv.out, 
                                       path = os.path.join(DIR_CONV_OUT, stack_name, 'type_analysis', SUB_DIRS[1]+'_{}'.format(noi), psf_def[0]),
                                       fidentifier = 'conv_{0[0]}x{0[1]}x{0[2]}_{1}.tif'.format(t_conv.out.shape, '{:0>3}'), 
                                       meta = conv_meta+['Added Noise: Gaussian, STD={}'.format(noi), 'Properties of Convoluted Images:']+sim_data.meta+['---']+t_psf.meta
                                       )
            t_stack.saveTiffStack(overwrite=True)
            
        
        
        for noi in poisson_noise:
            t_conv.out = temp_arr.copy()
            t_conv.addNoise(std=noi, typ='poisson')
            
            if not os.path.isdir(os.path.join(DIR_CONV_OUT, stack_name, 'type_analysis',SUB_DIRS[2]+'_{}'.format(noi))):
                os.mkdir(os.path.join(DIR_CONV_OUT, stack_name, 'type_analysis',SUB_DIRS[2]+'_{}'.format(noi)))            
            
            t_stack = tiff_s.TiffStack(t_conv.out, 
                                       path = os.path.join(DIR_CONV_OUT, stack_name, 'type_analysis', SUB_DIRS[2]+'_{}'.format(noi), psf_def[0]),
                                       fidentifier = 'conv_{0[0]}x{0[1]}x{0[2]}_{1}.tif'.format(t_conv.out.shape, '{:0>3}'), 
                                       meta = conv_meta+['Added Noise: Poisson, STD={}'.format(noi), 'Properties of Convoluted Images:']+sim_data.meta+['---']+t_psf.meta
                                       )
            t_stack.saveTiffStack(overwrite=True)
        
        
        for noi in lorentz_noise:
            t_conv.out = temp_arr.copy()
            t_conv.addNoise(std=noi, typ='lorentz', lorentz_max=5*noi)
            
            if not os.path.isdir(os.path.join(DIR_CONV_OUT, stack_name, 'type_analysis',SUB_DIRS[3]+'_{}'.format(noi))):
                os.mkdir(os.path.join(DIR_CONV_OUT, stack_name, 'type_analysis',SUB_DIRS[3]+'_{}'.format(noi)))                   
            
            
            t_stack = tiff_s.TiffStack(t_conv.out, 
                                       path = os.path.join(DIR_CONV_OUT, stack_name, 'type_analysis', SUB_DIRS[3]+'_{}'.format(noi), psf_def[0]),
                                       fidentifier = 'conv_{0[0]}x{0[1]}x{0[2]}_{1}.tif'.format(t_conv.out.shape, '{:0>3}'), 
                                       meta = conv_meta+['Added Noise: Lorentz, STD={}'.format(noi), 'Properties of Convoluted Images:']+sim_data.meta+['---']+t_psf.meta
                                       )
            t_stack.saveTiffStack(overwrite=True)
            
        
        
        
        
        
        


def Deconvolution_Analysis_2D(decon_algo = 'test'):
    
    #Load Desired part of TIFF image stack

    #Load or create PSF
    pass
    


def Deconvolution_Analysis_3D(decon_algo = 'test'):
    """
    Deconvolution Algorithms to test:
    - Inverse Filtering
    - Wiener Inverse Filtering
    - Gold Algo
    - Jannson Van Cittert
    - Richardson Lucy

    - Tikhonov?!
    
    - Blind Decon
    
    
    Testing:
    - Size Analysis
    - Noise Analysis
       - Noise Amplitude
       - Noise Type
    - PSF Analysis
       - PSF Type (Gauss, Lorentz; Spread, localized)
       - PSF Size
    """
    
    stack_name = 'sim_cyto_noRnd_9x9_256x256x32_inv'
    file_identifier_orig = 'cyto_stack_256x256_{:0>3}.tif'
    file_identifier = 'conv_256x256x32_{:0>3}.tif'
    
    sim_orig_data = tiff_l.readTiffStack(os.path.join(DIR_DATA,stack_name), file_identifier_orig, 0, 32)
    
    psf_paths = ['type_analysis']    
    
    
    psf_selection_type = [['psf_gauss_64x64x32_3Dlocal', 'psf_64x64x32_{:0>3}.tif', (64,64,32), .5, None, None, None],
                          #['psf_gauss_64x64x32_3Dlocal', 'psf_64x64x32_{:0>3}.tif', (64,64,32), (.5,.5,2.), None, None, None],
                          ['psf_gauss_64x64x32_3Dspread_NoSup_05-40', 'psf_64x64x32_{:0>3}.tif', (64,64,32), None, (.5,.5,), (4.,4.), (0.,0.)]#,
                          #['psf_lorentz_64x64x32_3Dlocal', 'psf_64x64x32_{:0>3}.tif', (64,64,32), 1., None, None, None],
                          #['psf_lorentz_64x64x32_3Dspread_NoSup_05-40', 'psf_64x64x32_{:0>3}.tif', (64,64,32), (.5, 4.), None, None, (0.,0.)]
                          ]
    
    sub_dirs = ['noise_free', 'gauss_noise',  'poisson_noise', 'lorentz_noise']    
    
    used_decon_algos = {'inverse':False, 'gold':False,'wiener':False,'richardson':False, 'jannson':False, 'blind_richardson':True}

    cur_recon = 1
    skip_recon = 13
    
    noise_dict = {'gauss':[.25,1.,3.],
                  'poisson':[.25,1.,3.],
                  'lorentz':[.25,1.,3.]
                  }
    
    tmp_path = os.path.join(DIR_CONV_OUT, stack_name, psf_paths[0])
    
    out_path = os.path.join(DIR_RECON_OUT, stack_name)
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    out_path = os.path.join(out_path, psf_paths[0])
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    
    for noise_typ in sub_dirs:
        
        if noise_typ is 'noise_free':
            dirs = ['noise_free']
        elif noise_typ is 'gauss_noise':
            dirs = ['gauss_noise_{}'.format(noi) for noi in noise_dict['gauss']]
        elif noise_typ is 'poisson_noise':
            dirs = ['lorentz_noise_{}'.format(noi) for noi in noise_dict['lorentz']]
        elif noise_typ is 'lorentz_noise':
            dirs = ['poisson_noise_{}'.format(noi) for noi in noise_dict['poisson']]
        else:
            raise ValueError

        for di in dirs:
            
            if not os.path.isdir(os.path.join(out_path, di)):
                os.mkdir(os.path.join(out_path, di))
            
            for psf_def in psf_selection_type:
                
                if cur_recon <= skip_recon:
                    print('Skipping Reconstruction No.{}.'.format(cur_recon))
                    cur_recon += 1
                    continue
                
                print('----------------------------------------------')
                print('Doing Reconstruction No.{}'.format(cur_recon))
                print('----------------------------------------------')
                tmp_dir = os.path.join(tmp_path, di, psf_def[0])
                
                if not os.path.isdir(os.path.join(out_path, di, psf_def[0])):
                    os.mkdir(os.path.join(out_path, di, psf_def[0]))
                
                conv_stack = tiff_l.readTiffStack(tmp_dir, file_identifier, 0, 32)
                psf_stack = tiff_l.readTiffStack(os.path.join(DIR_PSF, psf_paths[0], psf_def[0]), psf_def[1], 0, psf_def[2][2])
                
                if used_decon_algos['inverse']:
                    dec_temp = da.InverseFilter(conv_stack, 
                                     psf_stack, 
                                     psf_cutoff= 0.001, 
                                     isPSFCutoffRelative= True, 
                                     useCpxFFT=False )
                    dec_temp.debug = True
                    dec_temp.solve()
                    
                    t_stack = tiff_s.TiffStack(dec_temp.out, 
                                               os.path.join(out_path, di, psf_def[0], 'inverse'),
                                               'recon_'+file_identifier)
                    
                    t_meta = ['Type: Reconstruction', 'Reconstruction Algorithm: Inverse Filter','---', 'Properties of Convoluted Stack:']
                    t_meta += conv_stack.meta
                    
                    t_stack.meta = t_meta
                    t_stack.saveTiffStack(overwrite=True)
                    
                    
                
                if used_decon_algos['gold']:
                    dec_temp = da.Gold(conv_stack, 
                                       psf_stack, 
                                       useCpxFFT=False, 
                                       iniGuessAlgo= da.Gold.GLD_ORIGARRAY, 
                                       correctZeros= True, 
                                       maxIter= 1e3, 
                                       errTol= 1.,
                                       p=1 )
                    dec_temp.debug = True
                    dec_temp.solve()
                
                    t_stack = tiff_s.TiffStack(dec_temp.out, 
                                               os.path.join(out_path, di, psf_def[0], 'gold'),
                                               'recon_'+file_identifier)
                    
                    t_meta = ['Type: Reconstruction', 
                              'Reconstruction Algorithm: Gold Algorithm', 
                              'Iterations: {}'.format(dec_temp.curIter),
                              'Error: {}'.format(dec_temp.err),
                              '---', 
                              'Properties of Convoluted Stack:']
                    t_meta += conv_stack.meta
                    
                    t_stack.meta = t_meta
                    t_stack.saveTiffStack(overwrite=True)
                    
                
                if used_decon_algos['wiener']:
                    dec_temp = da.WienerFilter(conv_stack, 
                                               psf_stack, 
                                               useCpxFFT= False, 
                                               cutoff_noise= 1000., 
                                               noiseRelative=False)
                    dec_temp.debug = True
                    dec_temp.solve()
                    
                    t_stack = tiff_s.TiffStack(dec_temp.out, 
                                               os.path.join(out_path, di, psf_def[0], 'wiener'),
                                               'recon_'+file_identifier)
                    
                    t_meta = ['Type: Reconstruction', 
                              'Reconstruction Algorithm: Wiener Inverse Filter', 
                              '---', 
                              'Properties of Convoluted Stack:']
                    t_meta += conv_stack.meta
                    
                    t_stack.meta = t_meta
                    t_stack.saveTiffStack(overwrite=True)
                    
                
                if used_decon_algos['richardson']:
                    
                    dec_temp = da.RichardsonLucy(conv_stack, 
                                                 psf_stack, 
                                                 useCpxFFT=False, 
                                                 iniGuessAlgo=da.RichardsonLucy.RL_ORIGARRAY, 
                                                 correctZeros=True, 
                                                 maxIter=2e3, 
                                                 errTol=1., 
                                                 p=1.)
                    dec_temp.debug = True
                    dec_temp.solve()
                    
                    t_stack = tiff_s.TiffStack(dec_temp.out, 
                                               os.path.join(out_path, di, psf_def[0], 'richardson'),
                                               'recon_'+file_identifier)
                    
                    t_meta = ['Type: Reconstruction', 
                              'Reconstruction Algorithm: Richardson Lucy Algorithm', 
                              'Iterations: {}'.format(dec_temp.curIter),
                              'Error: {}'.format(dec_temp.err),
                              '---', 
                              'Properties of Convoluted Stack:']
                    t_meta += conv_stack.meta
                    
                    t_stack.meta = t_meta
                    t_stack.saveTiffStack(overwrite=True)
                    
                    
                
                if used_decon_algos['jannson']:
                    
                    dec_temp = da.JannsonVCittert(conv_stack,
                                                  psf_stack,
                                                  useCpxFFT=False, 
                                                  iniGuessAlgo=da.JannsonVCittert.JVC_ORIGARRAY, 
                                                  correctZeros=True, 
                                                  maxIter=2e3, 
                                                  errTol=1., 
                                                  p=1)
                    
                    dec_temp.debug = True
                    dec_temp.solve()
                    
                    t_stack = tiff_s.TiffStack(dec_temp.out, 
                                               os.path.join(out_path, di, psf_def[0], 'jannson'),
                                               'recon_'+file_identifier)
                    
                    t_meta = ['Type: Reconstruction', 
                              'Reconstruction Algorithm: Jannson Van Cittert Algorithm', 
                              'Iterations: {}'.format(dec_temp.curIter),
                              'Error: {}'.format(dec_temp.err),
                              '---', 
                              'Properties of Convoluted Stack:']
                    t_meta += conv_stack.meta
                    
                    t_stack.meta = t_meta
                    t_stack.saveTiffStack(overwrite=True)
                    
                if used_decon_algos['blind_richardson']:
                    
                    psf_stack = psf_g.createPSF((64,64,32), 
                                                1., 
                                                typ= psf_g.GAUSS_3D_LOCAL,
                                                normalize=True)
                    
                    dec_temp = da.BlindRichardsonLucy(conv_stack, 
                                                      psf_stack, 
                                                      iniGuessAlgo=da.BlindRichardsonLucy.BRL_ORIGARRAY, 
                                                      useCpxFFT=True, 
                                                      pad_psf=True, 
                                                      psf_init_guess=da.BlindDeconvAlgorithm.PSF_GAUSS_LOCAL, 
                                                      psf_shape=None, 
                                                      correctZeros=True, 
                                                      maxIter=1e3, 
                                                      errTol=1e-5, 
                                                      p=1, 
                                                      norm_psf=1.)
                    dec_temp.debug = True
                    dec_temp.solve()
                    
                    t_stack = tiff_s.TiffStack(dec_temp.out, 
                                               os.path.join(out_path, di, psf_def[0], 'blind_richardson'),
                                               'recon_'+file_identifier)                    
                    
                    t_meta = ['Type: Reconstruction', 
                              'Reconstruction Algorithm: Blind Richardson Lucy', 
                              'Iterations: {}'.format(dec_temp.curIter),
                              'Error: {}'.format(dec_temp.err),
                              '---', 
                              'Properties of Convoluted Stack:']
                    t_meta += conv_stack.meta
                    
                    t_stack.meta = t_meta
                    t_stack.saveTiffStack(overwrite=True)                    
                
                cur_recon +=1
                
                

def analyzeReconstruction():
        
    #stack_names = ['sim_cyto_noRnd_9x9_256x256x32', 'sim_cyto_noRnd_9x9_256x256x32_inv']
    stack_name = 'sim_cyto_noRnd_9x9_256x256x32_inv'
    file_identifier_recon = 'recon_conv_256x256x32_{:0>3}.tif'
    file_identifier_orig = 'cyto_stack_256x256_{:0>3}.tif'
    FTYPE = ".png"
    
    
    image_output = os.path.join(DIR_RECON_OUT, 'images')
    
    sim_orig_data = tiff_l.readTiffStack(os.path.join(DIR_DATA,stack_name), file_identifier_orig, 0, 32)
    psf_paths = ['type_analysis']       
    psf_selection_type = [['psf_gauss_64x64x32_3Dlocal', 'psf_64x64x32_{:0>3}.tif', (64,64,32), .5, None, None, None]]


    sub_dirs = ['noise_free', 'gauss_noise',  'poisson_noise', 'lorentz_noise']    
    
    used_decon_algos = {'inverse':False, 'gold':False,'wiener':False,'richardson':False, 'jannson':True}

    
    noise_dict = {'gauss':[.25,1.,3.],
                  'poisson':[.25,1.,3.],
                  'lorentz':[.25,1.,3.]
                  }    


    tmp_path = os.path.join(DIR_RECON_OUT, stack_name, psf_paths[0])

    for noise_typ in sub_dirs:
        
        if noise_typ is 'noise_free':
            dirs = ['noise_free']
        elif noise_typ is 'gauss_noise':
            dirs = ['gauss_noise_{}'.format(noi) for noi in noise_dict['gauss']]
        elif noise_typ is 'poisson_noise':
            dirs = ['lorentz_noise_{}'.format(noi) for noi in noise_dict['lorentz']]
        elif noise_typ is 'lorentz_noise':
            dirs = ['poisson_noise_{}'.format(noi) for noi in noise_dict['poisson']]
        else:
            raise ValueError    
    
        for di in dirs:
            
            for psf in psf_selection_type:
                
                
                algos = [alg for alg in used_decon_algos.keys() if used_decon_algos[alg]]
                
                for alg in algos:
                    cur_dir = os.path.join(tmp_path, di, psf[0], alg)
                    t_stack = tiff_l.readTiffStack(cur_dir, file_identifier_recon, 0, 32)
                    
                    fig = util.imshow3D(t_stack, 
                                        img_area=slice(None,None,1), 
                                        subpl_size=[5, 5], 
                                        plt_dim=2, 
                                        cmap='gray',
                                        cutoff=0.2,
                                        cutoffRelative=True)
                
                    fname = 'recon_' + psf[0] + '_' + alg + '_' + di + FTYPE
                
                    fig.savefig(os.path.join(image_output, fname), bbox_inches='tight')


def main():
    #createConvolutedTestImage_2D()
    #invertStackData()
    #createPSFs_3D()
    #convoluteIMG_PSF_3D()
    Deconvolution_Analysis_3D()
    #analyzeReconstruction()
    pass




if __name__ == '__main__':
    main()