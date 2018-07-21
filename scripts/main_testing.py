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
import sim_creator.psf_generator as psfg

import os
import numpy as np
import math as mt

import matplotlib.pyplot as plt
import matplotlib.animation as anim

from util.timer import Timer


#----------------------------------------------------------------------
#IMPLEMENT AS UTIL FUNCTION!
def readPSFstack(path, f_ident, nmin, nmax):
    
    img_stack = []
    
    for i in range(nmin, nmax+1):
        t_f_path = os.path.join(path, f_ident.format(i))
        with pytiff.Tiff(t_f_path, 'r') as handle:
            t_img = handle[:,:]
        print("Dtype: {} max: {} min: {}".format(t_img.dtype, t_img.max(), t_img.min()))
        img_stack.append(t_img)
        
        
    return np.array(img_stack, dtype='uint16')
    
def writePSFStack(psf_array, path, f_ident):
    
    img_count = psf_array.shape[0]
    t_img = np.zeros((psf_array.shape[1], psf_array.shape[2]), dtype = 'uint16')
    factor = int(65535./psf_array.max())
    
    for i in range(img_count):
        with pytiff.Tiff(os.path.join(path, f_ident.format(i)), 'w') as w_handle:
            t_img[:,:] = factor*psf_array[i,:,:]
            w_handle.write(t_img)
        

#----------------------------------------------------------------------
def timeFourierTransform():
    """"""

    size = 10
    num = 0

    #sizes = [10,20,30,40,50,60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400]
    #sizes = [9, 19, 29, 39, 49, 59, 79, 99, 119, 139, 159, 179, 199, 219, 239, 259, 279, 299, 319, 339, 359, 379, 399]
    #sizes = [8,16,32,64,128,256,512, 7, 15, 31, 63, 127, 255, 511, 9, 17, 33, 65, 129, 257, 513]
    #sizes = [7,8,9, 15,16,17, 31,32,33, 63,64,65, 127,128,129, 255,256,257, 511,512,513]
    sizes = [250,251,252,253,254,255,256,257,258,259,260]
    i = 0
    for size in sizes:

        arr = np.random.rand(size, size, size)
        #print('Performing Fourier-Transform for {}'.format(arr.shape))

        with Timer(name = '{}:{}'.format(i, arr.shape), outputWhenExit=True, acc = 5) as tim:
            f_arr = np.fft.fftn(arr)
        num += f_arr.real.sum()
        i += 1
        #size *= 1.25
        #size = int(size)


#----------------------------------------------------------------------
def optimizeSimulationSize():
    """"""
    arr_shape = [239,239,55]

    psf_shape_gauss = [17,17,17]
    psf_shape_defocus = [17,17,33]

    shape1 = [i[0] + i[1] for i in zip(arr_shape, psf_shape_defocus)]
    shape2 = [i[0] + i[1] for i in zip(arr_shape, psf_shape_gauss)]
    arr1 = np.random.random(shape1)
    arr2 = np.random.random(shape2)

    print('Smaller Array:')

    with Timer(name = 'Real:{}'.format(arr1.shape)) as tim:
        f_arr = np.fft.rfftn(arr1)
        r_arr = np.fft.irfftn(f_arr)

    with Timer(name = 'Cpx:{}'.format(arr1.shape)) as tim:
        f_arr = np.fft.fftn(arr1)
        r_arr = np.fft.ifftn(f_arr)


    with Timer(name = 'Real:{}'.format(arr2.shape)) as tim:
        f_arr = np.fft.rfftn(arr2)
        r_arr = np.fft.irfftn(f_arr)

    with Timer(name = 'Cpx:{}'.format(arr2.shape)) as tim:
        f_arr = np.fft.fftn(arr2)
        r_arr = np.fft.ifftn(f_arr)        


    '''
    #.......................................
    print('Benchmark: ')    

    arr_shape = [257,257,65]

    psf_shape_gauss = [17,17,17]
    psf_shape_defocus = [17,17,33]

    shape1 = [i[0] + i[1] for i in zip(arr_shape, psf_shape_defocus)]
    shape2 = [i[0] + i[1] for i in zip(arr_shape, psf_shape_gauss)]
    arr1 = np.random.random(shape1)
    arr2 = np.random.random(shape2)


    with Timer(name = 'Real:{}'.format(arr1.shape)) as tim:
        f_arr = np.fft.rfftn(arr1)
        r_arr = np.fft.irfftn(f_arr)

    with Timer(name = 'Cpx:{}'.format(arr1.shape)) as tim:
        f_arr = np.fft.fftn(arr1)
        r_arr = np.fft.ifftn(f_arr)


    with Timer(name = 'Real:{}'.format(arr2.shape)) as tim:
        f_arr = np.fft.rfftn(arr2)
        r_arr = np.fft.irfftn(f_arr)

    with Timer(name = 'Cpx:{}'.format(arr2.shape)) as tim:
        f_arr = np.fft.fftn(arr2)
        r_arr = np.fft.ifftn(f_arr) 
    '''


#----------------------------------------------------------------------
def optimizeSimulationSize2():
    """"""

    shape1 = 517
    shape2 = 129

    for i in range(20):
        arr = np.random.random([shape1,shape1, shape2])
        with Timer(name='Real:{}'.format([shape1,shape1,shape2])):
            f_arr = np.fft.rfftn(arr)
            r_arr = np.fft.irfftn(f_arr)
        with Timer(name='Cpx:{}'.format([shape1,shape1,shape2])):
            f_arr = np.fft.rfftn(arr)
            r_arr = np.fft.irfftn(f_arr)
        shape1 -= 2
    


#----------------------------------------------------------------------
def testISTA():
    """"""
    

    #gauss_single = [0.05, 0.15, 0.25, 0.5]
    gauss_single = [0.05, 0.5]
    gauss_multi = [#[0.05, 0., 0.15],
                   #[0.05, 0., 0.2],
                   [0.05, 0., 0.25],
                   #[0.05, 0., 0.4],
                   [0.1, 0., 0.15],
                   #[0.1, 0., 0.2],
                   #[0.1, 0., 0.25],  
                   #[0.1, 0., 0.4],
                   [0.2, 0., 0.15],
                   #[0.2, 0., 0.2],
                   [0.2, 0., 0.25]]
                   #[0.2, 0., 0.4]]

    #defocus = []

    
    defocus = [[0.3, 2000, 400],
               #[0.3, 2000, 200],
               [0.3, 2000, 100],
               #[0.3, 800, 400],
               [0.3, 800, 200],
               #[0.3, 800, 100],
               [0.1, 2000, 400],
               #[0.1, 2000, 200],
               #[0.1, 2000, 100],
               #[0.1, 800, 400],
               [0.1, 800, 200],
               #[0.1, 800, 100],
               [0.05, 2000, 400],
               #[0.05, 2000, 200],
               [0.05, 2000, 100],
               #[0.05, 800, 400],
               #[0.05, 800, 200],
               [0.05, 800, 100]]    
    


    dict_paths = []

    orig_path = util.ptjoin(util.SIM_DATA, 'cyto_sim_full', 'full_res_[0.4,0.4,0.4]')
    orig_fident = 'cyto_sim_{:0>3}.tif'

    for fl in gauss_single:
        conv_path = 'cyto_gauss_simple_sig-{:.2f}'.format(fl)
        conv_f_ident = 'cyto_conv_gauss_simple_{:0>3}.tif'

        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res_[0.2,0.2,0.2]')

        psf_path = 'gauss_simple_sig-{:.2f}'.format(fl)
        psf_f_ident = 'psf_gauss_simple_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')

        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})


    for lis in gauss_multi:
        conv_path = 'cyto_gauss_sig0-{0[0]:.2f}_sig1-{0[1]:.2f}_sig2-{0[2]:.2f}'.format(lis)
        conv_f_ident = 'cyto_conv_gauss_{:0>3}.tif'

        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res[0.2,0.2,0.2]')

        psf_path = 'gauss_sig0-{0[0]:.2f}_sig1-{0[1]:.2f}_sig2-{0[2]:.2f}'.format(lis)
        psf_f_ident = 'psf_gauss_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')

        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})


    for lis in defocus:
        conv_path = 'cyto_defocus_sig-{0[0]:.2f}_zi-{0[1]}_K-{0[2]}'.format(lis)
        conv_f_ident = 'cyto_conv_defocus_{:0>3}.tif'
        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res[0.2,0.2,0.2]')

        psf_path = 'defocussing_sig-{0[0]:.2f}_zi-{0[1]}_K-{0[2]}'.format(lis)
        psf_f_ident = 'psf_defocus_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')

        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})



    orig_stack, orig_meta = util.stack_loader.read_image_stack(orig_path, orig_fident, meta=True)

    #lambdas = [0.8, 0.85, 0.87, 0.9, 0.92, 0.94, 0.96, 0.98, 1., 1.02, 1.04, 1.06, 1.08, 1.1, 1.13, 1.15, 1.2]

    lambdas = [0.87,0.94, 0.98, 1., 1.02, 1.06, 1.13]

    for paths in dict_paths:

        for lam in lambdas:
            conv_path = util.ptjoin(util.SIM_CONVOLUTED, paths['conv_path'], 'full_res_[0.4,0.4,0.4]')
            psf_path = util.ptjoin(util.SIM_PSF, 'odd', paths['psf_path'], 'res_[0.4,0.4,0.4]')
    
            psf_stack, psf_meta = util.stack_loader.read_image_stack(psf_path, paths['psf_fident'], meta=True)
            conv_stack, conv_meta = util.stack_loader.read_image_stack(conv_path, paths['conv_fident'], meta=True)
        
            recon_path = util.ptjoin(util.SIM_RECON, 'parameter_study_ISTA', 'lambda-{}'.format(lam), paths['conv_path'], 'res_[0.4,0.4,0.4]')
            util.createAllPaths(recon_path)
            recon_fident = 'recon_{:0>3}.tif'
            
            ista = wave.ISTA(conv_stack, psf=psf_stack, iterSteps=200, groundTruth=orig_stack,
                            depth= 3, gamma= 1., lamb= lam, errTol= 1e-5, constraints= None, 
                            isPsfCentered= True, useCpxFFT= False, debugInt= 3, compareWithTruth= True, 
                            saveIntermediateSteps= 50)
            ista.initSaveParameters(recon_path, recon_fident)
            
            ista.prepare()
            ista.solve()
            ista.saveSolution()
        

#----------------------------------------------------------------------
def testFISTA():
    """"""
    

    #gauss_single = [0.05, 0.15, 0.25, 0.5]
    gauss_single = [0.05, 0.5]
    gauss_multi = [#[0.05, 0., 0.15],
                   #[0.05, 0., 0.2],
                   [0.05, 0., 0.25],
                   #[0.05, 0., 0.4],
                   [0.1, 0., 0.15],
                   #[0.1, 0., 0.2],
                   #[0.1, 0., 0.25],  
                   #[0.1, 0., 0.4],
                   [0.2, 0., 0.15],
                   #[0.2, 0., 0.2],
                   [0.2, 0., 0.25]]
                   #[0.2, 0., 0.4]]

    defocus = []

    """
    defocus = [[0.3, 2000, 400],
               #[0.3, 2000, 200],
               [0.3, 2000, 100],
               #[0.3, 800, 400],
               [0.3, 800, 200],
               #[0.3, 800, 100],
               [0.1, 2000, 400],
               #[0.1, 2000, 200],
               #[0.1, 2000, 100],
               #[0.1, 800, 400],
               [0.1, 800, 200],
               #[0.1, 800, 100],
               [0.05, 2000, 400],
               #[0.05, 2000, 200],
               [0.05, 2000, 100],
               #[0.05, 800, 400],
               #[0.05, 800, 200],
               [0.05, 800, 100]]    
    """


    dict_paths = []

    orig_path = util.ptjoin(util.SIM_DATA, 'cyto_sim_full', 'full_res_[0.4,0.4,0.4]')
    orig_fident = 'cyto_sim_{:0>3}.tif'

    for fl in gauss_single:
        conv_path = 'cyto_gauss_simple_sig-{:.2f}'.format(fl)
        conv_f_ident = 'cyto_conv_gauss_simple_{:0>3}.tif'

        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res_[0.2,0.2,0.2]')

        psf_path = 'gauss_simple_sig-{:.2f}'.format(fl)
        psf_f_ident = 'psf_gauss_simple_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')

        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})


    for lis in gauss_multi:
        conv_path = 'cyto_gauss_sig0-{0[0]:.2f}_sig1-{0[1]:.2f}_sig2-{0[2]:.2f}'.format(lis)
        conv_f_ident = 'cyto_conv_gauss_{:0>3}.tif'

        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res[0.2,0.2,0.2]')

        psf_path = 'gauss_sig0-{0[0]:.2f}_sig1-{0[1]:.2f}_sig2-{0[2]:.2f}'.format(lis)
        psf_f_ident = 'psf_gauss_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')

        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})


    for lis in defocus:
        conv_path = 'cyto_defocus_sig-{0[0]:.2f}_zi-{0[1]}_K-{0[2]}'.format(lis)
        conv_f_ident = 'cyto_conv_defocus_{:0>3}.tif'
        #conv_path = util.ptjoin(util.SIM_CONVOLUTED, path, 'full_res[0.2,0.2,0.2]')

        psf_path = 'defocussing_sig-{0[0]:.2f}_zi-{0[1]}_K-{0[2]}'.format(lis)
        psf_f_ident = 'psf_defocus_{:0>3}.tif'
        #psf_path = util.ptjoin(util.SIM_PSF, 'odd', psf_path, 'res_[0.2,0.2,0.2]')

        dict_paths.append({'conv_path':conv_path, 'conv_fident':conv_f_ident, 'psf_path':psf_path,'psf_fident':psf_f_ident})



    orig_stack, orig_meta = util.stack_loader.read_image_stack(orig_path, orig_fident, meta=True)

    #lambdas = [0.8, 0.85, 0.87, 0.9, 0.92, 0.94, 0.96, 0.98, 1., 1.02, 1.04, 1.06, 1.08, 1.1, 1.13, 1.15, 1.2]

    lambdas = [0.87,0.94, 0.98, 1., 1.02, 1.06, 1.13]

    for paths in dict_paths:

        for lam in lambdas:
            conv_path = util.ptjoin(util.SIM_CONVOLUTED, paths['conv_path'], 'full_res_[0.4,0.4,0.4]')
            psf_path = util.ptjoin(util.SIM_PSF, 'odd', paths['psf_path'], 'res_[0.4,0.4,0.4]')
    
            psf_stack, psf_meta = util.stack_loader.read_image_stack(psf_path, paths['psf_fident'], meta=True)
            conv_stack, conv_meta = util.stack_loader.read_image_stack(conv_path, paths['conv_fident'], meta=True)
        
            recon_path = util.ptjoin(util.SIM_RECON, 'test_FISTA', paths['conv_path'], 'res_[0.4,0.4,0.4]')
            util.createAllPaths(recon_path)
            recon_fident = 'recon_{:0>3}.tif'
            

            fista = wave.FISTA(conv_stack, psf=psf_stack, groundTruth=orig_stack, iterSteps=200, 
                               saveIntermediateSteps=50, compareWithTruth=True, debugInt=3)
            fista.initSaveParameters(recon_path, recon_fident)
            
            fista.prepare()
            fista.solve()
            fista.saveSolution()
        
#----------------------------------------------------------------------
def testConvolutionIdea():
    """"""
    
    orig_stack = np.ones((256,256, 64))
    orig_stack[128:130,128:130, 30:32 ] = 0.6
    orig_stack = orig_stack * 255
    
    psf = psfg.PSF_Generator([3.2,3.2,3.2], [0.1,0.1,0.1], psf_oddity='odd', psf_params={'sigma':0.2,'zi':1000, 'K':200})
    psf.createPSF(oversampling=3)
    
    #invert orig_stack
    orig_max = orig_stack.max()
    orig_min = orig_stack.min()
    
    orig_stack = (-1)*(orig_stack - orig_min) + (orig_max-orig_min)
    
    print(psf.out.max())
    print(psf.out.min())
    
    #conv = cn.Convoluter(orig_stack, psf.out, conv_method='cpx_fft', debug_int=3)
    #conv.convolute()
    
    #---------------------
    #PSF DEFOCUS QUADRIEREN?
    #---------------------
    
    #fig, ani = util.visu_util.imshow3D_ani(conv.out)
    
    #plt.show()
        
if __name__ == '__main__':
    #testFISTA()
    testConvolutionIdea()