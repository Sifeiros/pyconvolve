import numpy as np
import math as mt

import util
import util.stack_loader

import sim_creator.psf_generator as pg

import util
import util.visu_util as v_util
import matplotlib.pyplot as plt
import matplotlib.animation as anim



#----------------------------------------------------------------------
def createDefocusPSF():
    """"""
    
    #size = [6.2, 6.2, 6.4]
    size = [1.6, 1.6, 6.4]
    resolution = [0.1, 0.1, 0.1]
    
    
    
    sig = .05
    zi = 2000.
    K = 400.
    psf_defocus_params = {'sigma':sig, 'zi':zi, 'K':K}
    
    f_ident = 'psf_defocus_{:0>3}.tif'
    path = util.ptjoin(util.SIM_PSF, 'defocussing_sig-{:.2f}_zi-{:.0f}_K-{:.0f}'.format(sig, zi, K), 'res_[{0[0]:.1f},{0[1]:.1f},{0[2]:.1f}]'.format(resolution))
    
    psf_simu = pg.PSF_Generator(size , resolution, psf_type='positive' , psf_model='defocus' , psf_oddity='odd' , name='DefocusPSF', comment = 'ThirdIteration', psf_params= psf_defocus_params )
    psf_simu.createPSF(oversampling=5)
    
    
    psf_simu.initSaveParameters(path, f_ident, overwrite=True)
    psf_simu.saveSolution()
    #[fig, ani] = v_util.imshow3D_ani(psf_simu.out, scale_log=True)
    
    #plt.show()
     
     
#----------------------------------------------------------------------
def createDefocusBatch():
    """"""
    
    size = [3.2,3.2,6.4]
    #resolution = [[0.1,0.1,0.1],[0.2,0.2,0.2]]
    resolution = [[0.4,0.4,0.4],[0.8,0.8,0.8]]
    
    sig_list = [.05,0.1,0.2, 0.3]
    zi_list = [800, 1000, 2000]
    K_list = [100,200,300,400]
    
    f_ident = 'psf_defocus_{:0>3}.tif'    
    
    for res in resolution:
        for sig in sig_list:
            for zi in zi_list:
                for K in K_list:
                    
                    psf_defocus_params = {'sigma':sig, 'zi':zi, 'K':K}
                    path = util.ptjoin(util.SIM_PSF, 'odd', 'defocussing_sig-{:.2f}_zi-{:.0f}_K-{:.0f}'.format(sig, zi, K), 'res_[{0[0]:.1f},{0[1]:.1f},{0[2]:.1f}]'.format(res))
                    
                    psf_simu = pg.PSF_Generator(size , res, psf_type='positive' , psf_model='defocus' , psf_oddity='odd' , name='DefocusPSF', comment = 'FifthIteration', psf_params= psf_defocus_params )
                    psf_simu.createPSF(oversampling=5)
                    
                    
                    psf_simu.initSaveParameters(path, f_ident, overwrite=True)
                    psf_simu.saveSolution()                    
    

#----------------------------------------------------------------------
def createDefocusPSFOversampTest():
    """"""
    
    #size = [6.2, 6.2, 6.4]
    size = [1.6, 1.6, 3.2]
    resolution = [0.1, 0.1, 0.1]
    
    
    
    sig = .05
    zi = 800.
    K = 100.
    psf_defocus_params = {'sigma':sig, 'zi':zi, 'K':K}
    
    f_ident = 'psf_defocus_{:0>3}.tif'
    path = util.ptjoin(util.SIM_PSF, 'defocussing_sig-{:.2f}_zi-{:.0f}_K-{:.0f}'.format(sig, zi, K), 'res_[{0[0]:.1f},{0[1]:.1f},{0[2]:.1f}]'.format(resolution))
    
    psf_simu = pg.PSF_Generator(size , resolution, psf_type='positive' , psf_model='defocus' , psf_oddity='odd' , name='DefocusPSF', comment = 'SecondIteration', psf_params= psf_defocus_params )
    psf_simu.createPSF(oversampling = 3)
    
    psf_simu2 = pg.PSF_Generator(size , resolution, psf_type='positive' , psf_model='defocus' , psf_oddity='odd' , name='DefocusPSF', comment = 'SecondIteration', psf_params= psf_defocus_params )
    psf_simu2.createPSF(oversampling = 5)
    

    psf_diff = psf_simu2.out - psf_simu.out

    print(psf_simu.out.shape)
    print(psf_simu2.out.shape)
    #psf_simu.initSaveParameters(path, f_ident, overwrite=True)
    #psf_simu.saveSolution()
    #[fig, ani] = v_util.imshow3D_ani(psf_simu2.out, scale_log=False)
    #[fig, ani] = v_util.imshow3D_ani(psf_simu.out, scale_log=False)
    [fig, ani] = v_util.imshow3D_ani(psf_diff, scale_log=False)
    
    #plt.figure()
    #plt.subplot(1,2,1)
    #plt.plot(psf_simu.out[:,32,128])
    #plt.plot(psf_simu2.out[:,32,128])
    #plt.plot(psf_diff[:,32,128])
    
    plt.show()    


#----------------------------------------------------------------------
def createGaussPSF():
    """"""    
    size = [3.2, 3.2, 3.2]
    resolution = [0.1, 0.1, 0.1]
    
    sig0 = 0.15
    sig1 = 0.
    sig2 = 0.22
    
    psf_gauss_params = {'sigma_0':sig0, 'sigma_1':sig1, 'sigma_2':sig2}
    
    f_ident = 'psf_gauss_{:0>3}.tif'
    path = util.ptjoin(util.SIM_PSF, 'gauss_sig0-{:.2f}_sig1-{:.2f}_sig2-{:.2f}'.format(sig0, sig1, sig2), 'res_[{0[0]:.1f},{0[1]:.1f},{0[2]:.1f}]'.format(resolution))
    
    psf_simu = pg.PSF_Generator(size , resolution, psf_type='positive' , psf_model='gaussian' , psf_oddity='odd' , name='GaussianPSF', comment = 'SecondIteration', psf_params= psf_gauss_params )
    psf_simu.createPSF()    
    psf_simu.initSaveParameters(path, f_ident, overwrite=True)
    psf_simu.saveSolution()
    
    
    [fig, ani] = v_util.imshow3D_ani(psf_simu.out, scale_log=True)
    plt.show()
    
    
#----------------------------------------------------------------------
def createGaussBatch():
    """"""
    size = [3.2, 3.2, 3.2]
    #resolution = [[0.1, 0.1, 0.1],[0.2,0.2,0.2]]
    resolution = [[0.4,0.4,0.4],[0.8,0.8,0.8]]
    
    sig0_list = [0.05,0.10,0.15,0.2,0.3,0.4]
    sig1_list = [0]
    sig2_list = [0.15, 0.2, 0.25, 0.3, 0.4]
    
    f_ident = 'psf_gauss_{:0>3}.tif'


    for res in resolution:
        for sig0 in sig0_list:
            for sig1 in sig1_list:
                for sig2 in sig2_list:
                    psf_gauss_params = {'sigma_0':sig0, 'sigma_1':sig1, 'sigma_2':sig2}
                    path = util.ptjoin(util.SIM_PSF, 'odd', 'gauss_sig0-{:.2f}_sig1-{:.2f}_sig2-{:.2f}'.format(sig0, sig1, sig2), 'res_[{0[0]:.1f},{0[1]:.1f},{0[2]:.1f}]'.format(res))
                    
                    psf_simu = pg.PSF_Generator(size , res, psf_type='positive' , psf_model='gaussian' , psf_oddity='odd' , name='GaussianPSF', comment = 'FifthIteration', psf_params= psf_gauss_params )
                    psf_simu.createPSF(oversampling= 5)    
                    psf_simu.initSaveParameters(path, f_ident, overwrite=True)
                    psf_simu.saveSolution()                    
                    
    
    
#----------------------------------------------------------------------
def createGaussSimple():
    """"""
    
    size = [1.6, 1.6, 1.6]
    resolution = [0.1, 0.1, 0.1]
    
    sig = 0.3
    
    psf_gauss_simple = {'sigma':sig}
    
    f_ident = 'psf_gauss_simple_{:0>3}.tif'
    path = util.ptjoin(util.SIM_PSF, 'gauss_simple_sig-{:.2f}'.format(sig), 'res_[{0[0]:.1f},{0[1]:.1f},{0[2]:.1f}]'.format(resolution))
    
    psf_simu = pg.PSF_Generator(size , resolution, psf_type='positive' , psf_model='gaussian_simple' , psf_oddity='even' , name='GaussianPSF', comment = 'SecondIteration', psf_params= psf_gauss_simple)
    psf_simu.createPSF()    
    psf_simu.initSaveParameters(path, f_ident, overwrite=True)
    psf_simu.saveSolution()
    
    
    [fig, ani] = v_util.imshow3D_ani(psf_simu.out, scale_log=True)
    plt.show()

#----------------------------------------------------------------------
def createGaussSimpleBatch():
    """"""
    size = [1.6, 1.6, 1.6]
    #resolution = [[0.1, 0.1, 0.1],[0.2,0.2,0.2]]
    resolution = [[0.4,0.4,0.4],[0.8,0.8,0.8]]
    sig_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    f_ident = 'psf_gauss_simple_{:0>3}.tif'
    
    for res in resolution:
        for sig in sig_list:
            path = util.ptjoin(util.SIM_PSF, 'odd', 'gauss_simple_sig-{:.2f}'.format(sig), 'res_[{0[0]:.1f},{0[1]:.1f},{0[2]:.1f}]'.format(res))
            psf_gauss_simple = {'sigma':sig}
            
            psf_simu = pg.PSF_Generator(size , res, psf_type='positive' , psf_model='gaussian_simple' , psf_oddity='odd' , name='GaussianPSF', comment = 'FifthIteration', psf_params= psf_gauss_simple)
            psf_simu.createPSF(oversampling= 5)    
            psf_simu.initSaveParameters(path, f_ident, overwrite=True)
            psf_simu.saveSolution()            
    
if __name__ == '__main__':
    #createGaussPSF()
    #createDefocusPSF()
    #createGaussSimple()
    #createDefocusPSFOversamp()
    createDefocusBatch()
    createGaussBatch()
    createGaussSimpleBatch()
