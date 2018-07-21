# -*- coding: utf-8 -*-
"""
Part of the pyconvolve framework for convolution and deconvolution. 
Author: Lukas Küpper, 2018
License: GPLv3
"""
import numpy as np
import math as mt

import util
import util.visu_util
import util.stack_loader

import sim_creator.sample_builder as sb

import matplotlib.pyplot as plt
import matplotlib.animation as anim

#----------------------------------------------------------------------
def cyto_sim_with_inm(res, seed):
    """"""
    np.random.seed(seed)

    sample_size = [51.0, 51.0, 12.8]
    #sample_size = [47.6, 47.6, 11.2]
    #sample_resolution = [0.3, 0.3, 0.3]
    sample_resolution = res

    val_bg_inner = 220.
    val_bg_outer = 240.
    val_glas = 235.
    val_border = 190.
    val_blood_inner = 120.
    val_blood_outer = 100.
    val_INM = 150. 
    val_pyr_sm = 80.
    val_pyr_bg = 65.
    val_dirt = 110.

    pyr_small_density = 0.08
    pyr_small_size = 2.3

    pyr_big_density = 0.03
    pyr_big_size = 3.2

    n_dirt = 230
    z_pos_dirt = [9.5,-9.5]
    dirt_size = 0.2


    sim_sample = sb.Sim_Sample(sample_size, 
                            sample_resolution, 
                            background= val_bg_outer, 
                            imageConstructType= sb.Sim_Sample.SET, 
                            randomizeOrientation=0., 
                            randomizePosition=0., 
                            sampleParams=None,
                            name = 'Cyto_Sim_w_INM',
                            comments= 'Second Iteration, random seed: {}'.format(seed))

    #Gyros and Sulkus

    sample_params_inner = [['cylinder',{'signal':val_border, 'center':[-0.,-30.,0.], 'size':[42.,42.,25.], 'rotation':[0.,0.,0.], 'anisoDim':2}],
                           ['cylinder',{'signal':val_border, 'center':[-10.,30.,0.], 'size':[42.,42.,25.], 'rotation':[0.,0.,0.], 'anisoDim':2}],
                           ['cylinder',{'signal':val_bg_inner, 'center':[-0.,-30.,0.], 'size':[40.,40.,25.], 'rotation':[0.,0.,0.], 'anisoDim':2}],
                           ['cylinder',{'signal':val_bg_inner, 'center':[-10.,30.,0.], 'size':[40.,40.,25.], 'rotation':[0.,0.,0.], 'anisoDim':2}],
                           ['rectangle',{'signal':val_bg_inner, 'center':[-40.,0.,0.], 'size':[20.,52.,25.], 'rotation':[0.,0.,0.], 'anisoDim':2}]
                           ]
    sim_sample.addSampleParam(sample_params_inner)


    sample_params_INM_out =  [['round_cylinder',{'signal':val_INM, 'center':[0.,-12.,0.], 'size':[5.,0.7,0.7], 'rotation':[0.,0.,0.], 'anisoDim':0}],
                              ['round_cylinder',{'signal':val_INM, 'center':[0.,-9.,0.], 'size':[5.,0.7,0.7], 'rotation':[0.,0.,0.], 'anisoDim':0}],
                              ['round_cylinder',{'signal':val_INM, 'center':[0.,-6.5,0.], 'size':[5.,0.7,0.7], 'rotation':[0.,0.,mt.pi/7.], 'anisoDim':0}],
                              ['round_cylinder',{'signal':val_INM, 'center':[0.,-4.,0.], 'size':[5.,0.7,0.7], 'rotation':[0.,0.,0.], 'anisoDim':0}],
                              ['round_cylinder',{'signal':val_INM, 'center':[0.,-1.,0.], 'size':[5.,0.7,0.7], 'rotation':[0.,0.,0.], 'anisoDim':0}],
                              ['round_cylinder',{'signal':val_INM, 'center':[-3.3,0.7,0.], 'size':[2.5,0.7,0.7], 'rotation':[0.,0.,mt.pi/4.], 'anisoDim':0}],
                              ['round_cylinder',{'signal':val_INM, 'center':[-3.3,4.3,0.], 'size':[2.5,0.7,0.7], 'rotation':[0.,0.,-mt.pi/4.], 'anisoDim':0}],
                              ['round_cylinder',{'signal':val_INM, 'center':[0.,6.,0.], 'size':[5.,0.7,0.7], 'rotation':[0.,0.,0.], 'anisoDim':0}],
                              ['round_cylinder',{'signal':val_INM, 'center':[-3.3,10.3,0.], 'size':[2.5,0.7,0.7], 'rotation':[0.,0.,-mt.pi/4.], 'anisoDim':0}],
                              ['round_cylinder',{'signal':val_INM, 'center':[0.,12.,0.], 'size':[5.,0.7,0.7], 'rotation':[0.,0.,0.], 'anisoDim':0}]
                              ]    

    x_shift = 44.
    y_shift = 35.

    for samp in sample_params_INM_out:
        t_cent = samp[1]['center']
        t_cent = [t_cent[0]+x_shift, t_cent[1]+y_shift, t_cent[2]]
        samp[1]['center'] = t_cent
    sim_sample.addSampleParam(sample_params_INM_out)    


    sample_params_limits = [['rectangle',{'signal':val_glas, 'center':[0.,0.,11.2], 'size':[52.,52.,2.], 'rotation':[0.,0.,0.], 'anisoDim':2}],
                            ['rectangle',{'signal':val_glas, 'center':[0.,0.,-11.2], 'size':[52.,52.,2.], 'rotation':[0.,0.,0.], 'anisoDim':2}],
                            ['rectangle',{'signal':val_bg_outer, 'center':[0.,0.,11.2], 'size':[52.,52.,1.75], 'rotation':[0.,0.,0.], 'anisoDim':2}],
                            ['rectangle',{'signal':val_bg_outer, 'center':[0.,0.,-11.2], 'size':[52.,52.,1.75], 'rotation':[0.,0.,0.], 'anisoDim':2}]]
    sim_sample.addSampleParam(sample_params_limits)    


    #Constructing Dirt
    
    sample_params_dirt = []
    
    for z in z_pos_dirt:
        for n in range(n_dirt):
            
            t_rand = np.random.rand(3)
            x = (2*sample_size[0]-2.)*t_rand[0]- sample_size[0]
            y = (2*sample_size[1]-2.)*t_rand[1]- sample_size[1]
            
            pos = [x,y,z]
            size = t_rand[2]*0.8*dirt_size + 0.7*dirt_size
            
            sample_params_dirt.append(['ellipse',{'signal':val_dirt, 'center':pos, 'size':[size,size,size], 'rotation':[0.,0.,0.], 'anisoDim':0}])
        
    sim_sample.addSampleParam(sample_params_dirt)        


    #Pre-constructing index to delete unwanted pyramids
    sim_sample.constructSample()
    temp_index_glas = (sim_sample.out == val_glas)
    temp_index_bg = (sim_sample.out == val_bg_outer)
    temp_index_bg_inner = (sim_sample.out == val_bg_inner)
    temp_index_border =  (sim_sample.out == val_border)
    temp_index_INM = (sim_sample.out == val_INM)
    temp_index_dirt = (sim_sample.out == val_dirt)

    #del sim_sample
    #Constructing Pyramid Cells
    

    sim_sample = sb.Sim_Sample(sample_size, 
                            sample_resolution, 
                            background= val_bg_outer, 
                            imageConstructType= sb.Sim_Sample.SET, 
                            randomizeOrientation=0., 
                            randomizePosition=0., 
                            sampleParams=None,
                            name = 'Cyto_Sim_w_INM',
                            comments= 'random seed: {}'.format(seed))    
   

    #sim_sample.out[temp_index_glas] = val_glas
    #sim_sample.out[temp_index_bg] = val_bg_outer
    sim_sample.out[temp_index_bg_inner] = val_bg_inner
    #sim_sample.out[temp_index_border] = val_border
    #sim_sample.out[temp_index_INM] = val_INM
    #sim_sample.out[temp_index_dirt] = val_dirt
    

    sm_pyr_coordinates = [np.linspace(-sample_size[i]+1./pyr_small_density, sample_size[i]-1./pyr_small_density, 2*sample_size[i]*pyr_small_density) for i in range(3)]
    sm_pyr_coordinates[2] = [-6., 0., 6.]

    bg_pyr_coordinates = [np.linspace(-sample_size[i]+1./pyr_big_density, sample_size[i]-1./pyr_big_density, 2*sample_size[i]*pyr_big_density) for i in range(3)]
    bg_pyr_coordinates[2] = [-4., 4.] 
    

    h_sm = 0.5*mt.sqrt(3)*pyr_small_size
    h_bg = 0.5*mt.sqrt(3)*pyr_big_size

    rnd_disp = [0.5/pyr_small_density, 0.5/pyr_small_density, 0.5/pyr_small_density]
    for x in sm_pyr_coordinates[0]:
        for y in sm_pyr_coordinates[1]:
            for z in sm_pyr_coordinates[2]:

                t_cent = [x,y,z]

                t_displ = list(np.random.rand(3))
                t_displ = [t_displ[ind]*rnd_disp[ind] for ind in range(3)]
                t_rot = list(np.random.rand(3)*mt.pi)

                t_cent = [t_displ[ind]+t_cent[ind] for ind in range(3)]
                t_size_var = np.random.rand(1)*0.4+0.8

                if any([t_cent[ind] <= -sample_size[ind] or t_cent[ind] >= sample_size[ind] for ind in range(3)]):
                    continue

                sim_sample.addSampleParam(['pyramid',{'signal':val_pyr_sm, 'center':t_cent, 'size':[t_size_var*pyr_small_size,t_size_var*pyr_small_size,t_size_var*h_sm], 'rotation':t_rot, 'anisoDim':2}])

    rnd_disp = [0.5/pyr_big_density, 0.5/pyr_big_density, 0.5/pyr_big_density]            
    for x in bg_pyr_coordinates[0]:
        for y in bg_pyr_coordinates[1]:
            for z in bg_pyr_coordinates[2]:

                t_cent = [x,y,z]

                t_displ = list(np.random.rand(3))
                t_displ = [t_displ[ind]*rnd_disp[ind] for ind in range(3)]
                t_rot = list(np.random.rand(3)*mt.pi)

                t_cent = [t_displ[ind]+t_cent[ind] for ind in range(3)]
                t_size_var = np.random.rand(1)*0.4+0.8

                if any([t_cent[ind] <= -sample_size[ind] or t_cent[ind] >= sample_size[ind] for ind in range(3)]):
                    continue                

                sim_sample.addSampleParam(['pyramid',{'signal':val_pyr_bg, 'center':t_cent, 'size':[t_size_var*pyr_big_size,t_size_var*pyr_big_size,t_size_var*h_bg], 'rotation':t_rot, 'anisoDim':2}])    

    #Constructing Blood Vessels

    sample_params_blood_vessel_out = [['cylinder',{'signal':val_blood_inner, 'center':[-20.,-20.,0.], 'size':[3.,3.,30.], 'rotation':[mt.pi/6.,mt.pi/5.,0.], 'anisoDim':2}],
                                      ['cylinder',{'signal':val_blood_inner, 'center':[-35.,40.,0.], 'size':[3.2,2.7, 18.], 'rotation':[-0.1,0.65,0.], 'anisoDim':2}],
                                      ['cylinder',{'signal':val_blood_inner, 'center':[-40.,-40.,0.], 'size':[4.,3.2,18.], 'rotation':[0.4,-0.3,0.], 'anisoDim':2}],
                                      #['cylinder',{'signal':val_blood_inner, 'center':[22.,-30.,0.], 'size':[3.4,2.4,18.], 'rotation':[-0.3,0.4,0.], 'anisoDim':2}],
                                      ['cylinder',{'signal':val_blood_inner, 'center':[-20.,0.,0.], 'size':[5.1,4.6,18.], 'rotation':[-0.3,-0.2,0.], 'anisoDim':2}]]

    sample_params_blood_vessel_in = [['cylinder',{'signal':val_blood_outer, 'center':[-20.,-20.,0.], 'size':[2.,2.,30.], 'rotation':[mt.pi/6.,mt.pi/5.,0.], 'anisoDim':2}],
                                     ['cylinder',{'signal':val_blood_outer, 'center':[-35.,40.,0.], 'size':[2.2,1.9, 18.], 'rotation':[-0.1,0.65,0.], 'anisoDim':2}],
                                     ['cylinder',{'signal':val_blood_outer, 'center':[-40.,-40.,0.], 'size':[3.1,2.2,18.], 'rotation':[0.4,-0.3,0.], 'anisoDim':2}],
                                     #['cylinder',{'signal':val_blood_outer, 'center':[22.,-30.,0.], 'size':[2.6,1.8,18.], 'rotation':[-0.3,0.4,0.], 'anisoDim':2}],
                                     ['cylinder',{'signal':val_blood_outer, 'center':[-20.,0.,0.], 'size':[4.1,3.3,18.], 'rotation':[-0.3,-0.2,0.], 'anisoDim':2}]]    
    sim_sample.addSampleParam(sample_params_blood_vessel_out)
    sim_sample.addSampleParam(sample_params_blood_vessel_in)

    



    sim_sample.constructSample()

    sim_sample.out[temp_index_glas] = val_glas
    sim_sample.out[temp_index_bg] = val_bg_outer
    sim_sample.out[temp_index_border] = val_border
    sim_sample.out[temp_index_INM] = val_INM
    sim_sample.out[temp_index_dirt] = val_dirt
    
    return sim_sample



    
    
    
    
    
    
    
#----------------------------------------------------------------------
def create_cyto():
    """"""
    f_ident = 'cyto_sim_{:0>3}.tif'
    save_path = util.ptjoin(util.SIM_DATA, 'data', 'cyto_sim_resol-[0.2,0.2,0.2]')
    
    resolution = [0.2, 0.2, 0.2]
    seed = 12345
    
    sim_sample = cyto_sim_with_inm(resolution, seed)
    
    sim_sample.initSaveParameters(save_path, f_ident, overwrite= True)
    
    
    #[fig,ani] = util.visu_util.imshow3D_ani(sim_sample.out)
    
    #util.utilities.imshow3D_slice(sim_sample.out)
    #plt.show()

    print(sim_sample.out.shape)
    #sim_sample.out = sim_sample.out[0:512,0:512,0:128]
    sim_sample.saveSolution()


#----------------------------------------------------------------------
def create_psf_recon_sample():
    """"""

    resolution = [0.4,0.4,0.4]
    f_ident = 'psf_recon_{:0>3}.tif'
    
    save_path = util.ptjoin(util.SIM_DATA, 'psf_recon_full', 'full_res_[{0[0]:.1f},{0[1]:.1f},{0[2]:.1f}]'.format(resolution))
    
    seed = 12345
    
    sim_sample = psf_recon_sample(resolution, seed)
    sim_sample.initSaveParameters(save_path, f_ident, overwrite = True)
    sim_sample.saveSolution()
    
    
#----------------------------------------------------------------------
def psf_recon_sample(res, seed):
    """"""
    
    np.random.seed(seed)

    sample_size = [51.0, 51.0, 12.8]
    #sample_size = [47.6, 47.6, 11.2]
    #sample_resolution = [0.3, 0.3, 0.3]
    sample_resolution = res

    val_bg_inner = 220.
    val_bg_outer = 240.
    val_glas = 235.
    val_border = 190.
    val_blood_inner = 120.
    val_blood_outer = 100.
    val_INM = 150. 
    val_pyr_sm = 80.
    val_pyr_bg = 65.
    val_dirt = 110.

    pyr_small_density = 0.08
    pyr_small_size = 2.3

    pyr_big_density = 0.03
    pyr_big_size = 3.2

    n_dirt = 230
    z_pos_dirt = [9.5,-9.5]
    dirt_size = 0.2


    sim_sample = sb.Sim_Sample(sample_size, 
                            sample_resolution, 
                            background= val_bg_outer, 
                            imageConstructType= sb.Sim_Sample.SET, 
                            randomizeOrientation=0., 
                            randomizePosition=0., 
                            sampleParams=None,
                            name = 'Cyto_Sim_w_INM',
                            comments= 'Second Iteration, random seed: {}'.format(seed))

    #Gyros and Sulkus

    sample_params_inner = [['cylinder',{'signal':val_border, 'center':[-0.,-30.,0.], 'size':[42.,42.,25.], 'rotation':[0.,0.,0.], 'anisoDim':2}],
                           ['cylinder',{'signal':val_border, 'center':[-10.,30.,0.], 'size':[42.,42.,25.], 'rotation':[0.,0.,0.], 'anisoDim':2}],
                           ['cylinder',{'signal':val_bg_inner, 'center':[-0.,-30.,0.], 'size':[40.,40.,25.], 'rotation':[0.,0.,0.], 'anisoDim':2}],
                           ['cylinder',{'signal':val_bg_inner, 'center':[-10.,30.,0.], 'size':[40.,40.,25.], 'rotation':[0.,0.,0.], 'anisoDim':2}],
                           ['rectangle',{'signal':val_bg_inner, 'center':[-40.,0.,0.], 'size':[20.,52.,25.], 'rotation':[0.,0.,0.], 'anisoDim':2}]
                           ]
    sim_sample.addSampleParam(sample_params_inner)


    sample_params_INM_out =  [['round_cylinder',{'signal':val_INM, 'center':[0.,-12.,0.], 'size':[5.,0.7,0.7], 'rotation':[0.,0.,0.], 'anisoDim':0}],
                              ['round_cylinder',{'signal':val_INM, 'center':[0.,-9.,0.], 'size':[5.,0.7,0.7], 'rotation':[0.,0.,0.], 'anisoDim':0}],
                              ['round_cylinder',{'signal':val_INM, 'center':[0.,-6.5,0.], 'size':[5.,0.7,0.7], 'rotation':[0.,0.,mt.pi/7.], 'anisoDim':0}],
                              ['round_cylinder',{'signal':val_INM, 'center':[0.,-4.,0.], 'size':[5.,0.7,0.7], 'rotation':[0.,0.,0.], 'anisoDim':0}],
                              ['round_cylinder',{'signal':val_INM, 'center':[0.,-1.,0.], 'size':[5.,0.7,0.7], 'rotation':[0.,0.,0.], 'anisoDim':0}],
                              ['round_cylinder',{'signal':val_INM, 'center':[-3.3,0.7,0.], 'size':[2.5,0.7,0.7], 'rotation':[0.,0.,mt.pi/4.], 'anisoDim':0}],
                              ['round_cylinder',{'signal':val_INM, 'center':[-3.3,4.3,0.], 'size':[2.5,0.7,0.7], 'rotation':[0.,0.,-mt.pi/4.], 'anisoDim':0}],
                              ['round_cylinder',{'signal':val_INM, 'center':[0.,6.,0.], 'size':[5.,0.7,0.7], 'rotation':[0.,0.,0.], 'anisoDim':0}],
                              ['round_cylinder',{'signal':val_INM, 'center':[-3.3,10.3,0.], 'size':[2.5,0.7,0.7], 'rotation':[0.,0.,-mt.pi/4.], 'anisoDim':0}],
                              ['round_cylinder',{'signal':val_INM, 'center':[0.,12.,0.], 'size':[5.,0.7,0.7], 'rotation':[0.,0.,0.], 'anisoDim':0}]
                              ]    

    x_shift = 44.
    y_shift = 35.

    for samp in sample_params_INM_out:
        t_cent = samp[1]['center']
        t_cent = [t_cent[0]+x_shift, t_cent[1]+y_shift, t_cent[2]]
        samp[1]['center'] = t_cent
    #sim_sample.addSampleParam(sample_params_INM_out)    


    sample_params_limits = [['rectangle',{'signal':val_glas, 'center':[0.,0.,8.], 'size':[52.,52.,7.75], 'rotation':[0.,0.,0.], 'anisoDim':2}],
                            ['rectangle',{'signal':val_glas, 'center':[0.,0.,-8.], 'size':[52.,52.,7.75], 'rotation':[0.,0.,0.], 'anisoDim':2}],
                            ['rectangle',{'signal':val_bg_outer, 'center':[0.,0.,8.], 'size':[52.,52.,7.5], 'rotation':[0.,0.,0.], 'anisoDim':2}],
                            ['rectangle',{'signal':val_bg_outer, 'center':[0.,0.,-8.], 'size':[52.,52.,7.5], 'rotation':[0.,0.,0.], 'anisoDim':2}]]
    sim_sample.addSampleParam(sample_params_limits)    


    #Constructing Dirt
    
    sample_params_dirt = []
    
    for z in z_pos_dirt:
        for n in range(n_dirt):
            
            t_rand = np.random.rand(3)
            x = (2*sample_size[0]-2.)*t_rand[0]- sample_size[0]
            y = (2*sample_size[1]-2.)*t_rand[1]- sample_size[1]
            
            pos = [x,y,z]
            size = t_rand[2]*0.8*dirt_size + 0.7*dirt_size
            
            sample_params_dirt.append(['ellipse',{'signal':val_dirt, 'center':pos, 'size':[size,size,size], 'rotation':[0.,0.,0.], 'anisoDim':0}])
        
    sim_sample.addSampleParam(sample_params_dirt)        


    #Pre-constructing index to delete unwanted pyramids
    sim_sample.constructSample()
    temp_index_glas = (sim_sample.out == val_glas)
    temp_index_bg = (sim_sample.out == val_bg_outer)
    temp_index_bg_inner = (sim_sample.out == val_bg_inner)
    temp_index_border =  (sim_sample.out == val_border)
    temp_index_INM = (sim_sample.out == val_INM)
    temp_index_dirt = (sim_sample.out == val_dirt)

    #del sim_sample
    #Constructing Pyramid Cells
    

    sim_sample = sb.Sim_Sample(sample_size, 
                            sample_resolution, 
                            background= val_bg_outer, 
                            imageConstructType= sb.Sim_Sample.SET, 
                            randomizeOrientation=0., 
                            randomizePosition=0., 
                            sampleParams=None,
                            name = 'Cyto_Sim_w_INM',
                            comments= 'random seed: {}'.format(seed))    
   

    #sim_sample.out[temp_index_glas] = val_glas
    #sim_sample.out[temp_index_bg] = val_bg_outer
    sim_sample.out[temp_index_bg_inner] = val_bg_inner
    #sim_sample.out[temp_index_border] = val_border
    #sim_sample.out[temp_index_INM] = val_INM
    #sim_sample.out[temp_index_dirt] = val_dirt
    

    sm_pyr_coordinates = [np.linspace(-sample_size[i]+1./pyr_small_density, sample_size[i]-1./pyr_small_density, 2*sample_size[i]*pyr_small_density) for i in range(3)]
    sm_pyr_coordinates[2] = [-6., 0., 6.]

    bg_pyr_coordinates = [np.linspace(-sample_size[i]+1./pyr_big_density, sample_size[i]-1./pyr_big_density, 2*sample_size[i]*pyr_big_density) for i in range(3)]
    bg_pyr_coordinates[2] = [-4., 4.] 
    

    h_sm = 0.5*mt.sqrt(3)*pyr_small_size
    h_bg = 0.5*mt.sqrt(3)*pyr_big_size

    rnd_disp = [0.5/pyr_small_density, 0.5/pyr_small_density, 0.5/pyr_small_density]
    for x in sm_pyr_coordinates[0]:
        for y in sm_pyr_coordinates[1]:
            for z in sm_pyr_coordinates[2]:

                t_cent = [x,y,z]

                t_displ = list(np.random.rand(3))
                t_displ = [t_displ[ind]*rnd_disp[ind] for ind in range(3)]
                t_rot = list(np.random.rand(3)*mt.pi)

                t_cent = [t_displ[ind]+t_cent[ind] for ind in range(3)]
                t_size_var = np.random.rand(1)*0.4+0.8

                if any([t_cent[ind] <= -sample_size[ind] or t_cent[ind] >= sample_size[ind] for ind in range(3)]):
                    continue

                sim_sample.addSampleParam(['pyramid',{'signal':val_pyr_sm, 'center':t_cent, 'size':[t_size_var*pyr_small_size,t_size_var*pyr_small_size,t_size_var*h_sm], 'rotation':t_rot, 'anisoDim':2}])

    rnd_disp = [0.5/pyr_big_density, 0.5/pyr_big_density, 0.5/pyr_big_density]            
    for x in bg_pyr_coordinates[0]:
        for y in bg_pyr_coordinates[1]:
            for z in bg_pyr_coordinates[2]:

                t_cent = [x,y,z]

                t_displ = list(np.random.rand(3))
                t_displ = [t_displ[ind]*rnd_disp[ind] for ind in range(3)]
                t_rot = list(np.random.rand(3)*mt.pi)

                t_cent = [t_displ[ind]+t_cent[ind] for ind in range(3)]
                t_size_var = np.random.rand(1)*0.4+0.8

                if any([t_cent[ind] <= -sample_size[ind] or t_cent[ind] >= sample_size[ind] for ind in range(3)]):
                    continue                

                sim_sample.addSampleParam(['pyramid',{'signal':val_pyr_bg, 'center':t_cent, 'size':[t_size_var*pyr_big_size,t_size_var*pyr_big_size,t_size_var*h_bg], 'rotation':t_rot, 'anisoDim':2}])    

    #Constructing Blood Vessels

    sample_params_blood_vessel_out = [['cylinder',{'signal':val_blood_inner, 'center':[-20.,-20.,0.], 'size':[3.,3.,30.], 'rotation':[mt.pi/6.,mt.pi/5.,0.], 'anisoDim':2}],
                                      ['cylinder',{'signal':val_blood_inner, 'center':[-35.,40.,0.], 'size':[3.2,2.7, 18.], 'rotation':[-0.1,0.65,0.], 'anisoDim':2}],
                                      ['cylinder',{'signal':val_blood_inner, 'center':[-40.,-40.,0.], 'size':[4.,3.2,18.], 'rotation':[0.4,-0.3,0.], 'anisoDim':2}],
                                      #['cylinder',{'signal':val_blood_inner, 'center':[22.,-30.,0.], 'size':[3.4,2.4,18.], 'rotation':[-0.3,0.4,0.], 'anisoDim':2}],
                                      ['cylinder',{'signal':val_blood_inner, 'center':[-20.,0.,0.], 'size':[5.1,4.6,18.], 'rotation':[-0.3,-0.2,0.], 'anisoDim':2}]]

    sample_params_blood_vessel_in = [['cylinder',{'signal':val_blood_outer, 'center':[-20.,-20.,0.], 'size':[2.,2.,30.], 'rotation':[mt.pi/6.,mt.pi/5.,0.], 'anisoDim':2}],
                                     ['cylinder',{'signal':val_blood_outer, 'center':[-35.,40.,0.], 'size':[2.2,1.9, 18.], 'rotation':[-0.1,0.65,0.], 'anisoDim':2}],
                                     ['cylinder',{'signal':val_blood_outer, 'center':[-40.,-40.,0.], 'size':[3.1,2.2,18.], 'rotation':[0.4,-0.3,0.], 'anisoDim':2}],
                                     #['cylinder',{'signal':val_blood_outer, 'center':[22.,-30.,0.], 'size':[2.6,1.8,18.], 'rotation':[-0.3,0.4,0.], 'anisoDim':2}],
                                     ['cylinder',{'signal':val_blood_outer, 'center':[-20.,0.,0.], 'size':[4.1,3.3,18.], 'rotation':[-0.3,-0.2,0.], 'anisoDim':2}]]    
    sim_sample.addSampleParam(sample_params_blood_vessel_out)
    sim_sample.addSampleParam(sample_params_blood_vessel_in)

    



    sim_sample.constructSample()

    sim_sample.out[temp_index_glas] = val_glas
    sim_sample.out[temp_index_bg] = val_bg_outer
    sim_sample.out[temp_index_border] = val_border
    sim_sample.out[temp_index_INM] = val_INM
    sim_sample.out[temp_index_dirt] = val_dirt
    
    return sim_sample
    
    

#----------------------------------------------------------------------
def cyto_downsample():
    """"""
    
    f_ident = 'cyto_sim_{:0>3}.tif'
    in_main_path = util.ptjoin(util.SIM_DATA, 'cyto_sim_full')

    

    res_path_add = 'res_[{0[0]:.1f},{0[1]:.1f},{0[2]:.1f}]'

    out_path_add = 'downsample_{}'

    path_in = util.ptjoin(util.SIM_DATA, 'cyto_sim_full', res_path_add.format([0.1,0.1,0.1]))
    path_out = util.ptjoin(util.SIM_DATA, 'cyto_sim_full', out_path_add.format(res_path_add.format([0.2,0.2,0.2])))

    in_stack = util.stack_loader.read_image_stack(path_in, f_ident, meta=False)
    in_meta = util.stack_loader.read_meta_data_only(path_in, f_ident)
    
    cur_shape = in_stack.shape
    new_shape = [1+(csh-1)/2 for csh in cur_shape]
    arr_hyb = in_stack.copy()
    for dim in range(3):
        slic_ind1 = [slice(0,-1,2) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
        slic_ind2 = [slice(1,None,2) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
        slic_ind3 = [slice(2,None,2) if sl_ind == dim else slice(None, None, 1) for sl_ind in range(3)]
        arr_hyb = 0.25 * arr_hyb[slic_ind1]+ 0.5 *  arr_hyb[slic_ind2] + 0.25 * arr_hyb[slic_ind3]    
    
    in_meta.max_v = arr_hyb.max()
    out_stack = (255.*arr_hyb/arr_hyb.max()).astype('uint8')
    
    in_meta.size = new_shape
    in_meta.path = path_out
    in_meta.resolution = [0.2,0.2,0.2]
    in_meta.comment = in_meta.comment + '; Downsampled'
    
    util.stack_loader.write_image_stack(out_stack, path_out, f_ident, 0, meta_data=in_meta.toList())
    
    

#----------------------------------------------------------------------
def createSupCrops_forDownsample():
    """"""
    main_path = util.ptjoin(util.SIM_DATA,'cyto_sim_full')
    res_path_add = 'downsample_res_[0.2,0.2,0.2]'
    f_ident = 'cyto_sim_{:0>3}.tif'
    
    out_path_add = 'cyto_sim_crop_{}_{}'
    
    crop_diviser = 2
    
    path_in = util.ptjoin(main_path, res_path_add)
    
    in_stack = util.stack_loader.read_image_stack(path_in, f_ident, meta=False)
    in_meta = util.stack_loader.read_meta_data_only(path_in,f_ident)
    
    red_shape = [sh/crop_diviser for sh in in_stack.shape]
    meta_comment = in_meta.comment
    
    for x_ind in range(crop_diviser):
        for y_ind in range(crop_diviser):
            
            cur_path_out = util.ptjoin(util.SIM_DATA, out_path_add.format(x_ind,y_ind), res_path_add)
            print('Processing path: {}').format(cur_path_out)
            util.createAllPaths(cur_path_out)
            
            ind_slice = [slice(x_ind*red_shape[0],(x_ind+1)*red_shape[0],1), slice(y_ind*red_shape[1],(y_ind+1)*red_shape[1],1), slice(None,None,1)]
            
            cropped_stack = in_stack[ind_slice].copy()
            
            in_meta.comment = meta_comment+'; Cropped; Crop_index=[{},{}]'.format(x_ind,y_ind)
            util.stack_loader.write_image_stack(cropped_stack, cur_path_out, f_ident, 0, meta_data= in_meta.toList())            
        
    out_path_add = 'cyto_sim_crop_overlap_{}_{}'
        
        
    for x_ind in range(crop_diviser+1):
        for y_ind in range(crop_diviser+1):
            
            cur_path_out = util.ptjoin(util.SIM_DATA, out_path_add.format(x_ind,y_ind), res_path_add)
            print('Processing path:{}'.format(cur_path_out))
            util.createAllPaths(cur_path_out)
            
            ind_slice = [slice(x_ind*(red_shape[0]/2),(x_ind+2)*(red_shape[0]/2),1), slice(y_ind*(red_shape[1]/2),(y_ind+2)*(red_shape[1]/2),1), slice(None,None,1)]
            print(ind_slice)
            cropped_stack = in_stack[ind_slice].copy()
            
            in_meta.comment = meta_comment+'; Cropped; Crop_index=[{},{}]'.format(x_ind,y_ind)
            util.stack_loader.write_image_stack(cropped_stack, cur_path_out, f_ident, 0, meta_data= in_meta.toList())              
    
        

#----------------------------------------------------------------------
def create_sub_crops():
    """"""
    

    f_ident = 'cyto_sim_{:0>3}.tif'
    in_main_path = util.ptjoin(util.SIM_DATA,'cyto_sim_full')
    
    path_full = util.ptjoin(in_main_path, 'full_res_[0.2,0.2,0.2]')
    path_down = util.ptjoin(in_main_path, 'downsample_res_[0.2,0.2,0.2]')
    
    crop_diviser = 2
    
    crop_paths_full = ['full_crop_{}_{}_res_[0.2,0.2,0.2]'.format(i/crop_diviser, i%crop_diviser) for i in range(crop_diviser**2)]
    crop_paths_down = ['downsampled_crop_{}_{}_res_[0.2,0.2,0.2]'.format(i/crop_diviser, i%crop_diviser) for i in range(crop_diviser**2)]
    
    crop_paths_full = [util.ptjoin(in_main_path, pt) for pt in crop_paths_full]
    crop_paths_down = [util.ptjoin(in_main_path, pt) for pt in crop_paths_down]
    
    util.createAllPaths(crop_paths_full)
    util.createAllPaths(crop_paths_down)
    
    full_stack = util.stack_loader.read_image_stack(path_full, f_ident, meta=False)
    full_meta = util.stack_loader.read_meta_data_only(path_full, f_ident)
    
    t_reduced_shape = [sh/crop_diviser if sh%2 == 0 else 1+sh/crop_diviser for sh in full_stack.shape]
    meta_comment = full_meta.comment
    
    for i,crop_path in enumerate(crop_paths_full):
        x_ind = i/crop_diviser
        y_ind = i%crop_diviser
        
        sh_bool2 = [sh%2 == 0 for sh in t_reduced_shape]
        ind_slice = [slice(x_ind*t_reduced_shape[0] if sh_bool2[0] else x_ind*(t_reduced_shape[0]-1),
                           (x_ind+1)*t_reduced_shape[0],1),
                     slice(y_ind*t_reduced_shape[1] if sh_bool2[1] else y_ind*(t_reduced_shape[1]-1),
                           (y_ind+1)*t_reduced_shape[1],1),
                     slice(None,None,1)]
        
        cropped_stack = full_stack[ind_slice].copy()
        full_meta.comment = meta_comment+'; Cropped; Crop_index=[{},{}]'.format(x_ind, y_ind)
        util.stack_loader.write_image_stack(cropped_stack, crop_path, f_ident, 0, meta_data=full_meta.toList())
        
    down_stack = util.stack_loader.read_image_stack(path_down, f_ident, meta=False)
    down_meta = util.stack_loader.read_meta_data_only(path_down, f_ident)
    
    t_reduced_shape = [sh/crop_diviser if sh%2 == 0 else 1+sh/crop_diviser for sh in down_stack.shape]
    meta_comment = down_meta.comment    
    
    for i,crop_path in enumerate(crop_paths_down):
        x_ind = i/crop_diviser
        y_ind = i%crop_diviser
        
        sh_bool2 = [sh%2 == 0 for sh in t_reduced_shape]
        ind_slice = [slice(x_ind*t_reduced_shape[0] if sh_bool2[0] else x_ind*(t_reduced_shape[0]-1),
                           (x_ind+1)*t_reduced_shape[0],1),
                     slice(y_ind*t_reduced_shape[1] if sh_bool2[1] else y_ind*(t_reduced_shape[1]-1),
                           (y_ind+1)*t_reduced_shape[1],1),
                     slice(None,None,1)]
        
        cropped_stack = down_stack[ind_slice].copy()
        down_meta.comment = meta_comment+'; Cropped; Crop_index=[{},{}]'.format(x_ind, y_ind)
        util.stack_loader.write_image_stack(cropped_stack, crop_path, f_ident, 0, meta_data=down_meta.toList())    
            
                
            
    

#----------------------------------------------------------------------
def create_sub_crops_overlap():
    """"""
    
    f_ident = 'cyto_sim_{:0>3}.tif'
    in_main_path = util.ptjoin(util.SIM_DATA,'cyto_sim_full')
    
    path_full = util.ptjoin(in_main_path, 'full_res_[0.2,0.2,0.2]')
    path_down = util.ptjoin(in_main_path, 'downsample_res_[0.2,0.2,0.2]')
    
    crop_diviser = 3
    
    crop_paths_full = ['full_overlap_crop_{}_{}_res_[0.2,0.2,0.2]'.format(i/crop_diviser, i%crop_diviser) for i in range(crop_diviser**2)]
    crop_paths_down = ['downsampled_overlap_crop_{}_{}_res_[0.2,0.2,0.2]'.format(i/crop_diviser, i%crop_diviser) for i in range(crop_diviser**2)]
    
    crop_paths_full = [util.ptjoin(in_main_path, pt) for pt in crop_paths_full]
    crop_paths_down = [util.ptjoin(in_main_path, pt) for pt in crop_paths_down]
    
    util.createAllPaths(crop_paths_full)
    util.createAllPaths(crop_paths_down)
    
    full_stack = util.stack_loader.read_image_stack(path_full, f_ident, meta=False)
    full_meta = util.stack_loader.read_meta_data_only(path_full, f_ident)
    
    t_reduced_shape = [sh/crop_diviser if sh%2 == 0 else 1+sh/crop_diviser for sh in full_stack.shape]
    meta_comment = full_meta.comment
    
    for i,crop_path in enumerate(crop_paths_full):
        x_ind = i/crop_diviser
        y_ind = i%crop_diviser
        
        sh_bool2 = [sh%2 == 0 for sh in t_reduced_shape]
        ind_slice = [slice(x_ind*(t_reduced_shape[0]/2) if sh_bool2[0] else x_ind*((t_reduced_shape[0]-1)/2),
                           (x_ind+2)*(t_reduced_shape[0]/2) if sh_bool2[0] else 1+(x_ind+2)*(t_reduced_shape[0]-1)/2,
                           1), 
                     slice(y_ind*(t_reduced_shape[1]/2) if sh_bool2[1] else y_ind*((t_reduced_shape[1]-1)/2),
                           (y_ind+2)*(t_reduced_shape[1]/2) if sh_bool2[1] else 1+(y_ind+2)*(t_reduced_shape[1]-1)/2, 
                           1),
                     slice(None,None,1)]
        
        cropped_stack = full_stack[ind_slice].copy()
        full_meta.comment = meta_comment+'; Cropped; Crop_index=[{},{}]'.format(x_ind, y_ind)
        util.stack_loader.write_image_stack(cropped_stack, crop_path, f_ident, 0, meta_data=full_meta.toList())
        
    down_stack = util.stack_loader.read_image_stack(path_down, f_ident, meta=False)
    down_meta = util.stack_loader.read_meta_data_only(path_down, f_ident)
    
    t_reduced_shape = [sh/crop_diviser if sh%2 == 0 else 1+sh/crop_diviser for sh in down_stack.shape]
    meta_comment = down_meta.comment    
    
    for i,crop_path in enumerate(crop_paths_down):
        x_ind = i/crop_diviser
        y_ind = i%crop_diviser
        
        sh_bool2 = [sh%2 == 0 for sh in t_reduced_shape]
        ind_slice = [slice(x_ind*(t_reduced_shape[0]/2) if sh_bool2[0] else x_ind*((t_reduced_shape[0]-1)/2),
                           (x_ind+2)*(t_reduced_shape[0]/2) if sh_bool2[0] else 1+(x_ind+2)*(t_reduced_shape[0]-1)/2,
                           1), 
                     slice(y_ind*(t_reduced_shape[1]/2) if sh_bool2[1] else y_ind*((t_reduced_shape[1]-1)/2),
                           (y_ind+2)*(t_reduced_shape[1]/2) if sh_bool2[1] else 1+(y_ind+2)*(t_reduced_shape[1]-1)/2, 
                           1),
                     slice(None,None,1)]
        
        cropped_stack = down_stack[ind_slice].copy()
        down_meta.comment = meta_comment+'; Cropped; Crop_index=[{},{}]'.format(x_ind, y_ind)
        util.stack_loader.write_image_stack(cropped_stack, crop_path, f_ident, 0, meta_data=down_meta.toList())        



if __name__ == '__main__':
    #create_sub_crops()
    #create_sub_crops_overlap()
    #cyto_downsample()
    #create_cyto()
    #createSupCrops_forDownsample()
    create_psf_recon_sample()