import os
import numpy as np

import matplotlib.pyplot as plt 
import matplotlib.animation as anim
from matplotlib import cm   
 
 
 
CMAP = ['gray', 'Greys', 'viridis', 'inferno', 'CMRmap']
 
#----------------------------------------------------------------------
def imshow3D_ani(img_array, ani_dim = 2, cmap='gray', scale_log = False, cutoff=1., cutoffRelative=True, repeat_delay = 50, interval = 50, useArrayWideMax = True):
    """"""
    

        
        
    min_val = img_array.min()
    
    if cutoffRelative:
        max_val = img_array.max()*cutoff
    else:
        max_val = cutoff
    
    if scale_log:
        img_array = img_array.copy()    
        
        img_array = (img_array + min_val)
        img_array = (100.*img_array / img_array.mean())+1.
        img_array = np.log(img_array)
        
        min_val = img_array.min()
        max_val = img_array.max()
    
    
    sl_iter = [slice(None,None,1),slice(None,None,1),slice(None,None,1)]   
    
    fig = plt.figure()
    ims = []
    for i in range(img_array.shape[ani_dim]):
        sl_iter[ani_dim] = i
        if useArrayWideMax:
            ims.append([plt.imshow(img_array[sl_iter], cmap=cmap, interpolation='none', vmin=min_val, vmax=max_val, animated = True)])
        else:
            ims.append([plt.imshow(img_array[sl_iter], cmap=cmap, interpolation='none', animated = True)])
        
    
    im_ani = anim.ArtistAnimation(fig, ims, interval=interval, repeat_delay=repeat_delay, blit=True)
    
    return [fig, im_ani]
    
 
def imshow3D(img_array, img_area=slice(None,None,1),subpl_size=[5,5], plt_dim = 2, cmap='CMRmap', cutoff=1., cutoffRelative=True):
    #img_area gives the portion of the 3D object in the sliced dimension
    
    
    sl_sub = [slice(None,None,1),slice(None,None,1),slice(None,None,1)]
    sl_sub[plt_dim] = img_area
    
    img_shape = img_array[sl_sub].shape
    iter_size = img_shape[plt_dim]
    di = float(iter_size)/(subpl_size[0]*subpl_size[1])
    
    
    sl_iter = [slice(None,None,1),slice(None,None,1),slice(None,None,1)]   
    min_val = img_array.min()
    
    if cutoffRelative:
        max_val = cutoff*img_array.max()    
    else:
        max_val = cutoff
    
    fig = plt.figure()
    if subpl_size[0]*subpl_size[1] > img_array.shape[plt_dim]:
        for i in range(img_array.shape[plt_dim]):
            sl_iter[plt_dim] = i
            plt.subplot(subpl_size[0], subpl_size[1], i+1)
            plt.imshow(img_array[sl_sub][sl_iter], vmin=min_val, vmax=max_val, cmap=cmap, interpolation='none')
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])                    
    else:
        for i in range(subpl_size[0]*subpl_size[1]):
            
            sl_iter[plt_dim] = int((i*di))
            plt.subplot(subpl_size[0], subpl_size[1], i+1)
            plt.imshow(img_array[sl_sub][sl_iter], vmin=min_val, vmax=max_val, cmap=cmap, interpolation='none')
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])        
        
    plt.tight_layout(pad=1.09, h_pad=None, w_pad=None, rect=None)   
    return fig
       

def imshow3D_slice(img_array, cmap='CMRmap', center_coord = None, show_slice_lines = False):
    
    min_val = img_array.min()
    max_val = img_array.max()     
    
    if not center_coord:
        center_coord = [int(i/2) for i in img_array.shape]
    
    
    x_img = np.zeros((img_array.shape[1], img_array.shape[2]))
    x_img[:,:] = img_array[center_coord[0], :,:]
    
    y_img = np.zeros((img_array.shape[0], img_array.shape[2]))
    y_img[:,:] = img_array[:,center_coord[1], :]    
    
    z_img = np.zeros((img_array.shape[0], img_array.shape[1]))
    z_img[:,:] = img_array[:,:,center_coord[2]]
    
    if show_slice_lines:
        x_img[center_coord[1], :] = 0
        x_img[:, center_coord[2]] = 0
        y_img[center_coord[0], :] = 0
        y_img[:, center_coord[2]] = 0
        z_img[center_coord[0], :] = 0
        z_img[:, center_coord[1]] = 0
    
    plt.figure()
    plt.subplot(221)
    plt.imshow(x_img, cmap=cmap, vmin=min_val, vmax=max_val, interpolation='none')
    
    plt.subplot(222)
    plt.imshow(y_img, cmap=cmap, vmin=min_val, vmax=max_val, interpolation='none')    

    plt.subplot(223)
    plt.imshow(z_img, cmap=cmap, vmin=min_val, vmax=max_val, interpolation='none')





    

    

