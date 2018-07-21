
import os
import pytiff
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

import util


path_base = "C:\\Users\\lukas\\Master FZJ\\PSF_recon\\\\"

path_unreg = os.path.join(path_base, 'unregistrated')
path_reg = os.path.join(path_base, 'registrated')

path_slides = ["slide1_1000", "slide2_500", "slide3_500", "slide4_100", "slide5_100"]
p_edges = ["w_edges", "wo_edges"]

in_paths = {pt+ed:os.path.join(path_unreg, pt, ed) for pt in path_slides for ed in p_edges}
out_paths = {pt+ed:os.path.join(path_reg, pt, ed) for pt in path_slides for ed in p_edges}

crop_fname = "slide{}_crop_{}x{}_{:0>3}.tif"

unreg_size = [552,552]
reg_size = [512,512]


def read_im_stack():
    ims = []
    n_min = 1
    n_max = 199    
    
    for i in range(n_min,n_max+1):
        print("Reading tif No. {}...".format(i))
        pt_ident = "slide1_1000wo_edges"
        sl_no = 1
        print("path: {}".format(os.path.join(out_paths[pt_ident], crop_fname.format(sl_no,reg_size[0], reg_size[1],i)),))
        with pytiff.Tiff(os.path.join(out_paths[pt_ident], crop_fname.format(sl_no,reg_size[0], reg_size[1],i))) as r_handle:
            part = r_handle[:,:]
        del r_handle
    
        ims.append(part)
        
    return ims

fig = plt.figure()

ims = read_im_stack()

ims = [[plt.imshow(i, cmap="gray", animated=True)] for i in ims]




im_ani = anim.ArtistAnimation(fig, ims, interval=50, repeat_delay=20,
                                   blit=True)

plt.show()