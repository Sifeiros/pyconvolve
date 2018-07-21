# -*- coding: utf-8 -*-
"""
Part of the pyconvolve framework for convolution and deconvolution. 
Author: Lukas Küpper, 2018
License: GPLv3
"""
import os
from os.path import join as ptjoin


def has_file_ending(file_ident, file_ending):
    """"""
    return file_ident[::-1].startswith(file_ending[::-1] + '.')


def removeFileEnding(file_name):
    """"""
    return file_name[::-1].split('.',1)[1][::-1]


def checkAllPaths(paths):
    if type(paths) is list:
            ret = True
            for pt in paths:
                ret = ret and os.path.isdir(pt)
            return True
    elif type(paths) is str:
        return os.path.isdir(paths)
    else:
        raise ValueError

def createAllPaths(paths):
    if type(paths) is list:
        for pt in paths:
            if not os.path.isdir(pt):
                os.makedirs(pt)
    elif type(paths) is str:
        if not os.path.isdir(paths):
            os.makedirs(paths)
    else:
        raise ValueError

def isEmptyDir(path):
    return not bool(os.listdir(path))


if os.environ['COMPUTERNAME'] == 'SEVERUS':
    print('Using Surface Book Path specifications.')
    #ROOT = 'C:\\Users\\lukas\\Master FZJ\\'
    ROOT = ptjoin('c:', os.sep, 'Users', 'lukas', 'Master FZJ')
    #PSF_RECON_RAW = 'E:\\FZJ Daten\\psf_calibration\\'
    PSF_RECON_RAW = ptjoin('e:', os.sep, 'FZJ Daten', 'psf_calibration')
    #CODE_PATH = 'C:\\Users\\lukas\\OneDrive\\Lukas\\Dokumente\\Masterarbeit FZJ\\ownCode\\github\\PyDeconvolution'
    CODE_PATH = ptjoin('c:', os.sep, 'Users', 'lukas', 'OneDrive', 'Lukas', 'Dokumente', 'Masterarbeit FZJ', 'ownCode', 'github', 'PyDeconvolution')
	
elif os.environ['COMPUTERNAME'] == 'SIRIUS':
    print('Using Desktop Path specifications.')
    #ROOT = 'O:\\Master FZJ\\'
    ROOT = ptjoin('o:', os.sep, 'Master FZJ')
    PSF_RECON_RAW = None
    #CODE_PATH = 'C:\\Users\\lukas_000\\OneDrive\\Lukas\\Dokumente\\Masterarbeit FZJ\\ownCode\\github\\PyDeconvolution'
    CODE_PATH = ptjoin('c:', os.sep, 'Users', 'lukas_000', 'OneDrive', 'Lukas', 'Dokumente', 'Masterarbeit FZJ', 'ownCode', 'github', 'PyDeconvolution')
    
    
    
DATA_GIT = ptjoin(CODE_PATH,"data")


MICRO_DATA = ptjoin(ROOT, 'MicroData', 'Sept_2017')
MICRO_RECON = ptjoin(ROOT, 'MicroData_recon', 'Sept_2017')
SIM_DATA = ptjoin(ROOT, 'SimData')
SIM_PSF = ptjoin(ROOT, 'SimPsf')
SIM_CONVOLUTED = ptjoin(ROOT, 'SimConv')
SIM_RECON = ptjoin(ROOT, 'SimRecon')
PSF_DATA = ptjoin(ROOT, 'PSF_data_Sept_2017')
PSF_DATA_CROP = ptjoin(ROOT, 'PSF_data_crops', 'Sept_2017')
PSF_RECON = ptjoin(ROOT, 'PSF_recon', 'Sept_2017')
DECON_LAB = ptjoin(ROOT, 'FijiTestData')


#Paths to declare:
    
    #-DATA_GIT: data path of git repo: should not differ depending on system
    #-MICRO_DATA data path of microscopy data to be deconvolved
    #-SIM_DATA data path of simulation data
    #-SIM_PSF data path of simulated PSFs
    #-PSF_RECON_RAW data path of raw data for PSF reconstruction
    #-PSF_RECON_CROP data path of cropped / registrated data for PSF reconstruction
    #-PSF_RECON data path of reconstructed PSF 
    #-DECON_LAB data path of testing data sets from Fiji