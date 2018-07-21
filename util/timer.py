# -*- coding: utf-8 -*-
"""
Part of the pyconvolve framework for convolution and deconvolution. 
Author: Lukas Küpper, 2018
License: GPLv3
"""
import time


########################################################################
class Timer(object):
    """"""
    COUNT = 0
    #----------------------------------------------------------------------
    def __init__(self, name = None, outputWhenExit = True, acc = 4):
        """Constructor"""
        if name is None:
            self.name = Timer.COUNT
            Timer.COUNT += 1
        else: 
            self.name = name
        self.output = outputWhenExit
        self.acc = acc
        self.tstart = -1.
        self.tend = -1.
        self.telapsed = -1.
        self.output_str = 'Time elapsed for process {}: {} s'.format('{}', '{:.'+'{}'.format(acc)+'f}')
        
        
    def __enter__(self):
        self.tstart = time.time()
        
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tend = time.time()
        self.telapsed = self.tend - self.tstart
        
        if self.output:
            print('Time elapsed for process {}: {} s'.format(self.name, self.telapsed))
        
    