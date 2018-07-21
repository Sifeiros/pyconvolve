import os
import io
import numpy as np
import util
import pytiff
import time
from decon.abstract_decon import AbstractDecon
from decon.iterative import AbstractIterative
from sim_creator.convoluter import Convoluter
from sim_creator.convoluter import Noise_Adder
from sim_creator.sample_builder import Sim_Sample
from sim_creator.psf_generator import PSF_Generator

########################################################################
class MetaData(object):
    """
    General Image-Stack Data:
    -	file identifier    
    - 	stack path
    -	image type: psf, image, sample, truth
    -	image origin: synthetic, experiment or reconstruction
    -	stack size (X x Y x Z)
    -	maximum value after rescaling

    Experimental Properties:
    -	resolution
    -	is crop?
        -	original img path
        -	original img file identifier
        -	original image size
        -	original resolution
        -	coordinates of upper left crop corner
        -	slice skips of crop (to lower resolution)
    -	original experimental data
        -	sample ID
    	-	date of acquisition
        -	setup 
        -	additional comments

    Synthetic Properties:
    -	name of creation script
    -	parameters for creation
    - 	date of creation
    -	additional comments 

    Reconstruction Properties:
    -	Used Algorithm
    -	Algorithm Parameters
    -	comparison with Ground Truth?
    	-	path of ground Truth
        -	file identifier of ground truth
        -	difference to ground Truth (of final result)
    -	Time taken for reconstruction (overall)
    	- time for prepare
        - time for solve
    -	iterative algorithm?
    		-	iteration steps
        	-	evolution of error
                -	path of intermediate results
                -	f_ident of intermediate results
                -	average time per step
                -	times per step

    -	applied constraints
    -	Fourier Transform type
    -	Date of reconstruction
    -	path to raw image
    -	file identifier of raw image
    -	path to used sample/psf
    -	file identifier of used sample/psf
    - 	comment


    """

    TYPES = ['psf', 'image', 'sample', 'ground_truth']
    ORIGIN = ['experiment', 'synthetic', 'reconstruction', 'convolution']
    VER = 0.1 # First Version and layout. 10-10-2017
    FIELDS = ['f_ident', 'path', 'typ', 'orig', 'size', 'max_v', 'resolution', 'orig_path', 'orig_fident', 'orig_size', 'orig_resolution', 
              'crop_coord', 'crop_skip', 'sampleID', 'date', 'comment', 'algorithm', 'parameters', 'grtr_path', 'grtr_fident', 'grtr_error', 
              'timeOverall', 'timePrepare', 'timeSolve', 'iterSteps', 'errors', 'inter_result_path', 'inter_result_fident', 'avg_steptime', 
              'steptimes', 'constraints', 'usePpxFFT', 'date', 'raw_path', 'raw_fident', 'compl_path', 'compl_fident', 'comment']
    #----------------------------------------------------------------------
    def __init__(self, path = None, f_ident = None ):
        """Constructor"""

        self.f_ident = ''
        self.path = ''
        self.typ = ''
        self.orig = ''
        self.size = []
        self.max_v = 0.


        #if typ in MetaData.TYPES:
            #self.typ = typ
        #else:
            #print('typ has to be in {}. given:{}'.format(MetaData.TYPES, typ))
            #raise ValueError('typ has to be in {}. given:{}'.format(MetaData.TYPES, typ))

        if type(path) == str and type(f_ident) == str:
            if util.has_file_ending(f_ident, 'txt'):
                self.loadFromFile(util.ptjoin(path, f_ident))
            else:
                self.loadFromFile(util.ptjoin(path, util.removeFileEnding(f_ident.format('_META'))+'.txt'))
        elif type(path) == str:
            tmp = [util.ptjoin(path,f) for f in os.listdir(path) if os.path.isfile(util.ptjoin(path, f)) and util.has_file_ending(util.ptjoin(path, f), 'txt')]
            self.loadFromFile(tmp[0])

    #----------------------------------------------------------------------
    def set_meta_datum(self):
        """"""

    #----------------------------------------------------------------------
    def get_meta_datum(self):
        """"""

    #----------------------------------------------------------------------
    def toList(self):
        """"""

        meta_raw = []

        meta_raw.append('META DATA:')

        meta_raw.append('File Identifier:\t{}'.format(self.f_ident))
        meta_raw.append('Path:\t{}'.format(self.path))
        meta_raw.append('Type:\t{}'.format(self.typ))
        meta_raw.append('Origin:\t{}'.format(self.orig))
        meta_raw.append('Image Size:\t({0[0]}x{0[1]}x{0[2]})'.format(self.size))
        meta_raw.append('Data-Type:\t{}'.format(self.d_type))
        meta_raw.append('Max. Value:\t{}'.format(self.max_v))
        if self.orig == 'experiment':
            meta_raw.append('EXPERIMENTAL PROPERTIES:')
            meta_raw.append('Resolution:\t[{0[0]}x{0[1]}x{0[2]}]'.format(self.resolution))
            if self.isCrop:
                meta_raw.append('Is Crop:\tTRUE')
                meta_raw.append('Orig. Image Path:\t{}'.format(self.orig_path))
                meta_raw.append('Orig. Identifier:\t{}'.format(self.orig_fident))
                meta_raw.append('Orig. Image Size:\t({0[0]}x{0[1]}x{0[2]})'.format(self.orig_size))
                meta_raw.append('Orig. Resolution:\t({0[0]}x{0[1]}x{0[2]})'.format(self.orig_resolution))
                meta_raw.append('Crop Corner:\t({0[0]}x{0[1]}x{0[2]})'.format(self.crop_coord))
                meta_raw.append('Crop Skip:\t({0[0]}:{0[1]}:{0[2]})'.format(self.crop_skip))
            else:
                meta_raw.append('Is Crop:\tFALSE')
                meta_raw.append('Orig. Image Path:\t{}'.format('NA'))
                meta_raw.append('Orig. Identifier:\t{}'.format('NA'))
                meta_raw.append('Orig. Image Size:\t{}'.format('NA'))
                meta_raw.append('Orig. Resolution:\t{}'.format('NA'))
                meta_raw.append('Crop Corner:\t{}'.format('NA'))
                meta_raw.append('Crop Skip:\t{}'.format('NA')) 
            meta_raw.append('Sample ID:\t{}'.format(self.sampleID))
            meta_raw.append('Date:\t{}'.format(self.date))
            meta_raw.append('Setup:\t{}'.format(self.setup))
            meta_raw.append('Comment:\t{}'.format(self.comment))

        elif self.orig == 'synthetic' and self.typ == 'ground_truth':
            meta_raw.append('SYNTHETIC PROPERTIES:')
            meta_raw.append('Script Name:\t{}'.format(self.script_name))
            meta_raw.append('Resolution:\t({0[0]}x{0[1]}x{0[2]})'.format(self.resolution))
            meta_raw.append('Parameters:\t{}'.format(','.join(['{}:{}'.format(key,val) for key,val in self.parameters.items()])))
            meta_raw.append('Date:\t{}'.format(self.date))
            meta_raw.append('Comment:\t{}'.format(self.comment))
            
            
        elif self.orig == 'synthetic' and self.typ == 'psf':
            
            meta_raw.append('SYNTHETIC PROPERTIES:')
            meta_raw.append('PSF Model:\t{}'.format(self.psf_model))
            meta_raw.append('PSF Type:\t{}'.format(self.psf_type))
            meta_raw.append('Resolution:\t({0[0]}x{0[1]}x{0[2]})'.format(self.resolution))
            meta_raw.append('Parameters:\t{}'.format(','.join(['{}:{}'.format(key,val) for key,val in self.parameters.items()])))
            if self.noise_type:
                meta_raw.append('Noise Type:\t{}'.format(self.noise_type))
                meta_raw.append('Noise Parameters:\t{}'.format(','.join(['{}:{}'.format(key,val) for key,val in self.noise_params.items()])))
            else:
                meta_raw.append('Noise Type:\t{}'.format('no_noise'))
                meta_raw.append('Noise Parameters:\t{}'.format('No Parameters:0'))
            meta_raw.append('Date:\t{}'.format(self.date))
            meta_raw.append('Comment:\t{}'.format(self.comment))
            
        elif self.orig == 'convolution':
            meta_raw.append('CONVOLUTION PROPERTIES:')
            meta_raw.append('Convolution Method:\t{}'.format(self.conv_method))
            meta_raw.append('Sample Path:\t{}'.format(self.samp_path))
            meta_raw.append('Sample Fident:\t{}'.format(self.samp_fident))
            meta_raw.append('PSF Path:\t{}'.format(self.psf_path))
            meta_raw.append('PSF Fident:\t{}'.format(self.psf_fident))
            meta_raw.append('Noise Type:\t{}'.format(self.noise_type))
            meta_raw.append('Noise Parameters:\t{}'.format(','.join(['{}:{}'.format(key,val) for key,val in self.noise_params.items()])))
            meta_raw.append('Date:\t{}'.format(self.date))
            meta_raw.append('Comment:\t{}'.format(self.comment))
            
        elif self.orig == 'reconstruction':
            meta_raw.append('RECONSTRUCTION PROPERTIES:')
            meta_raw.append('Algorithm:\t{}'.format(self.algorithm))
            meta_raw.append('Algorithm Parameters:\t{}'.format(','.join(['{}:{}'.format(key,val) for key,val in self.parameters.items()])))
            if self.comparedWGrTr:
                meta_raw.append('Compared with GT:\t{}'.format('TRUE'))
                meta_raw.append('GT Path:\t{}'.format(self.grtr_path))
                meta_raw.append('GT Fident:\t{}'.format(self.grtr_fident))
                meta_raw.append('Error:\t{}'.format(self.grtr_error))
            else:
                meta_raw.append('Compared with GT:\t{}'.format('FALSE'))
                meta_raw.append('GT Path:\t{}'.format('NA'))
                meta_raw.append('GT Fident:\t{}'.format('NA'))
                meta_raw.append('Error:\t{}'.format('NA'))

            meta_raw.append('Reconstruction-Time:\t{:.4g}'.format(self.timeOverall))
            meta_raw.append('Pre-Processing-Time:\t{:.4g}'.format(self.timePrepare))
            meta_raw.append('Solving-Time:\t{:.4g}'.format(self.timeSolve))
            if self.isIterative:
                meta_raw.append('Iterative Reconstruction:\t{}'.format('TRUE'))
                meta_raw.append('Iteration Steps:\t{}'.format(self.iterSteps))
                meta_raw.append('Errors:\t{}'.format('['+','.join([str(err) for err in self.errors])+']'))
                meta_raw.append('Interm. Result Path:\t{}'.format(self.inter_result_path))
                meta_raw.append('Interm. Result Fident:\t{}'.format(self.inter_result_fident))
                meta_raw.append('Steptime Avg:\t{:.4g}'.format(self.avg_steptime))
                meta_raw.append('Steptimes:\t{}'.format('['+','.join(['{:.4g}'.format(tim) for tim in self.steptimes])+']'))
            else:
                meta_raw.append('Iterative Reconstruction:\t{}'.format('FALSE'))
                meta_raw.append('Iteration Steps:\t{}'.format('NA'))
                meta_raw.append('Errors:\t{}'.format('NA'))
                meta_raw.append('Interm. Result Path:\t{}'.format('NA'))
                meta_raw.append('Interm. Result Fident:\t{}'.format('NA'))
                meta_raw.append('Steptime Avg:\t{}'.format('NA'))
                meta_raw.append('Steptimes:\t{}'.format('NA'))
            str_constr = ';'.join([constr[0]+'['+','.join(['{}:{}'.format(key,val) for key, val in constr[1].items()])+']' for constr in self.constraints])
            meta_raw.append('Constraints:\t{}'.format(str_constr))
            if self.useCpxFFT:
                meta_raw.append('Fourier-Transform:\t{}'.format('REAL'))
            else:
                meta_raw.append('Fourier-Transform:\t{}'.format('CPX'))
            meta_raw.append('Date:\t{}'.format(self.date))
            meta_raw.append('RI Path:\t{}'.format(self.raw_path))
            meta_raw.append('RI Fident:\t{}'.format(self.raw_fident))
            meta_raw.append('Compl Path:\t{}'.format(self.compl_path))
            meta_raw.append('Compl Fident:\t{}'.format(self.compl_fident))
            meta_raw.append('Comment:\t{}'.format(self.comment))

        else:
            raise IOError        
        
        return meta_raw
        

    #----------------------------------------------------------------------
    def saveToFile(self, path):
        """"""
        meta_raw = self.toList()

        print(path)
        print(os.getcwd())
        with open(path, mode='w') as handler:
            for lin in meta_raw:
                handler.write(lin+'\n')


    #----------------------------------------------------------------------
    def getIntermediateMeta(self, reconAlgo, path):
        """"""

        meta_raw = []

        if isinstance(reconAlgo, AbstractIterative):


            meta_raw.append('META DATA:')

            meta_raw.append('File Identifier:\t{}'.format(reconAlgo.save_fident))
            meta_raw.append('Path:\t{}'.format(path))
            meta_raw.append('Type:\t{}'.format(reconAlgo.orig_type))
            meta_raw.append('Origin:\t{}'.format('intermediate'))
            meta_raw.append('Image Size:\t({0[0]}x{0[1]}x{0[2]})'.format(reconAlgo.curGuess.shape))
            meta_raw.append('Data-Type:\t{}'.format(reconAlgo.curGuess.dtype))
            meta_raw.append('Max. Value:\t{}'.format(reconAlgo.curGuess.max()))

            meta_raw.append('INTERMEDIATE PROPERTIES:')
            meta_raw.append('Algorithm:\t{}'.format(reconAlgo.algoName))
            meta_raw.append('Algorithm Parameters:\t{}'.format(','.join(['{}:{}'.format(key,val) for key,val in reconAlgo.getAlgoParameters().items()])))
            if reconAlgo.compareWithTruth:
                meta_raw.append('Compared with GT:\t{}'.format('TRUE'))
                meta_raw.append('GT Path:\t{}'.format(reconAlgo.orig_img_path))
                meta_raw.append('GT Fident:\t{}'.format(reconAlgo.orig_img_fident))
                meta_raw.append('Error:\t{}'.format(reconAlgo.curError))
            else:
                meta_raw.append('Compared with GT:\t{}'.format('FALSE'))
                meta_raw.append('GT Path:\t{}'.format('NA'))
                meta_raw.append('GT Fident:\t{}'.format('NA'))
                meta_raw.append('Error:\t{}'.format('NA'))
            meta_raw.append('Current Iteration Step:\t{}'.format(reconAlgo.curIter))
            meta_raw.append('Current-Step-Time:\t{}'.format(reconAlgo.timePerStep[-1]))

            #meta_raw.append('Algorithm Parameters:\t{}'.format(','.join(['{}:{}'.format(lis[0],lis[1]) for lis in reconAlgo.getConstraints()])))            
            str_constr = ';'.join([constr[0]+'['+','.join(['{}:{}'.format(key,val) for key, val in constr[1].items()])+']' for constr in reconAlgo.getConstraints()])
            #str_constr = ';'.join(['{}:{}'.format(lis[0],lis[1]) for lis in reconAlgo.getConstraints()])
            meta_raw.append('Constraints:\t{}'.format(str_constr))
            if reconAlgo.useCpxFFT:
                meta_raw.append('Fourier-Transform:\t{}'.format('REAL'))
            else:
                meta_raw.append('Fourier-Transform:\t{}'.format('CPX'))
            meta_raw.append('Date:\t{}'.format(time.strftime("%Y-%m-%d")))
            meta_raw.append('Comment:\t{}'.format(reconAlgo.comments))	
        return meta_raw



    #----------------------------------------------------------------------
    def loadFromFile(self, path):
        """"""

        if not os.path.isfile(path):
            raise ValueError('Path not valid: {}'.format(path))

        meta_raw = []

        with open(path, mode='r') as handler:
            for line in handler:
                meta_raw.append(line.strip())


        #First lines of file are the same, regardless of the stack type and origin
        #META DATA:
        meta_raw.pop(0)
        #File Identifier:\t        
        self.f_ident = meta_raw.pop(0).split('\t')[1]
        #Path:\t        
        self.path = meta_raw.pop(0).split('\t')[1]
        #Image Type:\t
        self.typ = meta_raw.pop(0).split('\t')[1]
        #Image Origin:\t        
        self.orig = meta_raw.pop(0).split('\t')[1]
        #Image Size:\t
        self.size = [int(st) for st in meta_raw.pop(0).split('\t')[1].strip('(').strip(')').split('x')]
        #Data-Type:
        self.d_type = meta_raw.pop(0).split('\t')[1]
        #Max. Value:\t
        self.max_v = float(meta_raw.pop(0).split('\t')[1])

        if not self.typ in MetaData.TYPES or not self.orig in MetaData.ORIGIN:
            raise IOError('type: {} origin:{}'.format(self.typ, self.orig))

        if self.orig == 'experiment':

            #EXPERIMENTAL PROPERTIES:
            t_line = meta_raw.pop(0)
            if not t_line == 'EXPERIMENTAL PROPERTIES:':
                raise IOError(t_line)
            #Resolution:
            self.resolution = [float(st) for st in meta_raw.pop(0).split('\t')[1].strip('[').strip(']').split('x')]
            #Is Crop:\tTRUE
            t_str = meta_raw.pop(0).split('\t')[1]
            if t_str == 'TRUE':
                self.isCrop = True
            elif t_str == 'FALSE':
                self.isCrop = False
            else:
                raise IOError(t_str)
            if not self.isCrop:
                for i in range(6): meta_raw.pop(0)
            else:
                #Orig. Image Path:\t   (NA if Is Crop:FALSE)
                self.orig_path = meta_raw.pop(0).split('\t')[1]
                #Orig. Identifier:\t   (NA if Is Crop:FALSE)
                self.orig_fident = meta_raw.pop(0).split('\t')[1]
                #Orig. Image Size:\t   (NA if Is Crop:FALSE)
                self.orig_size = [int(st) for st in meta_raw.pop(0).split('\t')[1].strip('(').strip(')').split('x')]
                #Orig. Resolution:\t	(NA if Is Crop:FALSE)
                self.orig_resolution = [float(st) for st in meta_raw.pop(0).split('\t')[1].strip('(').strip(')').split('x')]
                #Crop Corner:\t(XxYxZ) (NA if Is Crop:FALSE)
                self.crop_coord = [int(st) for st in meta_raw.pop(0).split('\t')[1].strip('(').strip(')').split('x')]
                #Crop Skip:\t(X:Y:Z)   (NA if Is Crop:FALSE)
                self.crop_skip = [int(st) for st in meta_raw.pop(0).split('\t')[1].strip('(').strip(')').split(':')]
            #Sample ID:\t
            self.sampleID = meta_raw.pop(0).split('\t')[1]
            #Date:\t

            self.date = meta_raw.pop(0).split('\t')[1]
            #Setup:\t
            self.setup = meta_raw.pop(0).split('\t')[1]
            #Comment:\t
            self.comment = meta_raw.pop(0).split('\t')[1]

        elif self.orig == 'synthetic' and self.typ == 'ground_truth':

            #SYNTHETIC PROPERTIES:
            t_line = meta_raw.pop(0)
            if not t_line == 'SYNTHETIC PROPERTIES:':
                raise IOError(t_line)  
            #Script Name:
            self.script_name = meta_raw.pop(0).split('\t')[1]
            
            #Resolution:
            self.resolution = [float(st) for st in meta_raw.pop(0).split('\t')[1].strip('(').strip(')').split('x')]            
            #Parameters:\tpara_name:para_val,para_name:para_val
            t_str = meta_raw.pop(0).split('\t')[1]
            self.parameters = {kv[0]:kv[1] for kv in [s.split(':') for s in t_str.split(',')]}
            #Date:
            self.date = meta_raw.pop(0).split('\t')[1]
            #Comment:
            self.comment = meta_raw.pop(0).split('\t')[1]

        elif self.orig == 'synthetic' and self.typ == 'psf':
            
            t_line = meta_raw.pop(0)
            if not t_line == 'SYNTHETIC PROPERTIES:':
                raise IOError(t_line)
            
            #PSF Model
            self.psf_model = meta_raw.pop(0).split('\t')[1]
            #PSF Type
            self.psf_type = meta_raw.pop(0).split('\t')[1]
            #Resolution
            self.resolution = [float(st) for st in meta_raw.pop(0).split('\t')[1].strip('(').strip(')').split('x')]
            #Parameters
            t_str = meta_raw.pop(0).split('\t')[1]
            self.parameters = {kv[0]:kv[1] for kv in [s.split(':') for s in t_str.split(',')]}
            t_str = meta_raw.pop(0)
            if t_str.startswith('Date'):
                #Date
                self.date = t_str.split('\t')[1]
                #Noise Type
                self.noise_type = 'no_noise'
                #Noise Parameters
                self.noise_params = {'No Parameters':0}
            elif t_str.startswith('Noise'):
                #Noise Type
                self.noise_type = t_str.split('\t')[1]
                #Noise Parameters
                t_str = meta_raw.pop(0).split('\t')[1]
                self.noise_params = {kv[0]:kv[1] for kv in [s.split(':') for s in t_str.split(',')]}
                #Date            
                self.date = meta_raw.pop(0).split('\t')[1]
            else:
                raise IOError
            #Comment
            self.comment = meta_raw.pop(0).split('\t')[1]


        elif self.orig == 'convolution':

            t_line = meta_raw.pop(0)
            if not t_line == 'CONVOLUTION PROPERTIES:':
                raise IOError(t_line)
            
            #Convolution Method
            self.conv_method = meta_raw.pop(0).split('\t')[1]
            #Sample Path
            self.samp_path = meta_raw.pop(0).split('\t')[1]
            #Sample Fident
            self.samp_fident = meta_raw.pop(0).split('\t')[1]
            #PSF Path
            self.psf_path = meta_raw.pop(0).split('\t')[1]
            #PSF Fident
            self.psf_fident = meta_raw.pop(0).split('\t')[1]
            #Noise Type
            self.noise_type = meta_raw.pop(0).split('\t')[1]
            #Noise Parameters
            t_str = meta_raw.pop(0).split('\t')[1]
            self.noise_params = {kv[0]:kv[1] for kv in [s.split(':') for s in t_str.split(',')]}
            
            #Date:
            self.date = meta_raw.pop(0).split('\t')[1]
            #Comment:
            self.comment = meta_raw.pop(0).split('\t')[1]

        elif self.orig == 'reconstruction':

            #RECONSTRUCTION PROPERTIES:

            t_line = meta_raw.pop(0)
            if not t_line == 'RECONSTRUCTION PROPERTIES:':
                raise IOError(t_line)
            #Used Algorithm:
            self.algorithm = meta_raw.pop(0).split('\t')
            #Algorithm Parameters:
            t_str = meta_raw.pop(0).split('\t')[1]
            self.parameters = {kv[0]:kv[1] for kv in [s.split(':') for s in t_str.split(',')]}
            #Compared with GT:

            t_str = meta_raw.pop(0).split('\t')[1]
            if t_str == 'TRUE':
                self.comparedWGrTr = True
            elif t_str == 'FALSE':
                self.comparedWGrTr = False
            else:
                raise IOError(t_str)

            if not self.comparedWGrTr:
                for i in range(3):
                    meta_raw.pop(0)
            else:
                #GT Path:
                self.grtr_path = meta_raw.pop(0).split('\t')[1]
                #GT Fident:                
                self.grtr_fident = meta_raw.pop(0).split('\t')[1]
                #Final Error:                
                self.grtr_error = float(meta_raw.pop(0).split('\t')[1])

            #Reconstruction-Time:            

            self.timeOverall = float(meta_raw.pop(0).split('\t')[1])
            #Pre-Processing-Time:
            self.timePrepare = float(meta_raw.pop(0).split('\t')[1])
            #Solving-Time:            
            self.timeSolve   = float(meta_raw.pop(0).split('\t')[1])
            #Iterative Reconstruction:
            t_str = meta_raw.pop(0).split('\t')[1]
            if t_str == 'TRUE':
                self.isIterative = True
            elif t_str == 'FALSE':
                self.isIterative = False
            else:
                raise IOError(t_str)

            if not self.isIterative:
                for i in range(6):
                    meta_raw.pop(0)
            else:
                #Iteration Steps:                
                self.iterSteps = int(meta_raw.pop(0).split('\t')[1])   
                #Errors:[err1,err2,err3,err4,err5,...,errN]                
                t_str = meta_raw.pop(0).split('\t')[1]                
                self.errors = [float(s.strip()) for s in t_str.strip('[').strip(']').split(',')]
                #Interm. Result Path:                
                self.inter_result_path = meta_raw.pop(0).split('\t')[1]
                #Interm. Result Fident:                
                self.inter_result_fident = meta_raw.pop(0).split('\t')[1]
                #Steptime Avg:                
                self.avg_steptime = float(meta_raw.pop(0).split('\t')[1])
                #Steptimes:[time1,time2,...,timeN]                
                t_str = meta_raw.pop(0).split('\t')[1]
                self.steptimes = [float(s.strip()) for s in t_str.strip('[').strip(']').split(',')]

            #Constraints:\t	constraint_name[name:val,name:val];constraint_name[name:val,name:val]            
            t_str = meta_raw.pop(0).split('\t')[1]
            self.constraints = [[c_str.split('[')[0], {kv[0]:float(kv[1]) for kv in [kv_str.split(':') for kv_str in c_str.split('[')[1].strip(']').split(',')]}] for c_str in t_str.split(';')]

            #Fourier-Transform:REAL/CPX            
            t_str = meta_raw.pop(0).split('\t')[1]
            if t_str == 'REAL':
                self.useCpxFFT = False
            elif t_str == 'CPX':
                self.useCpxFFT = True
            else:
                raise IOError(t_str)            

            #Date:
            self.date = meta_raw.pop(0).split('\t')[1]
            #RI Path:
            self.raw_path = meta_raw.pop(0).split('\t')[1]
            #RI Fident:            
            self.raw_fident = meta_raw.pop(0).split('\t')[1]
            #Compl Path:            
            self.compl_path = meta_raw.pop(0).split('\t')[1]
            #Compl Fident:            
            self.compl_fident = meta_raw.pop(0).split('\t')[1]
            #Comment:
            self.comment = meta_raw.pop(0).split('\t')[1]



    #----------------------------------------------------------------------
    def loadFromReconAlgo(self, reconAlgo):
        """"""

        if isinstance(reconAlgo, AbstractDecon):

            self.f_ident = reconAlgo.save_fident
            self.path = reconAlgo.save_path
            self.typ = reconAlgo.orig_type
            
            self.date = reconAlgo.timestring

            self.orig = 'reconstruction'
            self.size = reconAlgo.out.shape
            self.d_type = str(reconAlgo.out.dtype)
            self.max_v = reconAlgo.out.max()
            self.algorithm = reconAlgo.algoName
            self.parameters = reconAlgo.getAlgoParameters() #Format as {para_name:val, para_name:val}
            self.comparedWGrTr = reconAlgo.compareWithTruth

            if self.comparedWGrTr:
                self.grtr_path = reconAlgo.orig_truth_path
                self.grtr_fident = reconAlgo.orig_truth_fident
                self.grtr_error = reconAlgo.curError

            self.timeOverall = reconAlgo.timeOverall
            self.timePrepare = reconAlgo.timeToPrepare
            self.timeSolve = reconAlgo.timeToSolve

            self.isIterative = isinstance(reconAlgo, AbstractIterative)
            if self.isIterative:
                self.iterSteps = reconAlgo.curIter
                self.errors = reconAlgo.errors
                self.inter_result_path = reconAlgo.intermediate_path_add
                self.inter_result_fident = reconAlgo.save_fident
                self.avg_steptime = reconAlgo.avgTimeperStep
                self.steptimes = reconAlgo.timePerStep
            self.constraints = reconAlgo.getConstraints()
            self.useCpxFFT = reconAlgo.useCpxFFT
            self.comment  = ''
            self.raw_fident = reconAlgo.orig_img_fident
            self.raw_path = reconAlgo.orig_img_path
            if reconAlgo.solveFor == 'sample':
                self.typ = 'sample'
                self.compl_fident = reconAlgo.orig_psf_fident
                self.compl_path = reconAlgo.orig_psf_path
            elif reconAlgo.solveFor == 'psf':
                self.typ = 'psf'
                self.compl_fident = reconAlgo.orig_sample_fident
                self.compl_path = reconAlgo.orig_sample_path
        else:
            raise TypeError



    #----------------------------------------------------------------------
    def loadFromSimSample(self, sim_sample):
        """"""
        
        if isinstance(sim_sample, Sim_Sample):
            
            self.f_ident = sim_sample.save_fident
            self.path = sim_sample.save_path
            self.resolution = sim_sample.res
            self.typ = 'ground_truth'
            self.size = sim_sample.out.shape
            self.d_type = str(sim_sample.out.dtype)
            self.max_v = sim_sample.out.max()
            
            self.orig = 'synthetic'
            self.script_name = sim_sample.name
            self.parameters = {'Construct Type':sim_sample.img_type, 'random Orientation':sim_sample.randOrient, 'random Position':sim_sample.randPos}
            self.date = sim_sample.timestring
            self.comment = sim_sample.comments
        else:
            raise TypeError



    #----------------------------------------------------------------------
    def loadFromConvoluter(self, convoluter):
        """"""
        
        if isinstance(convoluter, Convoluter):
            
            self.f_ident = convoluter.save_fident
            self.path = convoluter.save_path
            self.typ = 'image'
            self.size = convoluter.out.shape
            self.d_type = str(convoluter.out.dtype)
            self.max_v = convoluter.out.max()

            self.date = convoluter.timestring
            
            self.orig = 'convolution'
            self.conv_method = convoluter.conv_method
            self.samp_path = convoluter.orig_img_path
            self.samp_fident = convoluter.orig_img_fident
            self.psf_path = convoluter.orig_psf_path
            self.psf_fident = convoluter.orig_psf_fident
            self.noise_type = Convoluter.NOISE_TYPES_INV[convoluter.noise_type]
            if convoluter.noise_type == -1:
                self.noise_params = {'No Parameters':0}
            else:
                self.noise_params = convoluter.noise_params
            self.comment = convoluter.comment
        
        else:
            raise TypeError

    #----------------------------------------------------------------------
    def loadFromNoiseAdder(self, noise_adder):
        """"""
        
        if isinstance(noise_adder, Noise_Adder):
            
            self.f_ident = noise_adder.save_fident
            self.path = noise_adder.save_path

            self.typ = noise_adder.img_type
            
            self.size = noise_adder.out.shape
            self.d_type = str(noise_adder.out.dtype)
            self.max_v = noise_adder.out.max()
            
            self.comment = noise_adder.comment
            if noise_adder.old_meta:
                self.orig = noise_adder.old_meta.orig
                if noise_adder.old_meta.orig == 'convolution':
                    self.date = noise_adder.old_meta.date
                    self.conv_method = noise_adder.old_meta.conv_method
                    self.samp_path = noise_adder.old_meta.samp_path
                    self.samp_fident = noise_adder.old_meta.samp_fident
                    self.psf_path = noise_adder.old_meta.psf_path
                    self.psf_fident = noise_adder.old_meta.psf_fident
                    self.noise_type = Noise_Adder.NOISE_TYPES_INV[noise_adder.noise_type]
                    self.noise_params = noise_adder.noise_params
                if noise_adder.old_meta.orig == 'synthetic' and noise_adder.old_meta.typ == 'psf':
                    self.psf_model = noise_adder.old_meta.psf_model
                    self.psf_type = noise_adder.old_meta.psf_type
                    self.resolution = noise_adder.old_meta.resolution
                    self.parameters = noise_adder.old_meta.parameters
                    
                    self.comment = noise_adder.old_meta.comment                    
                    
                    self.date = noise_adder.timestring
                    
                    self.noise_type = Noise_Adder.NOISE_TYPES_INV[noise_adder.noise_type]
                    self.noise_params = noise_adder.noise_params
            else:
                self.orig = 'synthetic'
                self.date = noise_adder.timestring
                self.samp_path = noise_adder.orig_img_path
                self.samp_fident = noise_adder.orig_img_fident
                self.noise_type = Noise_Adder.NOISE_TYPES_INV[noise_adder.noise_type]
                self.noise_params = noise_adder.noise_params
                
        else:
            raise TypeError
        
        
        
    #----------------------------------------------------------------------
    def loadFromPSFGen(self, psf_gen, PSF_Gen):
        """"""
        

        
        if isinstance(psf_gen, PSF_Generator):
        #if  True:
            
            self.f_ident = psf_gen.save_fident
            self.path = psf_gen.save_path
            self.typ = 'psf'
            self.orig = 'synthetic'
            self.size = psf_gen.out.shape
            self.d_type = str(psf_gen.out.dtype)
            self.max_v = psf_gen.out.max()
            
            self.noise_type = None
            
            self.psf_model = PSF_Generator.PSF_MODELS_REV[psf_gen.psf_model]
            self.psf_type = PSF_Generator.PSF_TYPES_REV[psf_gen.psf_type]
            self.resolution = psf_gen.res
            self.parameters = psf_gen.psf_params
            self.date = psf_gen.timestring
            self.comment = psf_gen.comment
        
        
        else:
            raise TypeError
        


    #----------------------------------------------------------------------
    def setFields(self, field_dict):
        """"""

        for key,val in field_dict.items():
            if key in MetaData.FIELDS:
                self.__dict__[key] = val
            else:
                raise ValueError('key {} not in FIELDS.'.format(key))





#----------------------------------------------------------------------
def read_image_stack(path, f_ident, n_min = -1, img_count = -1, meta=False):
    """"""

    if n_min == -1 or img_count == -1:
        t_min, t_count = _det_nmin_img_count(path, f_ident)
        if n_min == -1:
            n_min = t_min
        if img_count == -1:
            img_count = t_count
    
    temp_img = _readImage(path, f_ident, n_min)

    ret_array = np.zeros((temp_img.shape[0], temp_img.shape[1], img_count), dtype = temp_img.dtype)
    ret_array[:,:,0] = temp_img[:,:]

    print('Reading Image Stack: n_min={}, img_count={}'.format(n_min, img_count))
    print('File Identifier: {}, Stack Path: {}'.format(f_ident, path))

    for ind in range(1, img_count):

        temp_img = _readImage(path, f_ident, ind + n_min)

        ret_array[:,:,ind] = temp_img[:,:]

    if meta:
        f_name = util.removeFileEnding(f_ident.format('META')) +'.txt'
        meta_data = MetaData()
        meta_data.loadFromFile(util.ptjoin(path, f_name))
        
        if meta_data.max_v:
            ret_array = ((meta_data.max_v/ret_array.max()) * ret_array).astype(meta_data.d_type)
        
        return [ret_array, meta_data]
    else:
        return ret_array
    
    print('Done')
    
#----------------------------------------------------------------------
def _det_nmin_img_count(path, f_ident):
    """"""
    f_ident = util.removeFileEnding(f_ident)
    f_ident_beg, f_ident_end = f_ident.split('{')
    f_ident_end = f_ident_end.split('}')[1][::-1]
    
    file_listings = [fil for fil in os.listdir(path) if os.path.isfile(util.ptjoin(path, fil))]
    
    len_beg = len(f_ident_beg)
    len_end = len(f_ident_end)
    
    nums = []
    
    for fil in file_listings:
        fil = util.removeFileEnding(fil)
        
        if fil[:len_beg] == f_ident_beg and fil[-1:-1-len_end:-1] == f_ident_end:
            if len_end == 0:
                index = slice(len_beg, None, None)
            else:
                index = slice(len_beg,-len_end, None)
            t_str = fil[index]
            if t_str == 'META':
                continue
            else:
                nums.append(int(t_str))
    
    nums.sort()
    
    
    return [nums[0], len(nums)]
    
    
    

    
    
    
    


#----------------------------------------------------------------------
def read_meta_data_only(path, f_ident = None):
    """"""

    if f_ident is None:
        dir_content = os.listdir(path)
        dir_content = [f for f in dir_content if os.path.isfile(util.ptjoin(path, f)) and util.has_file_ending(f, 'txt')]
        f_name = dir_content[0]
    else:
        f_name = util.removeFileEnding(f_ident.format('META')) + '.txt'

    meta = MetaData()
    meta.loadFromFile(util.ptjoin(path, f_name))
    
    return meta

#----------------------------------------------------------------------
def _readImage(path, f_ident, ind):
    """"""

    
    #print("Path: {}; f_ident:{}, ind:{}".format(path, f_ident, ind))
    with pytiff.Tiff(os.path.join(path, f_ident.format(ind))) as r_handle:
        if r_handle.number_of_pages > 1:
            r_handle.set_page(0)
        #tags = r_handle.read_tags()
        array_w = r_handle[:,:]
    del r_handle        

    return array_w      

#----------------------------------------------------------------------
def _readMeta(path, f_ident):
    """"""

    f_name = util.removeFileEnding(f_ident.format('META')) +'.txt'

    if os.path.isfile(util.ptjoin(path, f_name)):
        meta = []
        with open(util.ptjoin(path, f_name)) as handler:
            for lin in handler:
                meta.append(lin)

        return meta
    else:
        return None

#----------------------------------------------------------------------
def write_image_stack(img_stack, path, f_ident, n_start, meta_data = None, meta_only = False):
    """"""

    util.createAllPaths(path)

    if not img_stack.dtype == 'uint8':
        raise TypeError
    
    if meta_data:
        if isinstance(meta_data, MetaData):
            meta_data = meta_data.toList()
        if not isinstance(meta_data, list):
            raise TypeError 
    
    if not meta_only:
        for ind in range(img_stack.shape[2]):
    
            temp_img = img_stack[:,:,ind].copy()
            _writeImage(temp_img, path, f_ident, ind)


    if meta_data:
        print('Writing Meta...')
        _writeMeta(meta_data, path, f_ident)

#----------------------------------------------------------------------
def _writeImage(img, path, f_ident, ind):
    """"""

    with pytiff.Tiff(os.path.join(path, f_ident.format(ind)), 'w') as w_handle:
        w_handle.write(img, method = "scanline")
        #w_handle.write(img)

    del w_handle     


#----------------------------------------------------------------------
def _writeMeta(meta, path, f_ident):
    """"""
    f_name = f_ident.format('META')

    f_name = util.removeFileEnding(f_name) + '.txt'

    with open(util.ptjoin(path, f_name), 'w') as handler:
        for lin in meta:
            handler.write(lin+'\n')
