import math as mt
import numpy as np
import numpy.random as np_rand

import util
import util.stack_loader

import datetime as dt



########################################################################
class Sim_Sample(object):
    """"""

    ADDITIVE = 0
    SUBSTRACTIVE = 1
    SET = 2
    
    IMG_CONSTRUCT = {'add':ADDITIVE, 'sub':SUBSTRACTIVE, 'set':SET}
    IMG_CONSTRUCT_REV = {v:k for k,v in IMG_CONSTRUCT.items()}

    #----------------------------------------------------------------------
    def __init__(self, size, resolution, background = 240., imageConstructType = 0, randomizeOrientation = 0., randomizePosition = 0., sampleParams = None, name = 'TestSample', comments = ['']):
        """Constructor"""
        
        
        self.name = name
        self.comments = comments
        self.timestring = ''
        self.meta = util.stack_loader.MetaData()
        
        
        if type(size) == int or type(size) == float:
            self.size = [float(size),float(size),float(size)]
        elif type(size) == np.ndarray and size.size == 3:
            self.size = np.zeros((3), dtype = 'float')
            self.size[:] = size[:]
        elif type(size) == list and len(size) == 3:
            size = [float(it) for it in size]
            self.size = np.array(size)        
        else:
            raise ValueError
            
        if type(resolution) == int or type(resolution) == float:
            self.res = [float(resolution), float(resolution), float(resolution)]
        elif type(resolution) == np.ndarray and resolution.size == 3:
            self.res = np.zeros((3))
            self.res[:] = resolution[:]
        elif type(resolution) == list and len(size) == 3:
            res = [float(it) for it in resolution ]
            self.res = np.array(res)
        else:
            raise ValueError        
        

        if imageConstructType in Sim_Sample.IMG_CONSTRUCT.values():
            self.img_type = imageConstructType

        self.shape = [1+2*int(siz/res) for siz,res in zip(self.size,self.res)]
        self.i_center = [int(siz/res) for siz,res in zip(self.size,self.res)]
        
        
        self.sampleParams = []
        
    
        if sampleParams:
            self.setSampleParams(sampleParams)
    
    
        self.background = background
        self.randOrient = randomizeOrientation
        self.randPos = randomizePosition
        
        self.save_fident = None
        self.save_path = None        
        
        self.out = np.ones(self.shape, dtype = 'float') * self.background 
        
        
        
    #----------------------------------------------------------------------
    def setSampleParams(self, sampleParams):
        """"""
        if not type(sampleParams) == list:
            raise ValueError
        
        
        for sample in sampleParams:
            
            self.addSampleParam(sample)
            
        
        
    #----------------------------------------------------------------------
    def addSampleParam(self, sampleParam):
        """"""
        
        if type(sampleParam) == list:
            if type(sampleParam[0]) == str and type(sampleParam[1]) == dict:
                sampleParam = [sampleParam]
            elif all(type(part) == list for part in sampleParam):
                pass
            else:
                raise TypeError
        else:
            raise TypeError
        
        
        for s_param in sampleParam:
            name = s_param[0]
            param_dict = s_param[1]
            
            
            if not name in SubStructure.TYPES:
                raise ValueError
            
            params = param_dict.keys()
            if 'signal' in params:
                if not (type(param_dict['signal']) == int or type(param_dict['signal']) == float):
                    raise ValueError
            else:
                raise ValueError
    
            if 'center' in params:
                t_val = param_dict['center']
                if ((type(t_val) == list or type(t_val) == tuple) and len(t_val) == 3) or (type(t_val) == np.ndarray and t_val.size == 3):
                    cen = []
                    for c in t_val:
                        cen.append(float(c))
                    param_dict['center'] = cen
                else:
                    raise ValueError
            else:
                raise ValueError
    
            if 'size' in params:
                t_val = param_dict['size']
                if ((type(t_val) == list or type(t_val) == tuple) and len(t_val) == 3) or (type(t_val) == np.ndarray and t_val.size == 3):
                    size = []
                    for s in t_val:
                        size.append(float(s))
                    param_dict['size'] = size
                else:
                    raise ValueError            
            else:
                raise ValueError
            
            if 'rotation' in params:
                t_val = param_dict['rotation']
                if ((type(t_val) == list or type(t_val) == tuple) and len(t_val) == 3) or (type(t_val) == np.ndarray and t_val.size == 3):
                    rot = []
                    for r in t_val:
                        rot.append(float(r))
                    param_dict['rotation'] = rot
                else:
                    raise ValueError   
            else:
                param_dict['rotation'] = [0.,0.,0.]
            
            
            if 'anisoDim' in params:
                t = int(param_dict['anisoDim'])
                if t <= 2 and t >= 0:
                    param_dict['anisoDim'] = t
            else:
                param_dict['anisoDim'] = 2
            
            param_dict['resolution'] = self.res
            param_dict['max_size'] = list(self.size)
            
            self.sampleParams.append([name,param_dict])
        
    #----------------------------------------------------------------------
    def coordinate_to_index(self, coord):
        """"""
        
        if any([coord[ind] > self.size[ind] or coord[ind] < -self.size[ind] for ind in range(3)]):
            print('Coord: {}, size: {}'.format(coord, self.size))
            raise ValueError
        
        return [self.i_center[ind] + int(round(coord[ind] / self.res[ind])) for ind in range(3)]
        
        
        
        
        

    #----------------------------------------------------------------------
    def index_to_coordinate(self, index):
        """"""
        
        if any([index[ind] < 0 or index[ind] > self.shape[ind] for ind in range(3)]):
            #print('Index: {}, shape: {}'.format(index, self.shape))
            raise ValueError
    
        return [ index[ind]*self.res[ind]-self.i_center[ind] for ind in range(3)]
    
    #----------------------------------------------------------------------
    def constructSample(self):
        """"""
        
        
        """
        Layout of entry in sample parameter:
        ['pyramid',{'signal':10., 'center':[0.,0.,0.], 'size':[2.,2.,5.], 'rotation':[np.pi,np.pi,np.pi], 'resolution':[0.1,0.1,0.1], anisoDim = 1}]
        
        """
        
        n_structs = len(self.sampleParams)
        n_cur = 1
        print('Starting Construction of Sample.')
        print('Sample-Size: {} with resolution: {}'.format(self.out.shape, self.res))
        print('Number of Substructures: {}'.format(n_structs))
        
        for struc in self.sampleParams:
            
            print('Building structure of type {}. \#{} of {}'.format(struc[0], n_cur, n_structs))
            
            t_struc = SubStructure.construct(struc[0], **struc[1])
            
            
            t_coord_struc = t_struc.center
            t_index_struc = self.coordinate_to_index(t_coord_struc)
            
            #Calculate the maximum size in index space (sample)
            #Compare this with the maximum size of the substructure
            
            ind_size_low = [min(t_index_struc[ind], (t_struc.safe_shape[ind]-1)/2) for ind in range(3)]
            ind_size_high = [min(self.shape[ind] - t_index_struc[ind] - 1, (t_struc.safe_shape[ind]-1)/2) for ind in range(3)]
            
            
            
            t_ind = t_struc.index[[slice(t_struc.i_center[ind]-ind_size_low[ind], t_struc.i_center[ind] + ind_size_high[ind] + 1, 1) for ind in range(3)]]
            #print(t_ind.shape)
            if self.img_type == Sim_Sample.ADDITIVE:
                self.out[[slice(t_index_struc[ind] - ind_size_low[ind], t_index_struc[ind] + ind_size_high[ind] + 1, 1) for ind in range(3)]][t_ind] += t_struc.signal
            elif self.img_type == Sim_Sample.SUBSTRACTIVE:
                self.out[[slice(t_index_struc[ind] - ind_size_low[ind], t_index_struc[ind] + ind_size_high[ind] + 1, 1) for ind in range(3)]][t_ind] -= t_struc.signal
            elif self.img_type == Sim_Sample.SET:
                self.out[[slice(t_index_struc[ind] - ind_size_low[ind], t_index_struc[ind] + ind_size_high[ind] + 1, 1) for ind in range(3)]][t_ind] = t_struc.signal
            else:
                raise ValueError
            
            del t_struc, t_ind
            n_cur += 1
            
        self.timestring = dt.datetime.now().strftime('%Y-%m-%d %H:%M')
            
    #----------------------------------------------------------------------
    def initSaveParameters(self, save_path, save_fident, overwrite = False):
        """"""
        
        self.save_fident = save_fident
        self.overwrite = overwrite
        
        util.createAllPaths(save_path)
        
        if not util.isEmptyDir(save_path) and not overwrite:
            raise ValueError('Save destination is not empty and overwriting is disabled.')
        else:
            self.save_path = save_path
            
            
            
            
    #----------------------------------------------------------------------
    def saveSolution(self):
        """"""
        self.meta.loadFromSimSample(self)
        
        meta = self.meta.toList()
        
        tmp = self.out.copy()
        max_v = tmp.max()
        tmp = tmp/max_v*255
        tmp = tmp.round()
        if tmp.max() > 255:
            raise ValueError
        tmp = tmp.astype('uint8')        
        
        util.stack_loader.write_image_stack(tmp, 
                                            self.save_path, 
                                            self.save_fident, 
                                            0, 
                                            meta_data=meta)    
            
       
            
            

        
        
########################################################################
class SubStructure(object):
    """"""


    PYRAMID = 'pyramid'
    RECTANGLE = 'rectangle'
    ELLIPSE = 'ellipse'
    CYLINDER = 'cylinder'
    ROUND_CYL = 'round_cylinder'    
    
    TYPES = [PYRAMID, RECTANGLE, ELLIPSE, CYLINDER, ROUND_CYL]

    #----------------------------------------------------------------------
    def __init__(self, signal, center, size, rotation, resolution, max_size = None):
        """Constructor"""
        #print('SubStructure Constructor')
        self.signal = signal
        
        
        if type(center) == np.ndarray and center.size == 3:
            self.center = np.zeros((3), dtype = 'float')
            self.center[:] = center[:]
        elif type(center) == list and len(center) == 3:
            center = [float(it) for it in center]
            self.center = np.array(center)
        else:
            raise ValueError
            
        if type(size) == int or type(size) == float:
            self.size = [float(size),float(size),float(size)]
        elif type(size) == np.ndarray and size.size == 3:
            self.size = np.zeros((3), dtype = 'float')
            self.size[:] = size[:]
        elif type(size) == list and len(size) == 3:
            size = [float(it) for it in size]
            self.size = np.array(size)        
        else:
            raise ValueError
            
        if type(resolution) == int or type(resolution) == float:
            self.res = [float(resolution), float(resolution), float(resolution)]
        elif type(resolution) == np.ndarray and resolution.size == 3:
            self.res = np.zeros((3))
            self.res[:] = resolution[:]
        elif type(resolution) == list and len(size) == 3:
            res = [float(it) for it in resolution ]
            self.res = np.array(res)
        else:
            raise ValueError
            
        if type(rotation) == np.ndarray and rotation.size == 3:
            self.rot = np.zeros((3), dtype = 'float')
            self.rot[:] = rotation[:]
        elif type(rotation) == list and len(rotation) == 3:
            rot = [float(it) for it in rotation]
            self.rot = np.array(rotation)          
        else:
            raise ValueError
        
        if not max_size is None:
            if type(max_size) == list and len(max_size) == 3:
                self.max_size = [float(ms) for ms in max_size]
            elif type(max_size) == float:
                self.max_size = [max_size for i in range(3)]
            else:
                print(max_size)
                raise TypeError
        else:
            self.max_size = None
            
        
        self.rotMat = self.rotationMatrix(self.rot[0], self.rot[1], self.rot[2])        
        self.safe_shape = []
        self.safe_size = []
        self._createSafeSize()
        self._checkAngleRange()
        
        self.i_center = [(self.safe_shape[ind]-1)/2 for ind in range(3)]
        #print('Determined:')
        #print('Safe-Size: {}'.format(self.safe_size))
        #print('Safe-Shape: {}'.format(self.safe_shape))
        #print('Resolution: {}'.format(self.res))
        #self.out = np.zeros(self.safe_shape,dtype = 'float')
        self.index = np.ones(self.safe_shape, dtype = 'bool')
    
    
    
    
    #----------------------------------------------------------------------
    def _createSafeSize(self):
        """"""
        raise NotImplementedError
        
            
    #----------------------------------------------------------------------
    def _checkAngleRange(self):
        """"""
        raise NotImplementedError
        
    
    #----------------------------------------------------------------------
    @staticmethod
    def construct(typ, **kwargs):
        """"""
        #print(kwargs)
        if typ == 'pyramid':
            t_struc = Pyramid(**kwargs)
        elif typ == 'rectangle':
            t_struc = Rectangle(**kwargs)
        elif typ == 'ellipse':
            t_struc = Ellipse(**kwargs)
        elif typ == 'cylinder':
            t_struc = Cylinder(**kwargs)
        elif typ == 'round_cylinder':
            t_struc = RoundedCylinder(**kwargs)
        else:
            raise ValueError
        
        return t_struc
        
        
        
        
        
        
    #----------------------------------------------------------------------
    def rotationMatrix(self, x_ang, y_ang, z_ang):
        """"""
        
        M1 = np.array([[1.,              0.,               0.],
                       [0.,              np.cos(x_ang),    np.sin(x_ang)],
                       [0.,             -np.sin(x_ang),    np.cos(x_ang)]])
    
        M2 = np.array([[np.cos(y_ang),     0.,               -np.sin(y_ang)],
                       [0.,                1.,               0.],
                       [np.sin(y_ang),     0.,               np.cos(y_ang)]])        
    
        M3 = np.array([[np.cos(z_ang),   np.sin(z_ang),  0.],
                       [-np.sin(z_ang),  np.cos(z_ang),  0.],
                       [0.,              0.,             1.]])                
    
        rot = np.dot(np.dot(M1, M2), M3)        
        return rot
    
    
    
########################################################################
class Pyramid(SubStructure):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, signal, center, size, rotation, resolution, anisoDim = -1, max_size = None):
        """Constructor
        
        Creates an anisotropic pyramid with an equilateral triangle as the base. Two of the three size-parameters
        have to be equal. These parameters define the dimensions of the equilateral triangle. The third parameters
        describes the size in the anisotropic dimension. 
        Assuming the equilateral triangle is in the 0th and 1st dimension, the center[0] and center[1] give the 
        coordinates of the center of mass of the base triangle. center[2] gives the coordinate of the half distance
        between base and top. 
        
        signal = value inside the rectangle
        center = coordinates of the ellipsoidal center [x0,y0,z0]
        size = in isotropic dimensions: side-lenght of the triangle, in anisotropic dimension: lenght from base to top
        rotation = rotation about the [x,y,z] axes in units of pi. allowed values range from -pi/2 to pi/2
        resolution = unit distance between two pixels in x,y,z direction
        
        center size and resolution have to be given in the same units of measurement.            
        
        """
        
        super(Pyramid, self).__init__(signal, center, size, rotation, resolution, max_size)
        
        
        
        if self.size[0] == self.size[1]:
            self.anisoDim = 2
        elif self.size[0] == self.size[2]:
            self.anisoDim = 1
        elif self.size[1] == self.size[2]:
            self.anisoDim = 0
        else:
            raise ValueError
        #print('Anisotropic Dimension: {}'.format(self.anisoDim))
        self._createHessForms()
        self._fill_output()
        
        
        
    #----------------------------------------------------------------------
    def _checkAngleRange(self):
        """"""
        pi2 = np.pi/2j
        
        for i in range(3):
            if self.rot[i] > np.pi:
                self.rot[i] -= 2*np.pi
            if self.rot[i] < -np.pi:
                self.rot[i] += 2*np.pi
        
    #----------------------------------------------------------------------
    def _createSafeSize(self):
        """"""
        if all([r==0. for r in self.rot]):
            self.safe_size = [self.res[ind]*(int(self.size[ind]/self.res[ind])+1) for ind in range(3)]
        else:        
            self.safe_size = [re*int(np.sqrt(2)/re * self.size.max()) for re in self.res]
        
        if not self.max_size is None:
            self.safe_size = [min(self.safe_size[ind], self.max_size[ind]) for ind in range(3)]
        self.safe_shape = [1+2*int(self.safe_size[ind]/self.res[ind]) for ind in range(3)]        
        
    #----------------------------------------------------------------------
    def _createHessForms(self):
        """"""
        
        hess_vect = np.zeros((4,4))
        
        
        
        h = self.size[(self.anisoDim)]
        a = self.size[(self.anisoDim+1)%3]
        q = mt.sqrt(12.*h**2 + a**2)
        
        #print('h = {}'.format(h))
        #print('a = {}'.format(a))
        #print('q = {}'.format(q))
        
        
        d0 = h*0.25
        d1 = a*h/q
        
        n_aniso = a/q
        
        #print('d0 = {}'.format(d0))
        #print('d1 = {}'.format(d1))
        #print('n_an = {}'.format(n_aniso))        
        
        hess_vect[0,3] = d0
        hess_vect[1,3] = d1
        hess_vect[2,3] = d1
        hess_vect[3,3] = d1
        
        hess_vect[0,self.anisoDim] = -1.
        hess_vect[1:4,self.anisoDim] = n_aniso
        
        hess_vect[1,(self.anisoDim+1) % 3] = -mt.sqrt(12)*h / q
        hess_vect[1,(self.anisoDim+2) % 3] = 0.
        
        hess_vect[2,(self.anisoDim+2) % 3] = 3.*h / q
        hess_vect[2,(self.anisoDim+1) % 3] = mt.sqrt(3)*h /q 

        hess_vect[3,(self.anisoDim+2) % 3] = -3.*h / q
        hess_vect[3,(self.anisoDim+1) % 3] = mt.sqrt(3)*h / q 
        
        
        #print('rotation Matrix:')
        #print(self.rot)
        #print(self.rotMat)
        
        for i in range(4):
            #print('nx: {}, ny: {}, nz: {}, |n|: {}'.format(hess_vect[i][0], hess_vect[i][1], hess_vect[i][2], hess_vect[i][0]**2+hess_vect[i][1]**2+hess_vect[i][2]**2))
            hess_vect[i,0:3] = np.dot(self.rotMat, hess_vect[i, 0:3])
            #print('nx: {}, ny: {}, nz: {}, |n|: {}'.format(hess_vect[i][0], hess_vect[i][1], hess_vect[i][2], hess_vect[i][0]**2+hess_vect[i][1]**2+hess_vect[i][2]**2))
        
        self.hess_vect = hess_vect
        
        
        
    #----------------------------------------------------------------------
    def _fill_output(self):
        """"""
        #ranges = [np.arange(-self.safe_size[i], -self.safe_size[i]+self.safe_shape[i]*self.res[i], self.res[i]) for i in range(3)]
        ranges = [np.linspace(-self.safe_size[i], self.safe_size[i], self.safe_shape[i]) for i in range(3)]
        X,Y,Z = np.meshgrid(*ranges, indexing='ij')
        
        index = self.index
        norm_vec = np.zeros((3))
        
        
        
        for i in range(4):
            norm_vec[:] = self.hess_vect[i,0:3]
            d = self.hess_vect[i,3]
            index = np.logical_and(index, X*norm_vec[0]+Y*norm_vec[1]+Z*norm_vec[2] <= d)
        
        #self.out[index] = self.signal
        del X,Y,Z, ranges
        self.index = index
                
        
        
########################################################################
class Ellipse(SubStructure):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, signal, center, size, rotation, resolution, anisoDim = -1, max_size = None):
        """Constructor
        
        signal = value inside the rectangle
        center = coordinates of the ellipsoidal center [x0,y0,z0]
        size = semi-axes in orientation to x,y,z axes before rotation
        rotation = rotation about the [x,y,z] axes in units of pi. allowed values range from -pi/2 to pi/2
        resolution = unit distance between two pixels in x,y,z direction
        
        center size and resolution have to be given in the same units of measurement.         
        
        """
        
        
        super(Ellipse, self).__init__(signal, center, size, rotation, resolution, max_size)
        
        self._fill_output()
        
        
    
    #----------------------------------------------------------------------
    def _createSafeSize(self):
        """"""
        if all([r==0. for r in self.rot]):
            self.safe_size = [self.res[ind]*(int(self.size[ind]/self.res[ind])+1) for ind in range(3)]
        else:
            self.safe_size = [re*int(np.sqrt(2)/re * self.size.max()) for re in self.res]
        if not self.max_size is None:
            self.safe_size = [min(self.safe_size[ind], self.max_size[ind]) for ind in range(3)]
        self.safe_shape = [1+2*int(self.safe_size[ind]/self.res[ind]) for ind in range(3)]        
        
    #----------------------------------------------------------------------
    def _checkAngleRange(self):
        """"""
        pi2 = np.pi/2j
        
        for i in range(3):
            if self.rot[i] > pi2:
                self.rot[i] -= np.pi
            if self.rot[i] < -pi2:
                self.rot[i] += np.pi
    
        
    #----------------------------------------------------------------------
    def _fill_output(self):
        """"""
        
        #x_ang,y_ang,z_ang = self.rot
        
        
        #M1_trans = np.array([[1.,              0.,               0.],
                             #[0.,              np.cos(x_ang),    -np.sin(x_ang)],
                             #[0.,              np.sin(x_ang),    np.cos(x_ang)]])
    
        #M2_trans = np.array([[np.cos(y_ang),     0.,               np.sin(y_ang)],
                             #[0.,                1.,               0.],
                             #[-np.sin(y_ang),    0.,               np.cos(y_ang)]])        
    
        #M3_trans = np.array([[np.cos(z_ang),   -np.sin(z_ang), 0.],
                             #[np.sin(z_ang),   np.cos(z_ang),  0.],
                             #[0.,              0.,             1.]])            
        
        #M_trans = np.dot(M3_trans,np.dot(M2_trans,M1_trans))
        M = self.rotMat
        #ranges = [np.arange(-self.safe_size[i], -self.safe_size[i]+self.safe_shape[i]*self.res[i], self.res[i]) for i in range(3)]
        ranges = [np.linspace(-self.safe_size[i], self.safe_size[i], self.safe_shape[i]) for i in range(3)]
        
        
        X0,Y0,Z0 = np.meshgrid(*ranges, indexing='ij')        
        
        
        #XT = M_trans[0,0] * X0 + M_trans[1,0]* Y0 + M_trans[2,0] * Z0
        #YT = M_trans[0,1] * X0 + M_trans[1,1]* Y0 + M_trans[2,1] * Z0
        #ZT = M_trans[0,2] * X0 + M_trans[1,2]* Y0 + M_trans[2,2] * Z0
        
        XE = (M[0,0] * X0 + M[0,1] * Y0 + M[0,2] * Z0)/self.size[0]
        YE = (M[1,0] * X0 + M[1,1] * Y0 + M[1,2] * Z0)/self.size[1]
        ZE = (M[2,0] * X0 + M[2,1] * Y0 + M[2,2] * Z0)/self.size[2]
        
        #index = (XT*XE + YT*YE + ZT*ZE <= 1.)
        index = (XE*XE + YE*YE + ZE*ZE <= 1.)
        #index = (XT*XT + YT*YT + ZT*ZT <= 1.)
        
        #self.out[index] = self.signal
        del X0,Y0,Z0, XE, YE, ZE, ranges, M
        
        self.index = index
        
        
    
########################################################################
class Rectangle(SubStructure):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, signal, center, size, rotation, resolution, anisoDim = -1, max_size = None):
        """Constructor
        
        signal = value inside the rectangle
        center = coordinates of the rectangle center [x0,y0,z0]
        size = length of the rectangle in x,y,z direction before rotation 
        rotation = rotation about the [x,y,z] axes in units of pi. allowed values range from -pi/2 to pi/2
        resolution = unit distance between to pixels in x,y,z direction
        
        center size and resolution have to be given in the same units of measurement. 
        
        
        """
        #max_size = None
        super(Rectangle, self).__init__(signal, center, size, rotation, resolution, max_size)
        
        
        self._createHessForms()
        self._fill_output()
        
        
        
    #----------------------------------------------------------------------
    def _createSafeSize(self):
        """"""
        if all([r==0. for r in self.rot]):
            self.safe_size = [self.res[ind]*(int(self.size[ind]/self.res[ind])+1) for ind in range(3)]
            #print('no rot')
        else:        
            self.safe_size = [re*int(np.sqrt(2)/re * self.size.max()) for re in self.res]
        
        if not self.max_size is None:
            self.safe_size = [min(self.safe_size[ind], self.max_size[ind]) for ind in range(3)]
        self.safe_shape = [1+2*int(self.safe_size[ind]/self.res[ind]) for ind in range(3)]        
        
    #----------------------------------------------------------------------
    def _checkAngleRange(self):
        """"""
        pi2 = np.pi/2j
        
        for i in range(3):
            if self.rot[i] > pi2:
                self.rot[i] -= np.pi
            if self.rot[i] < -pi2:
                self.rot[i] += np.pi
            
        
        
        
        
        
    #----------------------------------------------------------------------
    def _createHessForms(self):
        """"""
        
        hess_vect = np.zeros((6,4))
        
        hess_vect[0,0] = 1.
        hess_vect[1,1] = 1.
        hess_vect[2,2] = 1.
        hess_vect[3,0] = -1.
        hess_vect[4,1] = -1.
        hess_vect[5,2] = -1.        
        
        #hess_vect[0,3] = self.size[0]/2.
        #hess_vect[1,3] = self.size[1]/2.
        #hess_vect[2,3] = self.size[2]/2.
        #hess_vect[3,3] = self.size[0]/2.
        #hess_vect[4,3] = self.size[1]/2.
        #hess_vect[5,3] = self.size[2]/2.
        
        hess_vect[0,3] = self.size[0]
        hess_vect[1,3] = self.size[1]
        hess_vect[2,3] = self.size[2]
        hess_vect[3,3] = self.size[0]
        hess_vect[4,3] = self.size[1]
        hess_vect[5,3] = self.size[2]
                
        
        
        for i in range(6):
            hess_vect[i,0:3] = np.dot(self.rotMat, hess_vect[i, 0:3])
        
        self.hess_vect = hess_vect
        
        
        
    #----------------------------------------------------------------------
    def _fill_output(self):
        """"""
        #ranges = [np.arange(-self.safe_size[i], -self.safe_size[i]+self.safe_shape[i]*self.res[i], self.res[i]) for i in range(3)]
        ranges = [np.linspace(-self.safe_size[i], self.safe_size[i], self.safe_shape[i]) for i in range(3)]
        X,Y,Z = np.meshgrid(*ranges, indexing='ij')
        
        #print(X.shape)
        #print(ranges[0])
        index = self.index
        norm_vec = np.zeros((3))
        
        for i in range(6):
            norm_vec[:] = self.hess_vect[i,0:3]
            d = self.hess_vect[i,3]
            index = np.logical_and(index, X*norm_vec[0]+Y*norm_vec[1]+Z*norm_vec[2] <= d)
        
        #self.out[index] = self.signal
        del X,Y,Z, ranges
        self.index = index
        
        
    
########################################################################
class Cylinder(SubStructure):
    """"""

    #----------------------------------------------------------------------
    def __init__(self, signal, center, size, rotation, resolution, anisoDim = 2, max_size = None):
        """Constructor"""
        
        super(Cylinder, self).__init__(signal, center, size, rotation, resolution, max_size)
        
        self.anisoDim = anisoDim
        self._fill_output()
        
    #----------------------------------------------------------------------    
    def _createSafeSize(self):
        """"""
        
        if all([r==0. for r in self.rot]):
            self.safe_size = [self.res[ind]*(int(self.size[ind]/self.res[ind])+1) for ind in range(3)]
        else:
            self.safe_size = [re*int(np.sqrt(2)/re * self.size.max()) for re in self.res]
        if not self.max_size is None:
            self.safe_size = [min(self.safe_size[ind], self.max_size[ind]) for ind in range(3)]
        self.safe_shape = [1+2*int(self.safe_size[ind]/self.res[ind]) for ind in range(3)]
        
    #----------------------------------------------------------------------
    def _checkAngleRange(self):
        """"""
        pi2 = np.pi/2j
        
        for i in range(3):
            if self.rot[i] > pi2:
                self.rot[i] -= np.pi
            if self.rot[i] < -pi2:
                self.rot[i] += np.pi
        
        
    #----------------------------------------------------------------------
    def _fill_output(self):
        """"""

        #x_ang, y_ang, z_ang = self.rot
        #M1_trans = np.array([[1.,              0.,               0.],
                             #[0.,              np.cos(x_ang),    -np.sin(x_ang)],
                             #[0.,              np.sin(x_ang),    np.cos(x_ang)]])
    
        #M2_trans = np.array([[np.cos(y_ang),     0.,               np.sin(y_ang)],
                             #[0.,                1.,               0.],
                             #[-np.sin(y_ang),    0.,               np.cos(y_ang)]])        
    
        #M3_trans = np.array([[np.cos(z_ang),   -np.sin(z_ang), 0.],
                             #[np.sin(z_ang),   np.cos(z_ang),  0.],
                             #[0.,              0.,             1.]])            
        
        #M_trans = np.dot(M3_trans,np.dot(M2_trans,M1_trans))
        M = self.rotMat
        #ranges = [np.arange(-self.safe_size[i], -self.safe_size[i]+self.safe_shape[i]*self.res[i], self.res[i]) for i in range(3)]
        ranges = [np.linspace(-self.safe_size[i], self.safe_size[i], self.safe_shape[i]) for i in range(3)]
        X0,Y0,Z0 = np.meshgrid(*ranges, indexing='ij')        
        
        
        #print(M_trans)
        #print(self.rotMat)
        
        #XT = M_trans[0,0] * X0 + M_trans[1,0]* Y0 + M_trans[2,0] * Z0
        #YT = M_trans[0,1] * X0 + M_trans[1,1]* Y0 + M_trans[2,1] * Z0
        #ZT = M_trans[0,2] * X0 + M_trans[1,2]* Y0 + M_trans[2,2] * Z0
        
        XE = (M[0,0] * X0 + M[0,1] * Y0 + M[0,2] * Z0)
        YE = (M[1,0] * X0 + M[1,1] * Y0 + M[1,2] * Z0)
        ZE = (M[2,0] * X0 + M[2,1] * Y0 + M[2,2] * Z0)
        
        if self.anisoDim == 0:
            index = (YE*YE/self.size[1]**2 + ZE*ZE/self.size[2]**2 <= 1)
            index = np.logical_and(index, np.logical_and(XE <= self.size[0], XE >= -self.size[0]))
        elif self.anisoDim == 1:
            index = (XE*XE/self.size[0]**2 + ZE*ZE/self.size[2]**2 <= 1)
            index = np.logical_and(index, np.logical_and(YE <= self.size[1], YE >= -self.size[1]))
        else:
            index = (YE*YE/self.size[1]**2 + XE*XE/self.size[0]**2 <= 1)
            index = np.logical_and(index, np.logical_and(ZE <= self.size[2], ZE >= -self.size[2]))
        
        del XE,YE,ZE, ranges, M
        #self.out[index] = self.signal
        self.index = index
        
    
########################################################################
class RoundedCylinder(Cylinder):
    """"""



    #----------------------------------------------------------------------
    def __init__(self, signal, center, size, rotation, resolution, anisoDim = 2, max_size = None):
        """Constructor"""
        
        super(RoundedCylinder, self).__init__(signal, center, size, rotation, resolution, anisoDim, max_size)
    
    
    
    #----------------------------------------------------------------------    
    def _createSafeSize(self):
        """"""
        if all([r==0. for r in self.rot]):
            self.safe_size = [self.res[ind]*(int(self.size[ind]/self.res[ind]*1.4)) for ind in range(3)]
        else:        
            self.safe_size = [re*int(2./re * self.size.max()) for re in self.res]        
        if not self.max_size is None:
            self.safe_size = [min(self.safe_size[ind], self.max_size[ind]) for ind in range(3)]
        self.safe_shape = [1+2*int(self.safe_size[ind]/self.res[ind]) for ind in range(3)]        
    
    
    #----------------------------------------------------------------------
    def _fill_output(self):
        """"""

        #x_ang, y_ang, z_ang = self.rot
        #M1_trans = np.array([[1.,              0.,               0.],
                             #[0.,              np.cos(x_ang),    -np.sin(x_ang)],
                             #[0.,              np.sin(x_ang),    np.cos(x_ang)]])
    
        #M2_trans = np.array([[np.cos(y_ang),     0.,               np.sin(y_ang)],
                             #[0.,                1.,               0.],
                             #[-np.sin(y_ang),    0.,               np.cos(y_ang)]])        
    
        #M3_trans = np.array([[np.cos(z_ang),   -np.sin(z_ang), 0.],
                             #[np.sin(z_ang),   np.cos(z_ang),  0.],
                             #[0.,              0.,             1.]])            
        
        #M_trans = np.dot(M3_trans,np.dot(M2_trans,M1_trans))
        M = self.rotMat
        #ranges = [np.arange(-self.safe_size[i], -self.safe_size[i]+self.safe_shape[i]*self.res[i], self.res[i]) for i in range(3)]
        ranges = [np.linspace(-self.safe_size[i], self.safe_size[i], self.safe_shape[i]) for i in range(3)]
        X0,Y0,Z0 = np.meshgrid(*ranges, indexing='ij')        
        
        
   
        
        #XT = M_trans[0,0] * X0 + M_trans[1,0]* Y0 + M_trans[2,0] * Z0
        #YT = M_trans[0,1] * X0 + M_trans[1,1]* Y0 + M_trans[2,1] * Z0
        #ZT = M_trans[0,2] * X0 + M_trans[1,2]* Y0 + M_trans[2,2] * Z0
        
        XE = (M[0,0] * X0 + M[0,1] * Y0 + M[0,2] * Z0)
        YE = (M[1,0] * X0 + M[1,1] * Y0 + M[1,2] * Z0)
        ZE = (M[2,0] * X0 + M[2,1] * Y0 + M[2,2] * Z0)
        
        ell_prop = np.zeros((3,3))
        ell_prop[0,self.anisoDim] = self.size[self.anisoDim]
        ell_prop[1,self.anisoDim] = -self.size[self.anisoDim]
        
  

        ell_prop[2,:] = self.size[:]**2
        ell_prop[2,self.anisoDim] = 0.25*(self.size[(self.anisoDim+1)%3]**2 + self.size[(self.anisoDim+2)%3]**2)
        
        
        index = self.index
        if self.anisoDim == 0:
            index = ((YE/self.size[1])**2 + (ZE/self.size[2])**2 <= 1)
            index = np.logical_and(index, np.logical_and(XE <= self.size[0], XE >= -self.size[0]))

        elif self.anisoDim == 1:
            index = ((XE/self.size[0])**2 + (ZE/self.size[2])**2 <= 1)
            index = np.logical_and(index, np.logical_and(YE <= self.size[1], YE >= -self.size[1]))
        else:
            index = ((YE/self.size[1])**2 + (XE/self.size[0])**2 <= 1)
            index = np.logical_and(index, np.logical_and(ZE <= self.size[2], ZE >= -self.size[2]))


        
        index = np.logical_or(index, ((XE-ell_prop[0,0])**2/ell_prop[2,0] + (YE-ell_prop[0,1])**2/ell_prop[2,1] + (ZE-ell_prop[0,2])**2/ell_prop[2,2] <= 1))
        index = np.logical_or(index, ((XE-ell_prop[1,0])**2/ell_prop[2,0] + (YE-ell_prop[1,1])**2/ell_prop[2,1] + (ZE-ell_prop[1,2])**2/ell_prop[2,2] <= 1))


        del XE,YE,ZE,X0,Y0,Z0, ranges, M, ell_prop
        #self.out[index] = self.signal
        self.index = index       
  
        
    
    
    
#----------------------------------------------------------------------
def main_substructureTest():
    """"""
    signal = 20.
    size = [2.,2.,5.]
    resolution = [0.2, 0.2, 0.2]
    
    no_rot = [0.,0.,0.]
    rotation1 = [-.01, 0. , 0.]    
    rotation2 = [mt.pi/4.,0, mt.pi/4.]    
    rotation3 = [mt.pi/4.,mt.pi/4.,mt.pi/4.]
    
    
    #rect = Rectangle(signal, [0.,0.,0.], size, rotation2, resolution)
    #pyr = Pyramid(signal, [0.,0.,0], size, no_rot, resolution)
    #ell = Ellipse(signal, [0.,0.,0.], size, rotation3, resolution)
    #cyl = Cylinder(signal, [0.,0.,0.], size, rotation3, resolution, anisoDim=2)
    #round_cyl = RoundedCylinder(signal, [0,0,0], size, no_rot, resolution, anisoDim=2)
    
    round_cyl = SubStructure.construct(typ = SubStructure.CYLINDER, signal = signal, center = [0,0,0], size = size, rotation = rotation1, resolution = resolution, anisoDim = 2)
    round_cyl2 = SubStructure.construct(typ = SubStructure.CYLINDER, signal = signal, center = [0,0,0], size = size, rotation = no_rot, resolution = resolution, anisoDim = 2)
    
    [fig, ani] = util.utilities.imshow3D_ani(round_cyl.out)
    util.utilities.imshow3D_slice(round_cyl.out)
    util.utilities.imshow3D_slice(round_cyl2.out)
    plt.show()


#---------------------------------------------------------------------- 
def main_neg_rot_test():
    signal = 20.
    size = [1.,1.,3.]
    resolution = [0.1, 0.1, 0.1]
    
    no_rot = [0.,0.,0.]
    pos_rot = [0.5, 0., 0.]
    neg_rot = [-rot for rot in pos_rot]
    
    pos_rot2 = [-0.15, -0.07, 0.]
    neg_rot2 = [-rot for rot in pos_rot2]
    
    #rect_neg = SubStructure.construct(typ=SubStructure.RECTANGLE, signal = signal, center = [0,0,0], size = size, rotation=neg_rot, resolution=resolution, anisoDim = 2)
    #rect_pos = SubStructure.construct(typ=SubStructure.RECTANGLE, signal = signal, center = [0,0,0], size = size, rotation=pos_rot, resolution=resolution, anisoDim = 2)
    #rect_no = SubStructure.construct(typ=SubStructure.RECTANGLE, signal = signal, center = [0,0,0], size = size, rotation=no_rot, resolution=resolution, anisoDim = 2)
    
    #pyr_neg = SubStructure.construct(typ=SubStructure.PYRAMID, signal = signal, center = [0,0,0], size = size, rotation=neg_rot, resolution=resolution, anisoDim = 2)
    #pyr_pos = SubStructure.construct(typ=SubStructure.PYRAMID, signal = signal, center = [0,0,0], size = size, rotation=pos_rot, resolution=resolution, anisoDim = 2)    
    #pyr_no = SubStructure.construct(typ=SubStructure.PYRAMID, signal = signal, center = [0,0,0], size = size, rotation=no_rot, resolution=resolution, anisoDim = 2)    
    
    #ell_no = SubStructure.construct(typ=SubStructure.ELLIPSE, signal = signal, center = [0,0,0], size = size, rotation=no_rot, resolution=resolution, anisoDim = 2)
    #ell_neg = SubStructure.construct(typ=SubStructure.ELLIPSE, signal = signal, center = [0,0,0], size = size, rotation=neg_rot, resolution=resolution, anisoDim = 2)
    #ell_pos = SubStructure.construct(typ=SubStructure.ELLIPSE, signal = signal, center = [0,0,0], size = size, rotation=pos_rot, resolution=resolution, anisoDim = 2)    
    
    #cyl_no = SubStructure.construct(typ=SubStructure.CYLINDER, signal = signal, center = [0,0,0], size = size, rotation=no_rot, resolution=resolution, anisoDim = 2)
    #cyl_neg = SubStructure.construct(typ=SubStructure.CYLINDER, signal = signal, center = [0,0,0], size = size, rotation=neg_rot, resolution=resolution, anisoDim = 2)
    #cyl_pos = SubStructure.construct(typ=SubStructure.CYLINDER, signal = signal, center = [0,0,0], size = size, rotation=pos_rot, resolution=resolution, anisoDim = 2)    
    
    r_cyl_no = SubStructure.construct(typ=SubStructure.ROUND_CYL, signal = signal, center = [0,0,0], size = size, rotation=no_rot, resolution=resolution, anisoDim = 2)
    r_cyl_neg = SubStructure.construct(typ=SubStructure.ROUND_CYL, signal = signal, center = [0,0,0], size = size, rotation=neg_rot, resolution=resolution, anisoDim = 2)
    r_cyl_pos = SubStructure.construct(typ=SubStructure.ROUND_CYL, signal = signal, center = [0,0,0], size = size, rotation=pos_rot, resolution=resolution, anisoDim = 2)        
    
    c_struc_neg = r_cyl_neg
    c_struc_pos = r_cyl_pos
    c_struc_no = r_cyl_no
    
    [fig, ani] = util.utilities.imshow3D_ani(c_struc_neg.out)
    
    util.utilities.imshow3D_slice(c_struc_neg.out)
    util.utilities.imshow3D_slice(c_struc_pos.out)
    util.utilities.imshow3D_slice(c_struc_no.out)
    plt.show()
    
#----------------------------------------------------------------------
def main_SampleTest():
    """"""
    
    sim_sample = Sim_Sample([51.2, 51.2, 12.8], 
                            [0.5, 0.5, 1.], 
                            background=240., 
                            imageConstructType= Sim_Sample.SET, 
                            randomizeOrientation=0., 
                            randomizePosition=0., 
                            sampleParams=None)
    
    sample_params = [['pyramid',{'signal':10., 'center':[0.,0.,0.], 'size':[2.,2.,5.], 'anisoDim':1}],
                     ['pyramid',{'signal':10., 'center':[2.,2.,2.], 'size':[2.,2.,5.], 'anisoDim':1}],
                     ['pyramid',{'signal':10., 'center':[4.,4.,4.], 'size':[2.,2.,5.], 'anisoDim':1}]]
    
    
    sim_sample.setSampleParams(sample_params)
    #print(sim_sample.sampleParams)
    sim_sample.constructSample()
    
    [fig,ani] = util.utilities.imshow3D_ani(sim_sample.out)
    
    
    plt.show()


    
if __name__ == '__main__':
    pass
    #construct_recon_sample()
    #main_substructureTest()
    #main_neg_rot_test()
    