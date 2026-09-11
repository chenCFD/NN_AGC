import os, ctypes, pkg_resources
import numpy as np

class Functions(object):
    
    def __init__(self, lib_name, n_mic=1, frame_len=480, freq_len=481, fs=24000):
        lib_name = self.get_lib_name(lib_name)
        self.fs = fs
        self.n_mic = n_mic
        self.frame_size = frame_len
        self.freq_len = freq_len
        self.c_short_p = ctypes.POINTER(ctypes.c_short)
        self.c_float_p = ctypes.POINTER(ctypes.c_float)
        self.lib = np.ctypeslib.load_library(lib_name, '.')
        
        self.lib.agc_feature_generation.argtypes = \
            [ctypes.c_void_p, 
            self.c_short_p,
            self.c_float_p]
        self.lib.agc_feature_generation.restype = ctypes.c_void_p
        
        self.lib.agc_create.restype = ctypes.c_void_p
        self.lib.agc_destroy.argtypes = [ctypes.c_void_p]
        self.agc_obj = self.lib.agc_create(None)
        self.feature = np.zeros(self.freq_len).astype(ctypes.c_float)
        
        
    def get_lib_name(self, lib_name):
        pkg_name = __name__
        subname = 'libnnagc'
        
        # If no specified library name
        if not lib_name:
            found_lib_name = pkg_resources.resource_filename(pkg_name, './libs/{}.so.1'.format(subname))
            if not os.path.exists(found_lib_name):
                raise NameError('Default library not exist!\n')
                
        else:
            available_libs = pkg_resources.resource_listdir(pkg_name, './libs/')
            found_lib_name = None
            for lib in available_libs:
                if lib.find(lib_name) != -1:
                    found_lib_name = pkg_resources.resource_filename(pkg_name, './libs/{}'.format(lib_name))
                    
            if not found_lib_name:
                raise NameError('{} library not found\n'.format(lib_name))
        
        return found_lib_name
    
    def feature_generation_training(self, mic):
        mic_buf = mic.astype(ctypes.c_short)
        mic_ptr = mic_buf.ctypes.data_as(self.c_short_p)

        feature_ptr = self.feature.ctypes.data_as(self.c_float_p)
        
        self.lib.agc_feature_generation(self.agc_obj, mic_ptr, feature_ptr)