#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 13:14:37 2018

@author: ikaya
"""

### 21/09/18
### This is to change the Keras backend, easily.
### You can change from one to another given below
#   'cntk'
#   'theano'
#   'tensorflow'
#   'mxnet'

import os
import json

class kerasbackend():
    def __init__(self, backend = 'tensorflow' ):
        self.KERAS_BACKEND = backend
       
        
        if 'KERAS_HOME' in os.environ:
            _keras_dir = os.environ.get('KERAS_HOME')
        else:
            _keras_base_dir = os.path.expanduser('~')
            if not os.access(_keras_base_dir, os.W_OK):
                _keras_base_dir = '/tmp'
            _keras_dir = os.path.join(_keras_base_dir, '.keras')
        self._config_path = os.path.expanduser(os.path.join(_keras_dir, 'keras.json'))
        
        
        with open(self._config_path) as f:
            self._config = json.load(f)        
        self._config['backend'] = self.KERAS_BACKEND
        
        
        with open(self._config_path, 'w') as f:
            f.write(json.dumps(self._config, indent=4))
            
        print('Keras backend is: ', self.KERAS_BACKEND)
        
    def change_backend(self,new_backend):
#        with open(self._config_path) as f:
#            _config = json.load(f) 
        self.KERAS_BACKEND = new_backend
        self._config['backend'] = self.KERAS_BACKEND
        
        
        with open(self._config_path, 'w') as f:
            f.write(json.dumps(self._config, indent=4))
        
        
        print('Keras backend is changed to: ', self.KERAS_BACKEND)