#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 08:16:07 2020

@author: alain
"""

import numpy as np

import bpy

from wrapanime.utils.errors import WrapException

from wrapanime.wrappers.generated_wrappers import WTexture

class Noise(WTexture):
    def __init__(self, texture):
        if type(texture) is str:
            tx = bpy.data.textures.get(texture)
            if tx is None:
                raise WrapException(
                    f"Noise initialization error: '{texture}' texture not found."
                    )
        else:
            tx = texture
            
        super().__init__(tx, None)
        self._min = 0.
        self._amp = 1.
        
    @property
    def minimum(self):
        return self._min
    
    @minimum.setter
    def minimum(self, value):
        self._min = value
        
    @property
    def maximum(self):
        return self._min + self._amp

    @maximum.setter
    def maximum(self, value):
        self._amp = value - self._min
        
    @property
    def grayscale(self):
        return self.obj.cloud_type == 'GRAYSCALE'
    
    @grayscale.setter
    def grayscale(self, value):
        self.obj.cloud_type = 'GRAYSCALE'if value else 'COLOR'
        
    def __call__(self, v):
        tx = self.obj
        
        vs = np.array(v, np.float)
        if len(vs.shape) == 0:
            single = True
            noise = np.array([tx.evaluate((0., 0., v))])
        elif len(vs.shape) == 1:
            single = True
            noise = np.array([tx.evaluate(v)])
        elif len(vs.shape) == 2:
            single = False
            if vs.shape[-1] == 1:
                f = np.vectorize(lambda t: np.array(tx.evaluate((0., 0., t))), signature="(1)->(4)")
            else:
                f = np.vectorize(lambda t: np.array(tx.evaluate(t)), signature="(3)->(4)")
                #f = np.vectorize(test, signature="(3)->(4)")
            noise = f(vs)
        else:
            raise WrapException(
                f"Noise evaluation error: evaluation can't be made on arrays of shape {tx.shape}"
                )
            
        if tx.cloud_type == 'GRAYSCALE':
            if single:
                return self._min + noise[0, 3]*self._amp
            else:
                return self._min + noise[:, 3]*self._amp
        else:
            if single:
                return self._min + noise[0]*self._amp
            else:
                return self._min + noise*self._amp
            
