#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 12:12:27 2020

@author: alain
"""

import numpy as np

#from wrapanime.utils.errors import WrapException

WrapException  = Exception

class VertArray():
    def __init__(self, length=0, buffer=100, vector_size=3, vtype=np.float):
        self.length = length
        self.buffer = max(buffer, self.length//10)
        self.vsize  = vector_size
        self.vtype  = vtype
        self._array = np.zeros((max(self.buffer, self.length), self.vsize), vtype)
        
    def __repr__(self):
        return f"VertArray[verts: {self.length} of size {self.vsize}] array size={self._array.size} shape={self.array.shape}]\n{self.array}"
    
    # Array implementation
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return self._array[index]
    
    def __setitem__(self, index, V):
        self._array[index] = V
        
    # User array
        
    @property
    def array(self):
        return self._array[:self.length]
    
    @array.setter
    def array(self, a):
        a = np.array(a)
        self.length = a.size // self.vsize
        self._array = a.reshape(self.length, self.vsize)
    
    @property
    def linear_array(self):
        return self.array.reshape(self.length*self.vsize)
        
    def update_buffer(self, target):
        if target > len(self._array):
            target += self.buffer
            self._array = np.resize(self._array, (target, self.vsize))
            
    def set_length(self, length):
        if length > self.length:
            self.update_buffer(length)
        self.length = length
        
    def add(self, vs):
        a = np.array(vs)
        count = a.size//self.vsize
        l = self.length
        
        self.set_length(l + count)
        self._array[l:l+count] = a.reshape(count, self.vsize)
        
    @property
    def xs(self):
        return self._array[:self.length, 0]
    
    @property
    def ys(self):
        return self._array[:self.length, 1]
    
    @property
    def zs(self):
        return self._array[:self.length, 2]
    
    @xs.setter
    def xs(self, v):
        a = np.array(v)
        if (a.size != 1) and (a.size != self.length):
            raise WrapException("Vertices array error: impossible to set x values",
                                f"There are {len(self)} vertices, impossible to set them with {a.size} values"
                                )
        self._array[:self.length, 0] = a
        
    @ys.setter
    def ys(self, v):
        a = np.array(v)
        if (a.size != 1) and (a.size != self.length):
            raise WrapException("Vertices array error: impossible to set y values",
                                f"There are {len(self)} vertices, impossible to set them with {a.size} values"
                                )
            
        self._array[:self.length, 1] = a
        
    @zs.setter
    def zs(self, v):
        a = np.array(v)
        if (a.size != 1) and (a.size != self.length):
            raise WrapException("Vertices array error: impossible to set z values",
                                f"There are {len(self)} vertices, impossible to set them with {a.size} values"
                                )
            
        self._array[:self.length, 2] = a
    
