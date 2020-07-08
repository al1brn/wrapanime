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
        self._array = np.zeros(max(self.buffer, self.length)*self.vsize, vtype)
        
    def __repr__(self):
        return f"Vertices: {self.length}\n" + self.array.__repr__()
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        if index < 0 or index >= self.length:
            raise WrapException(
                    "Vertices array index error",
                    f"Array length is {self.length}, requested index is {index}"
                    )
        ix = self.vsize * index
        return self._array[ix:ix+3]
    
    def __setitem__(self, index, V):
        if index < 0 or index >= self.length:
            raise WrapException(
                    "Vertices array index error",
                    f"Array length is {self.length}, requested index is {index}"
                    )
        ix = self.vsize * index
        self._array[ix:ix+3] = V
        
    def __array__(self):
        return self.linear_array
    
    @property
    def array(self):
        return self._array[:self.length*self.vsize].reshape((self.length, self.vsize))
    
    @property
    def full_array(self):
        return self._array.reshape(self.length//self.vsize, self.vsize)
    
    @property
    def linear_array(self):
        return self._array[:self.length*self.vsize]
        
    def update_buffer(self, target):
        bcount, rem = divmod(target, self.buffer)
        if rem > 0:
            bcount += 1
        target = bcount*self.buffer
        a_size = target*self.vsize
        if a_size > len(self._array):
            self._array = np.resize(self._array, a_size)
            
    def set_length(self, length):
        if length > self.length:
            self.update_buffer(length)
        self.length = length
        
    def add(self, Vs):
        vs     = np.array(Vs)
        vcount = vs.size
        if vcount % self.vsize != 0:
            raise WrapException(
                    "Vertices array error:; impossible to add an array of vectors",
                    Vs,
                    f"The size of the array ({vs.size}) must be a multiple of {self.vsize}"
                    )
            
        count = vcount // self.vsize
            
        ix = self.length * self.vsize
        self.set_length(self.length + count)
        self._array[ix:ix+vcount] = vs.reshape(vcount)
        
    @property
    def x(self):
        return self.array[:, 0]
    
    @property
    def y(self):
        return self.array[:, 1]
    
    @property
    def z(self):
        return self.array[:, 1]
    
    @x.setter
    def x(self, v):
        npx = np.array(v)
        if (npx.size != 1) and (npx.size != self.length):
            raise WrapException("Vertices array error: impossible to set x values",
                                f"There are {len(self)} vertices, impossible to set them with {npx.size} values"
                                )
            
        a = self._array.reshape(len(self._array)//self.vsize, self.vsize)
        a[:self.length, 0] = npx
        self._array = a.reshape(a.size)
        
    @y.setter
    def y(self, v):
        npx = np.array(v)
        if (npx.size != 1) and (npx.size != self.length):
            raise WrapException("Vertices array error: impossible to set y values",
                                f"There are {len(self)} vertices, impossible to set them with {npx.size} values"
                                )
            
        a = self._array.reshape(len(self._array)//self.vsize, self.vsize)
        a[:self.length, 1] = npx
        self._array = a.reshape(a.size)
        
    @z.setter
    def z(self, v):
        npx = np.array(v)
        if (npx.size != 1) and (npx.size != self.length):
            raise WrapException("Vertices array error: impossible to set z values",
                                f"There are {len(self)} vertices, impossible to set them with {npx.size} values"
                                )
            
        a = self._array.reshape(len(self._array)//self.vsize, self.vsize)
        a[:self.length, 2] = npx
        self._array = a.reshape(a.size)
    
