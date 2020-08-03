#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 09:42:35 2020

@author: alain
"""

import numpy as np

# *****************************************************************************************************************************
# Root class: array of items
# Items can be of simple types such as float, vector or str
# They can also be complex types such as Object or Curves
# 
# Properties of managed items are accessibles through cached vectorized properties
        
class ArrayOf():
    
    def __init__(self, Cls, wowner):
        self.Cls    = Cls
        self.wowner = wowner
        self._array = np.empty(0, 'object')
        
    @property
    def array(self):
        return self._array
    
    @array.setter
    def array(self, values):
        if values is None:
            self.new_array_cache(0)
        else:
            self.new_array_cache(len(values))
            for i in range(len(values)):
                self._array[i] = self.Cls.Wrap(values[i], None)
                
    def new_array_cache(self, length):
        self.erase_cache()
        if len(self._array) != length:
            del self._array
            self._array = np.empty(length, 'object')
            
    def __len__(self):
        return len(self._array)
    
    def __getitem__(self, index):
        return self._array[index]     
    
    def set_length(self, length):
        if length != len(self):
            self.new_array_cache(length)
    
    # Cache management
    def erase_cache(self):
        if self._array is not None:
            for o in self._array:
                o.erase_cache()
                

# *****************************************************************************************************************************
# Wrapper root class
#
# Wrapper sub class can be enriched
# 1) By generating the code declaring the attributes of the wrapped class
# 2) Adding specific cached properties
#
# Exemple with Blender Object wrapping
# 1) In Blender generate the code for the wrapping properties
# > obj = bpy.data.objects["Cube"]
# > write_codes(file_name, wrap_properties(obj, exclude=[]]))
#
# 2) Add additional properties
# def get_mesh(self):
#     Get mesh data taking into account the transformations
#
# 3) Add dynamic properties
# add_cached_property(Wrapper, 'verts', create_verts, None, keep=False)

                
class Wrapper():
    
    def __init__(self, obj, wowner):
        self.obj    = obj
        self.wowner = wowner
        
    @classmethod
    def Wrap(Cls, obj, wowner):
        if issubclass(type(obj), Wrapper):
            return obj
        else:
            return Cls(obj, wowner)
        
    # Cached properties are created dynamically
    # The names of the caches which can be erased are in class list
        
    def erase_cache(self):
        pass

