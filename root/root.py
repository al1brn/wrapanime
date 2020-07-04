#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 09:42:35 2020

@author: alain
"""

import itertools

import numpy as np

import wrapanime.cython.arrays as arrays
from wrapanime.utils.errors import WrapShapeException, WrapException

# =============================================================================================================================
# Adust the shape of a value
# If the shape is a single value, it is used to fill an array
# Otherwise, raise an error

def to_array(values, shape, error_message):

    # Shape can be built as (n, 1) in that case we have to use (n)
    if len(shape) > 1 and shape[-1] == 1:
        shape = [shape[i] for i in range(len(shape)-1)]

    # Values is a list
    if hasattr(values, '__len__'):
        vals = np.array(values)

    # Values is a simple value
    else:
        # A single value is acceptable only if the target has one dimension
        if len(shape) == 1:
            return np.full(shape, values)

        # Otherwise, it doesn't work
        raise WrapShapeException(type(values).__name__, shape, "Expected: {}, received: {}".format(error_message, values))

    # If the elements count is the one required, we simply reshape
    target_size = 1
    for n in shape:
        target_size *= n

    if target_size == vals.size:
        return vals.reshape(shape)

    # The number of elements doesn't match the one which is required
    # We can see if it is the single array value to set to an array of such arrays
    # Typical use: array[Vector] = Vector

    if target_size // shape[0] == vals.size:
        # Tile the resulting array with tile reps
        # (first dime size, 1, 1, 1)
        reps = list(shape)
        for i in range(1, len(reps)):
            reps[i] = 1
        return np.tile(vals, reps)

    # We can't to nothing !!!

    raise WrapShapeException(vals.shape, shape, "Expected: {}, received: {}".format(error_message, values))

# =============================================================================================================================
# A single value can be vertorized using an iterator

def vectorize(values, shape, caller="unknown"):

    if len(shape) > 1 and shape[-1] == 1:
        shape = shape[:-1]

    if not hasattr(values, '__len__'):
        if len(shape) == 1:
            return itertools.repeat(values, shape[0])
        else:
            raise WrapException(
             "Vectorization error for '{}'".format(caller),
             "Expected: array of {} items of shape {}".format(shape[0], shape[1:]),
             "Received: single value {}".format(values)
             )

    # Target size
    size = 1
    for d in shape:
        size *= d

    # Needs the shape
    vals = np.array(values)

    # Shape is good
    if vals.size == size:
        return vals.reshape(shape)

    # If shape has only one dimension, bad news
    if len(shape) == 1:
        raise WrapException(
         "Vectorization error for '{}'".format(caller),
         "Expected: array of {} items".format(shape[0]),
         "Received: array of {} items".format(vals.size)
         )

    # Shape is considered as list of shape[0] items of shape [1:]
    item_size = size / shape[0]
    if vals.size == item_size:
        return itertools.repeat(vals.reshape(shape[1:]), shape[0])

    raise WrapException(
     "Vectorization error for '{}'".format(caller),
     "Expected: array of {} items of shape {}, total= {} values".format(shape[0], shape[1:], size),
     "Received: array of {} items, total= {} values".format(vals.shape[0], vals.size)
     )


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

    def wrap(self, obj):
        return self.Cls.Wrap(obj, None)

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
                self._array[i] = self.wrap(values[i])

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

    # loop
    def for_each(self, f, **kwargs):
        arrays.for_each(self.array, f, **kwargs)

    # Cache management
    def erase_cache(self):
        arrays.for_each(self.array, lambda o: o.erase_cache())
        return

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
