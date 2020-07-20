#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 09:42:35 2020

@author: alain
"""

import itertools

import numpy as np

import bpy
from mathutils import Vector, Quaternion

from wrapanime.utils import geometry as geo
from wrapanime.functions.bezier import BezierCurve

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

    def __init__(self, wowner, cls):
        self.wowner = wowner
        self.cls    = cls
        self._array = np.empty(0, 'object')

    def wrap(self, obj):
        return self.cls.Wrap(obj, None)

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

    def __init__(self, obj, wowner=None):
        self.obj    = obj
        self.wowner = wowner
        
    def __repr__(self):
        if hasattr(self.obj, 'name'):
            name = self.obj.name
        else:
            name = f"{self.obj}"
            
        return f"Wrapper[{name} type:{type(self.obj).__name__}]"
        

    @classmethod
    def Wrap(cls, obj, wowner=None):
        if issubclass(type(obj), Wrapper):
            return obj
        else:
            return cls(obj, wowner)
        
    # Cached properties are created dynamically
    # The names of the caches which can be erased are in class list

    def erase_cache(self):
        pass
    
    # Shortcut for keyframes
    def kf_set(self, name, value, frame, index=-1):
        curr = getattr(self.obj, name)
        setattr(self.obj, name, value)
        self.obj.keyframe_insert(name, index=index, frame=frame)
        setattr(self.obj, name, curr)
        
    def kf_anime(self, name, value0, frame0, value1, frame1, interpolation='LINEAR', index=-1):
        curr = getattr(self.obj, name)
        
        setattr(self.obj, name, value0)
        self.obj.keyframe_insert(name, index=index, frame=frame0)
        
        setattr(self.obj, name, value1)
        self.obj.keyframe_insert(name, index=index, frame=frame1)
        
        setattr(self.obj, name, curr)
    
    
# *****************************************************************************************************************************
# Collection wrapper
# Eg: vertices collection in a mesh

class CollWrapper():

    def __init__(self, coll, wowner, cls):
        self.coll   = coll
        self.wowner = wowner
        self.cls    = cls
        
    def __len__(self):
        return len(self.coll)
    
    def __getitem__(self, index):
        return self.cls(self.coll[index], self)

    # loop
    def for_each(self, f, **kwargs):
        for w in self:
            f(w, **kwargs)
            
# *****************************************************************************************************************************
# Enrich WObject with usefull methods
# The Object wrapper will use this class and provide ArraOf methods            
            
class WObjectRoot(Wrapper):

    def __init__(self, obj, wowner=None):
        super().__init__(WObjectRoot.get_object(obj), wowner)
        self._eval_object = None
        self._averts      = None

    @classmethod
    def get_object(cls, obj, otype=None):
        if type(obj) is str:
            o = bpy.data.objects.get(obj)
            if o is None:
                raise WrapException(f"Impossible de initialize wrapper: Blender object '{obj}' not found")
            obj = o
        if otype is not None:
            if obj.type !=  otype:
                raise WrapException(
                        f"{cls.__name__} wrapper expect Blender object of type '{otype}'.",
                        f"Blender object '{obj.name}' is type'{obj.type}'."
                    )
        return obj

    def erase_cache(self):
        super().erase_cache()
        self._eval_object = None
        self._averts      = None

    @property
    def eval_object(self):
        if self._eval_object is None:
            depsgraph = bpy.context.evaluated_depsgraph_get()
            evobj = self.obj.evaluated_get(depsgraph)
            self._eval_object = evobj

        return self._eval_object

    @property
    def full_matrix(self):
        return self.obj.matrix_world

    @property
    def rot_matrix(self):
        return self.obj.matrix_world.to_3x3().normalized().to_4x4()
    
    # Properties which will be vectorized

    @property
    def quaternion(self):
        if self.rotation_mode == 'QUATERNION':
            return self.rotation_quaternion
        else:
            return self.rotation_euler.to_quaternion()

    @quaternion.setter
    def quaternion(self, value):
        if self.rotation_mode == 'QUATERNION':
            self.rotation_quaternion = value
        else:
            self.rotation_euler = value.to_euler(self.rotation_euler.order)

    @property
    def hide(self):
        return self.obj.hide_render

    @hide.setter
    def hide(self, value):
        self.obj.hide_render   = value
        self.obj.hide_viewport = value

    # transformation

    def orient(self, axis):
        self.quaternion = geo.tracker_quaternion(self.track_axis, axis, up=self.up_axis)

    def track_to(self, location):
        self.quaternion = geo.tracker_quaternion(self.track_axis, Vector(location)-self.location, up=self.up_axis)

    # Distance

    def distance(self, location):
        return (Vector(location)-self.location).length       


# =============================================================================================================================
# Spline
        
class WSplineRoot(Wrapper):
    
    @property
    def bezier_count(self):
        return len(self.obj.bezier_points)

    @property
    def points_count(self):
        return len(self.obj.points)
        
    # ---------------------------------------------------------------------------
    # Bezier points
    
    @property
    def bpoints(self):
        bp = self.obj.bezier_points
        count  = len(bp)
        
        bpoints = np.full((count, 9), 99., np.float)
        vx = np.empty(count*3, np.float)
        
        bp.foreach_get('co',           vx)
        bpoints[:, 3:6] = vx.reshape(count, 3)
        bp.foreach_get('handle_left',  vx)
        bpoints[:, 0:3] = vx.reshape(count, 3)
        bp.foreach_get('handle_right', vx)
        bpoints[:, 6:9] = vx.reshape(count, 3)

        return bpoints
    
    @bpoints.setter
    def bpoints(self, bpoints):
        
        bpoints = bpoints.reshape(np.array(bpoints).size//9, 9)
        count   = len(bpoints)
        
        bp = self.obj.bezier_points
        
        if len(bp) > count:
            raise WrapException(
                "WSpline set Bezier points error: the number of control points don't match",
                f"WSpline required bezier points: {len(bp)}",
                f"Number of given points:         {count}"
                )
            
        bp.add(count-len(bp))
        
        bp.foreach_set('co',           bpoints[:, 3:6].reshape(count*3))
        bp.foreach_set('handle_left',  bpoints[:, 0:3].reshape(count*3))
        bp.foreach_set('handle_right', bpoints[:, 6:9].reshape(count*3))
        
    # ---------------------------------------------------------------------------
    # Spline points
    
    @property
    def spoints(self):
        return self.wpoints.cos
    
    @spoints.setter
    def spoints(self, points):
        points = np.array(points)
        count = points.size // 4
        points = points.reshape(points.size)
        
        if len(self.obj.points) > count:
            raise WrapException(
                "WSpline set points error: the number of control points don't match",
                f"WSpline required points: {len(self.obj.points)}",
                f"Number of given points:  {count}"
                )
        
        self.obj.points.add(count - len(self.obj.points))
        self.wpoints.cos = points
        
    # ---------------------------------------------------------------------------
    # Bezier curve
        
    @property
    def bezier_curve(self):
        return BezierCurve(self.bpoints, 0., 1.)
    
    @bezier_curve.setter
    def bezier_curve(self, bezier_curve):
        self.bpoints = bezier_curve.bpoints
        
    # ---------------------------------------------------------------------------
    # By function
    
    def set_function(self, f, t0, t1, count=20):
        if self.type == 'BEZIER':
            self.bezier_curve = BezierCurve.FromFunction(f, t0, t1, count=count)
        else:
            dt = (t1-t0)/(count-1)
            a = np.array([f(t0 + i*dt) for i in range(count)])
            self.spoints = np.insert(a, 3, 1, axis=1)
            
    def update_function(self, f, t0, t1):
        if self.type == 'BEZIER':
            self.bezier_curve = BezierCurve.FromFunction(f, t0, t1, count=self.bezier_count)
        else:
            dt = (t1-t0)/(self.points_count-1)
            a = np.array([f(t0 + i*dt) for i in range(self.points_count)])
            self.spoints = np.insert(a, 3, 1, axis=1)
        
        
# =============================================================================================================================
# Splines
        
class WSplinesRoot(CollWrapper):
    
    def clear(self):
        self.coll.clear()
        
    def new(self, type='BEZIER'):
        return self.cls(self.coll.new(type), self)
    
    def get_bpoints(self, index):
        return self[index].bpoints

    def set_bpoints(self, index, bpoints):
        self[index].bpoints = bpoints
    
    def get_points(self, index):
        return self[index].points

    def set_points(self, index, points):
        self[index].points = points
    
    def get_bezier_curve(self, index):
        return self[index].bezier_curve
    
    def set_bezier_curve(self, index, bezier_curve):
        self[index].set_bezier_curve(bezier_curve)
        
    def update_bezier_curve(self, index, bezier_curve):
        self[index].update_bezier_curve(bezier_curve)

    def add_bezier_curve(self, bezier_curve):
        wspline = self.new('BEZIER')
        wspline.bezier_curve = bezier_curve
        return wspline

    def add_function(self, f, t0, t1, count=20, type='BEZIER'):
        wspline = self.new(type)
        wspline.set_function(f, t0, t1, count=count)
        return wspline
    
    def update_function(self, index, f, t0, t1):
        self[index].update_function(f, t0, t1)
    
        
        
        
        
        
        
     

