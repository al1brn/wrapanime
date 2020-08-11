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

from wrapanime.utils import blender

from wrapanime.utils import geometry as geo
from wrapanime.functions.bezier import BezierCurve

import wrapanime.cython.arrays as arrays
from wrapanime.utils.errors import WrapException

from wrapanime.functions.tween import WFCurve

# =============================================================================================================================
# Adjust the shape of a value
# If the shape is a single value, it is used to fill an array
# Otherwise, raise an error

def to_array(values, shape, error_message):

    # Shape can be dynamically built as (n, 1) in that case we have to use (n)
    if len(shape) > 1 and shape[-1] == 1:
        shape = np.array(shape)[:len(shape)-1]
        
    # Target size
    target_size = np.prod(shape)
    
    # Source size
    source_size = np.array(values).size

    # Sizes are the same (n, 3) <- (n, 3)
    if source_size == target_size:
        return values
    
    # A single value as an input : (n, 3) <- 1
    if source_size == 1:
        return np.full(shape, values)
    
    # Partial shape : (n, 3) <- (3)
    for i in range(len(shape)-1):
        if source_size == np.prod(shape[i-1:]):
            return np.full(shape, values)
        
    raise WrapException(
        f"Broadcast error : impossible to braodcast array of shape {np.array(values).shape} to array {shape} ({error_message})"
        )
    

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

class ArrayOf_DEPRECATED():

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

    def __init__DEPRECATED(self, root_wrapper=None, wowner=None, index=None):

        """
        if type(name) is str:
            self.name = name
        else:
            try:
                self.name = name.name
            except:
                raise WrapException(
                    "Wrapper initialization error: initialization needs either or name or an object with an name attribute.",
                    f"{name} is not valid for initialization"
                    )
        """
        
        if root_wrapper is not None:
            self.root_wrapper = root_wrapper
        if wowner is not None:
            self.wowner = wowner
        if index is not None:
            self.windex = index
            
    @property
    def top_obj_DEPRECATED(self):
        return self.root_wrapper.top_obj
            
            
    # Must implement two properties
    # top_obj
    # obj for objects wrappers and coll for coll wrappers
                
    #@property
    #def top_obj(self):
    #    pass
    
    #@property
    #def obj(self):
    #    pass
        
    def __repr__(self):
        name = type(self).__name__.split(".")[-1]
        return f"[{name}({type(self.bstruct)}]"
        
    # Cached properties are created dynamically
    # The names of the caches which can be erased are in class list

    def erase_cache(self):
        pass
    
    # ----------------------------------------------------------------------------------------------------
    # Key frame animation
    
    def _path_index(self, name):
        """Transform a user path en blender (data_path, array_index) couple.
        
        Parameters
        ----------
        name: str
            An extended data_path string.
            - 'y' and 'location.y' are interpreted as ('location', 1)
            - 'data.attr' is interpretated as ('attr', 0) for data property 
            
        Returns
        -------
        triplet (object, data_path, array_index)
        """
        
        indices = ['x', 'y', 'z', 'w']
        
        dic = {
            'x' : ('location',       0), 'y' : ('location',       1), 'z' : ('location',       2), 
            'sx': ('scale',          0), 'sy': ('scale',          1), 'sz': ('scale',          2), 
            'rx': ('rotation_euler', 0), 'ry': ('rotation_euler', 1), 'rz': ('rotation_euler', 2), 
            }
        
        index = 0
        parts = name.split('.')
        if parts[-1] in indices:
            index = indices.index(parts[-1])
            parts = parts[:-1]
        else:
            try:
                n, index = dic(parts[-1])
                parts[-1] = n
            except:
                pass
            
        obj = self.bstruct
        for i in range(len(parts)-1):
            try:
                obj = getattr(obj, parts[i])
            except:
                raise WrapException(
                    f"Incorrect animation path: '{name}' is not valid for {self}"
                    )
                
        return obj, parts[-1], index

    # ----------------------------------------------------------------------------------------------------
    # Key frame animation
        
    @property
    def is_animated(self):
        return blender.is_animated(self.bstruct)
        
    @property
    def animation(self):
        return blender.animation_data(self.bstruct)
        
    @property
    def action(self):
        return blender.animation_action(self.bstruct)
    
    @property
    def fcurves(self):
        return blender.get_fcurves(self.bstruct)
    
    def get_fcurve(self, name):
        return blender.get_fcurve(self.bstruct, name)
    
    def new_curve(self, name):
        return blender.new_curve(self.bstruct, name)
    
    def set_fcurve(self, name, fcurve):
        blender.set_fcurve(self.bstruct, name, fcurve)
    
    # ---------------------------------------------------------------------------
    # Delete key frames
    
    def kf_delete(self, name, frame0=None, frame1=None):
        blender.kf_delete(self.bstruct, name, frame0, frame1)
    
    # ---------------------------------------------------------------------------
    # Set a key frame at a given frame
    
    # Shortcut for keyframes
    def kf_insert(self, name, frame, value):
        blender.kf_insert(self.bstruct, name, frame, value)
        
    def kf_interval(self, name, frame0, frame1, value0, value1, interpolation='LINEAR'):
        blender.kf_interval(self.bstruct, name, frame0, frame1, value0, value1, interpolation)
            
# *****************************************************************************************************************************
# Enrich WObject with usefull methods
# The Object wrapper will use this class and provide ArraOf methods            
            
class WObjectRoot(Wrapper):

    @property
    def eval_object(self):
        return self.object.evaluated_get(bpy.context.evaluated_depsgraph_get())

    @property
    def full_matrix(self):
        return self.object.matrix_world

    @property
    def rot_matrix(self):
        return self.object.matrix_world.to_3x3().normalized().to_4x4()
    
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
            self.rotation_quaternion = Quaternion(value)
        else:
            self.rotation_euler = Quaternion(value).to_euler(self.rotation_euler.order)

    @property
    def hide(self):
        return self.object.hide_render

    @hide.setter
    def hide(self, value):
        self.object.hide_render   = value
        self.object.hide_viewport = value

    # transformation

    def orient(self, axis):
        self.quaternion = geo.q_tracker(self.track_axis, axis, up=self.up_axis)

    def track_to(self, location):
        q = geo.q_tracker(self.track_axis, Vector(location)-self.location, up=self.up_axis)
        #print("TRACK_TO", geo._str(q, 1, 'quat'))
        self.quaternion = q

    # Distance

    def distance(self, location):
        return (Vector(location)-self.location).length       


# =============================================================================================================================
# Mesh root
        
class WMeshRoot(Wrapper):
    pass
            

# =============================================================================================================================
# Spline
        
class WSplineRoot(Wrapper):
    
    @property
    def bezier_count(self):
        return len(self.spline.bezier_points)

    @property
    def points_count(self):
        return len(self.spline.points)
        
    # ---------------------------------------------------------------------------
    # Bezier points
    
    @property
    def bpoints(self):
        bp = self.spline.bezier_points
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
        
        bp = self.spline.bezier_points
        
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
        
        if len(self.spline.points) > count:
            raise WrapException(
                "WSpline set points error: the number of control points don't match",
                f"WSpline required points: {len(self.spline.points)}",
                f"Number of given points:  {count}"
                )
        
        self.spline.points.add(count - len(self.spline.points))
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
        
class WSplinesRoot(Wrapper):
    
    def clear(self):
        self.splines.clear()
        
    def new(self, type='BEZIER'):
        index = len(self.splines)
        self.splines.new(type)
        return self.item_class(self.obj_name, windex=index)
    
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
    
        
        
        
        
        
        
     

