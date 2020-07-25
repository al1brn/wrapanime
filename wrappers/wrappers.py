#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 09:42:35 2020

@author: alain
"""

import numpy as np

import bpy

from wrapanime.utils.errors import WrapException
from wrapanime.utils import blender as blender

from wrapanime.utils import cy_object
from wrapanime.utils import geometry as geo
import wrapanime.wrappers.root as root

from wrapanime.wrappers.generated_wrappers import WObject, WMesh
from wrapanime.wrappers import generated_wrappers as wgen

from wrapanime.functions.bezier import BezierCurve

import importlib
importlib.reload(geo)
importlib.reload(wgen)
importlib.reload(root)
importlib.reload(blender)

WObjects = wgen.WObjects


# =============================================================================================================================
# To vector

def to_vector(value, dim=3):
    return root.to_array(value, [dim], f"(to_vector) single value or a {dim}-vector, not {value}")


# =============================================================================================================================
# Empty object wrapper

class WEmpty(WObject):
    def __init__(self, obj):
        obj = blender.get_object(obj, 'EMPTY')
        super().__init__(obj)
        
    @classmethod
    def New(cls, name=None):
        return WObject.New(name=name, otype='EMPTY')
    
        

# =============================================================================================================================
# Mesh object wrapper

class WMeshObject(WObject):
    def __init__(self, obj):

        obj = blender.get_object(obj, 'MESH')

        super().__init__(obj)
        self.wmesh = WMesh(self.obj.data, self)

        self._verts             = None
        self._verts_normals     = None

        self._faces_centers     = None
        self._faces_normals     = None
        self._faces             = None

        self._edges             = None

    def erase_cache(self):
        super().erase_cache()

        self._verts             = None
        self._verts_normals     = None

        self._faces_centers     = None
        self._faces_normals     = None
        self._faces             = None

        self._edges             = None
        
    # ---------------------------------------------------------------------------
    # New
    
    @classmethod
    def New(cls, shape='CUBE', name=None, **kwargs):
        
        if name is not None:
            obj = blender.get_object(name, mandatory=False, otype='MESH')
            if obj is not None:
                return cls(obj)
        
        if shape == 'CIRCLE':
            bpy.ops.mesh.primitive_circle_add(**kwargs)
        elif shape == 'CONE':
            bpy.ops.mesh.primitive_cone_add(**kwargs)
        elif shape == 'CUBE':
            bpy.ops.mesh.primitive_cube_add(**kwargs)
        elif shape == 'CUBE_GIZMO':
            bpy.ops.mesh.primitive_cube_add_gizmo(**kwargs)
        elif shape == 'CYLINDER':
            bpy.ops.mesh.primitive_cylinder_add(**kwargs)
        elif shape == 'GRID':
            bpy.ops.mesh.primitive_grid_add(**kwargs)
        elif shape == 'ICO_SPHERE':
            bpy.ops.mesh.primitive_ico_sphere_add(**kwargs)
        elif shape == 'MONKEY':
            bpy.ops.mesh.primitive_monkey_add(**kwargs)
        elif shape == 'PLANE':
            bpy.ops.mesh.primitive_plane_add(**kwargs)
        elif shape == 'TORUS':
            bpy.ops.mesh.primitive_torus_add(**kwargs)
        elif shape == 'UV_SPHERE':
            bpy.ops.mesh.primitive_uv_sphere_add(**kwargs)
        else:
            raise WrapException(
                f"New mesh object error: the shape '{shape}' is not valid",
                "Shape must be in [CIRCLE, CONE, CUBE, CYLINDER, GRID, ICO_SPHERE, MONKEY, PLANE, TORUS, UV_SPHERE]"
                )

        obj = bpy.context.active_object
        if name is not None:
            obj.name = name

        return cls(obj)
        

    # ---------------------------------------------------------------------------
    # Mesh ease functions

    # === Vertices without transformation

    @property
    def verts(self):
        if self._verts is None:
            mesh = self.eval_object.data
            self._verts = cy_object.get_vertices(mesh, self.full_matrix)
        return self._verts

    @verts.setter
    def verts(self, values):
        nverts = len(self.wmesh.vertices)
        shape = (nverts, 3)
        vals = root.to_array(values, shape, f"a single vector or an array of {nverts} vectors")
        cy_object.set_vertices(self.obj.data, vals, mask='XYZ')
        self._verts = None

    def set_verts(self, verts, mask='XYZ'):
        nverts = len(self.wmesh.vertices)
        shape = (nverts, 3)
        vals = root.to_array(verts, shape, "a single vector or an array of {} vectors".format(nverts))
        cy_object.set_vertices(self.obj.data, vals, mask=mask)
        self._verts = None

    # === Normals to vertices (read only)

    @property
    def verts_normals(self):
        if self._verts_normals is None:
            mesh = self.eval_object.data
            self._verts_normals = cy_object.get_transformed_array(mesh.vertices, 'normal', self.rot_matrix)
        return self._verts_normals

    # === faces

    @property
    def faces_centers(self):
        if self._faces_centers is None:
            mesh = self.eval_object.data
            self._faces_centers = cy_object.get_transformed_array(mesh.polygons, 'center', self.full_matrix)
        return self._faces_centers

    @property
    def faces_normals(self):
        if self._faces_normals is None:
            mesh = self.eval_object.data
            self._faces_normals = cy_object.get_transformed_array(mesh.polygons, 'normal', self.rot_matrix)
        return self._faces_normals

    @property
    def faces(self):
        if self._faces is None:
            mesh = self.obj.data
            polys = mesh.polygons
            self._faces = np.empty(len(polys), np.object)
            for ipoly, poly in enumerate(polys):
                self._faces[ipoly] = [self.verts[iv] for iv in poly.vertices]
        return self._faces

    @property
    def edges(self):
        if self._edges is None:
            mesh = self.obj.data
            self._edges = [[self.verts[edge.vertices[0]], self.verts[edge.vertices[1]]] for edge in mesh.edges]
        return self._edges


# =============================================================================================================================
# Curve
        
class WCurve(wgen.WCurve):
    @property
    def t0(self):
        return self.bevel_factor_start
    @t0.setter
    def t0(self, value):
        self.bevel_factor_start = value
        
    @property
    def t1(self):
        return self.bevel_factor_end
    @t1.setter
    def t1(self, value):
        self.bevel_factor_end = value


# =============================================================================================================================
# Curve object wrapper

class WCurveObject(WObject):
    def __init__(self, obj):
        obj = blender.get_object(obj, 'CURVE')
        super().__init__(obj)
        
        self.wcurve   = WCurve(self.obj.data, self)
        self.wsplines = self.wcurve.wsplines
        
    # ---------------------------------------------------------------------------
    # New Curve object
    
    @classmethod
    def New(cls, shape='BEZIER', name=None, **kwargs):
        
        if name is not None:
            obj = blender.get_object(name, mandatory=False, otype='CURVE')
            if obj is not None:
                return cls(obj)
        
        if shape == 'CIRCLE':
            bpy.ops.curve.primitive_bezier_circle_add(**kwargs)
        elif shape == 'BEZIER':
            bpy.ops.curve.primitive_bezier_curve_add(**kwargs)
            
        elif shape == 'NURBS_CIRCLE':
            bpy.ops.curve.primitive_nurbs_circle_add(**kwargs)
        elif shape == 'NURBS_CURVE':
            bpy.ops.curve.primitive_nurbs_curve_add(**kwargs)
        elif shape == 'PATH':
            bpy.ops.curve.primitive_nurbs_path_add(**kwargs)
        
        else:
            raise WrapException(
                f"New curve object error: the shape '{shape}' is not valid",
                "Shape must be in [CIRCLE, BEZIER, NURBS_CIRCLE, NURBS_CURVE, PATH]"
                )

        obj = bpy.context.active_object
        if name is not None:
            obj.name = name
            
        return cls(obj)
    
        
    # ---------------------------------------------------------------------------
    # Bezier points
    
    @property
    def bpoints(self):
        return self.wcurve.bpoints
    
    @bpoints.setter
    def bpoints(self, bpoints):
        self.wcurve.bpoints = bpoints
        
    @property
    def bezier_curve(self):
        bc = self.wcurve.bezier_curve
        bc.length = self.obj.path_duration
        return bc

    def set_function(self, f, t0, t1, count=100):
        self.wcurve.set_function(f, t0, t1, count=count)
        

# =============================================================================================================================
# Wrap a Blender object

def wrap(obj):
    obj = blender.get_object(obj)

    if obj.type == 'MESH':
        return WMeshObject(obj)
    if obj.type == 'CURVE':
        return WCurveObject(obj)
    elif obj.type == 'EMPTY':
        return WEmpty(obj)
    else:
        return WObject(obj)
    
        #raise WrapException(
        #    "Object not wrappable", obj,
        #    f"No wrapper is implemented for type '{obj.type}'")

# =============================================================================================================================
# Keyframe curves
        
class WKeyFrames(wgen.WKeyFrames):
    
    def set_length(self, length):
        
        # Remove the ones which are two numerous
        exceed = length-len(self)
        for i in range(exceed):
            self.obj.keyframes.remove(self.obj[-1], fast=True)
            
        # Complete if not enough
        need = length - len(self)
        if need > 0:
            self.obj.add(need)
    
    def set_bezier_func(self, bezier):
        self.set_length(len(bezier))
        self.bpoints2 = bezier.points
    
    
class KFAnimation():
    def __init__(self, obj):
        self.obj = blender.get_object(obj)
        
    @property
    def is_animated(self):
        return self.obj.animation_data is not None
        
    @property
    def animation(self):
        animation = self.obj.animation_data
        if animation is None:
            return self.obj.animation_data_create()
        else:
            return animation
        
    @property
    def action(self):
        action = self.animation.action
        if action is None:
            self.animation.action = bpy.data.actions.new(name="WA action")
            return self.animation.action
        
        return action
    
    @property
    def fcurves(self):
        return self.action.fcurves
    
    @classmethod
    def fcurve_of(cls, fcurve, name, index):
        if index < 0:
            return fcurve.data_path == name
        else:
            return (fcurve.data_path == name) and (fcurve.array_index == index)
    
    def fcurve(self, name, index=0):
        if self.fcurves is None:
            return None
        
        for curve in self.fcurves:
            if self.fcurve_of(curve, name, index):
                return curve
                
        return None
    
    def new(self, name, index=0):
        curve = self.fcurve(name, index)
        if curve is None:
            curve = self.fcurves.new(data_path=name, index=index)
        return curve
    
    # ===== Manage the keyframes and curves
    
    def keyframes(self, name, index=0, create=True):
        
        if create:
            curve = self.new(name, index)
        else:
            curve = self.fcurve(self, name, index)
        
        if curve is None:
            return None
        
        return WKeyFrames(curve.keyframe_points, wowner=None)
            

