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

from wrapanime.mesh.surface import Surface

import wrapanime.wrappers.root as root

#from wrapanime.wrappers.generated_wrappers import WMesh
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
# Evaluated object
    
class WObject(wgen.WObject):
    
    def __init__(self, name, evaluated=False):
        super().__init__(name)
        self.evaluated = evaluated
        if self.evaluated:
            self._top_obj = bpy.data.objects[name].evaluated_get(bpy.context.evaluated_depsgraph_get())
            
    @property
    def top_obj(self):
        if self.evaluated:
            return self._top_obj
        else:
            return bpy.data.objects[self.name]
        
    def evaluated(self):
        return WObject(self.name, evaluated=True)
    
    @classmethod
    def New(cls, name, type, **kwargs):
        bpy.ops.object.add(type=type, **kwargs)
        obj =  bpy.context.active_object
        obj.name = name
        return cls(obj.name)
    
    # ----------------------------------------------------------------------------------------------------
    # Transform an array of vertices
    
    def world_transform(self, verts, rotation_only=False):
        
        M = self.top_obj.matrix_world
        if rotation_only:
            M = M.to_3x3().normalized().to_4x4()
            
        count = len(verts)
    
        v4d = np.ones((count, 4), np.float)
        v4d[:, :-1] = verts
    
        return np.einsum('ij,aj->ai', M, v4d)[:, :-1]
    

# =============================================================================================================================
# Empty object wrapper

class WEmpty(WObject):
    def __init__(self, obj):
        obj = blender.get_object(obj, 'EMPTY')
        super().__init__(obj.name)
        
    @classmethod
    def New(cls, name, **kwargs):
        return WObject.New(name, 'EMPTY', **kwargs)

# =============================================================================================================================
# Mesh object wrapper

class WMeshObject(WObject):
    
    def __init__(self, name):
        obj = blender.get_object(name, 'MESH')
        super().__init__(obj.name)

        self.wmesh              = wgen.WMesh(self)
        self.wvertices          = wgen.WMeshVertices(self)
        self.wedges             = wgen.WEdges(self)
        self.wloops             = wgen.WLoops(self)
        self.wpolygons          = wgen.WPolygons(self)

        self._surface           = None
        self.surface_sk         = None

    # ---------------------------------------------------------------------------
    # New
    
    @classmethod
    def New(cls, name, shape='CUBE', **kwargs):
        
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
        obj.name = name

        return cls(obj.name)

    # ---------------------------------------------------------------------------
    # Mesh ease functions

    # === Vertices without transformation

    @property
    def verts(self):
        vs = self.wvertices.vertices
        if self.evaluated:
            return self.world_transform(vs)
        else:
            return vs

    @property
    def verts_normals(self):
        vs = self.wvertices.normals
        if self.evaluated:
            return self.world_transform(vs, rotation_only=True)
        else:
            return vs
        
    @property
    def faces_centers(self):
        vs = self.polygons.centers
        if self.evaluated:
            return self.world_transform(vs)
        else:
            return vs

    @property
    def faces_normals(self):
        vs = self.polygons.normals
        if self.evaluated:
            return self.world_transform(vs, rotation_only=True)
        else:
            return vs

    @property
    def faces(self):
        verts = self.verts
        return [[verts[iv] for iv in poly.vertices] for poly in self.top_obj.data.polygons]

    @property
    def edges(self):
        verts = self.verts
        return [[verts[edge.vertices[0]], verts[edge.vertices[1]]] for edge in self.top_obj.data.polygons]
        
    # ----------------------------------------------------------------------------------------------------
    # Shape keys
    
    @classmethod
    def sk_name(cls, name, step=None):
        return name if step is None else f"{name} {step:03d}"
    
    def get_sk(self, name, step=None, create=True):
        
        name = self.sk_name(name, step)
        
        mesh = self.object.data
        obj  = self.object
    
        if mesh.shape_keys is None:
            if create:
                obj.shape_key_add(name=name)
                mesh.shape_keys.use_relative = False
            else:
                return None
        
        # Does the shapekey exists?
        res = mesh.shape_keys.key_blocks.get(name)
        
        # No !
        if (res is None) and create:
            
            eval_time = mesh.shape_keys.eval_time 
            
            if step is not None:
                # Ensure the value is correct
                mesh.shape_keys.eval_time = step*10
            
            res = obj.shape_key_add(name=name)
            
            # Less impact as possible :-)
            mesh.shape_keys.eval_time = eval_time
            
        return res

    def sk_exists(self, name, step):
        return self.get_sk(name, step, create=False) is not None

    def on_sk(self, name, step=None):
    
        if not self.sk_exists(name, step):
            raise WrapException(f"The shape key '{self.sk_name(name, step)}' doesn't exist in object '{self.object.name}'!")
            
        mesh = self.object.data
    
        mesh.shape_keys.eval_time = self.get_sk(name, step).frame
        return mesh.shape_keys.eval_time

    def delete_sk(self, name=None, step=None):
        
        if self.object.data.shape_keys is None:
            return
        
        if name is None:
            self.object.shape_key_clear()
        else:
            key = self.get_sk(name, step)
            if key is not None:
                self.object.shape_key_remove(key)
                

    # ----------------------------------------------------------------------------------------------------
    # Short cut to eval time
    
    @property
    def eval_time(self):
        mesh = self.object.data
        if mesh.shape_keys is None:
            return 0.
        else:
            return mesh.shape_keys.eval_time 
    
    @eval_time.setter
    def eval_time(self, value):
        mesh = self.object.data
        if mesh.shape_keys is None:
            raise WrapException(
                f"WMeshObject eval_time error: no shape keys are defined for the object {self.object.name}"
                )
        else:
            mesh.shape_keys.eval_time = value
                
    # ----------------------------------------------------------------------------------------------------
    # Surface computation
        
    @property
    def surface(self):
        return self._surface
    
    @surface.setter
    def surface(self, value):
        self._surface = value
        if self._surface is not None:
            self.surface_init()
            
    def set_function(self, func, coords='XYZ'):
        
        # If the shape key named "Basis" exists, initialize the vertices from this surface
        basis = self.get_sk("Basis", create=True)
        count = len(basis.data)
        
        verts = np.empty(count*3, np.float)
        basis.data.foreach_get('co', verts)
        
        verts = verts.reshape(count, 3)
        self.surface_sk = "Deformed"
        self._surface   = Surface.FromVertices(verts, coords=coords, func=func)
        self.compute(None)
            
    def surface_init(self, t=None):
        if self._surface is None:
            return
        
        # The mesh
        mesh = self.object.data
        
        # Create the new geometry
        mesh.clear_geometry()
        
        # Vertices and faces
        mesh.from_pydata(self._surface.compute(t), [], self._surface.faces())
        
        # Create the uv layer
        uv_layer = mesh.uv_layers.new()

        # Rapid if uv are specified for all faces
        uvs = [uv_co for uv in self._surface.uvs() for uv_co in uv]
        
        for i, uv in enumerate(uv_layer.data):
            uv.uv = uvs[i]
        
        # Make sure it is ok
        mesh.update()
        mesh.validate()
            
    def compute(self, t):
        
        if self._surface is None:
            raise WrapException(
                f"WMesh compute error: no surface attribute is set to mesh object '{self.object.name}'"
                )
            
        # Compute the vertices
        verts = self._surface.compute(t)
        
        # The mesh
        mesh = self.object.data
        
        if len(verts) != len(mesh.vertices):
            self.surface_init()
        else:
            verts = verts.reshape(len(verts)*3)
            if self.surface_sk is None:
                mesh.vertices.foreach_set('co', verts)
            else:
                sk = self.get_sk(self.surface_sk, create=True)
                sk.data.foreach_set('co', verts)
            
    def compute_shapekeys(self, t0, t1, steps=10):
        
        steps = max(2, steps)
        dt = (t1-t0)/steps
        
        # Initial key to ensure everything is correctly initialized
        self.compute(t0)
        count = len(self.object.data.vertices)*3
        
        # Loop on shape keys
        for step in range(steps+1):
            sk = self.get_sk(name='Surface', step=step)
            sk.data.foreach_set('co', self._surface.compute(t0 + dt*step).reshape(count))    

# =============================================================================================================================
# Curve object wrapper

class WCurveObject(WObject):
    
    def __init__(self, name):
        obj = blender.get_object(name, 'CURVE')
        super().__init__(obj.name)
        
        self.wcurve   = wgen.WCurve(self)
        self.wsplines = wgen.WSplines(self)
        
    # ---------------------------------------------------------------------------
    # New Curve object
    
    @classmethod
    def New(cls, name, shape='BEZIER', **kwargs):
        
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
        obj.name = name
            
        return cls(obj.name)
    
        
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
        bc.length = self.object.path_duration
        return bc

    def set_function(self, f, t0, t1, count=100):
        self.wcurve.set_function(f, t0, t1, count=count)
        

# =============================================================================================================================
# Wrap a Blender object

def wrap(obj_or_name, create=None, collection=None, **kwargs):
    
    obj = blender.getcreate_object(obj_or_name, create, collection, **kwargs)

    if obj.type == 'MESH':
        return WMeshObject(obj.name)
    
    if obj.type == 'CURVE':
        return WCurveObject(obj.name)
    
    elif obj.type == 'EMPTY':
        return WEmpty(obj.name)
    
    else:
        return WObject(obj.name)

# =============================================================================================================================
# Keyframe curves

"""
        
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
            
"""
