#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 09:42:35 2020

@author: alain
"""

import numpy as np

import bpy
from mathutils import Vector, Matrix, Quaternion, Euler

from wrapanime.functions.functions import Function

from wrapanime.utils import cy_object
from wrapanime.utils.errors import WrapException
from wrapanime.utils import geometry as geo
from wrapanime.root import root
from wrapanime.root import generated_wrappers as wgen

import importlib
importlib.reload(geo)
importlib.reload(wgen)
importlib.reload(root)


# =============================================================================================================================
# To vector

def to_vector(value, dim=3):
    return to_array(value, [dim], f"(to_vector) single value or a {dim}-vector, not {value}"

# =============================================================================================================================
# Mesh data

class WMesh(wgen.WMesh):
    pass

# =============================================================================================================================
# Curve data

class WCurve(wgen.WCurve):
    pass

# =============================================================================================================================
# Object Wrapper
# Root for other high level objects

class WObject(wgen.WObject):

    def __init__(self, obj):

        super().__init__(WObject.get_object(obj), None)
        self._eval_object = None
        self._averts      = None

    @classmethod
    def get_object(cls, obj):
        if type(obj) is str:
            o = bpy.data.objects.get(obj)
            if o is None:
                raise errors.WrapException("Impossible de initialize wrapper: Blender object '{}' not found".format(obj))
            obj = o
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
        return self.hide_render

    @hide.setter
    def hide(self, value):
        self.hide_render   = value
        self.hide_viewport = value

    # transformation

    def orient(self, axis):
        self.quaternion = geo.tracker_quaternion(self.track_axis, axis, up=self.up_axis)

    def track_to(self, location):
        self.quaternion = geo.tracker_quaternion(self.track_axis, Vector(location)-self.location, up=self.up_axis)

    # Distance

    def distance(self, location):
        return (Vector(location)-self.location).length

# =============================================================================================================================
# Empty object wrapper

class WEmpty(WObject):
    def __init__(self, obj):

        obj = WObject.get_object(obj)

        if obj.type != 'EMPTY':
            wa_utils.error("Empty object wrapper initialization error", obj, "Object {} type must be 'EMPTY' not '{}'.".format(obj.name, obj.type))

        super().__init__(obj)

# =============================================================================================================================
# Mesh object wrapper

class WMeshObject(WObject):
    def __init__(self, obj):

        obj = WObject.get_object(obj)

        if obj.type != 'MESH':
            wa_utils.error("Mesh object wrapper initialization error", obj, "Object {} type must be 'MESH' not '{}'.".format(obj.name, obj.type))

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
        vals = to_array(values, shape, "a single vector or an array of {} vectors".format(nverts))
        cy_object.set_vertices(self.obj.data, vals, mask='XYZ')
        self._verts = None

    def set_verts(self, verts, mask='XYZ'):
        nverts = len(self.wmesh.vertices)
        shape = (nverts, 3)
        vals = to_array(verts, shape, "a single vector or an array of {} vectors".format(nverts))
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
# Mesh object wrapper

class WCurveObject(WObject):
    def __init__(self, obj):

        obj = WObject.get_object(obj)

        if obj.type != 'CURVE':
            raise WrapException(
                "Curve object wrapper initialization error", obj,
                f"Object {obj.name} type must be 'CURVE' not '{obj.type}'."

        super().__init__(obj)
        self.wcurve = WCurve(self.obj.data, self)

    def set_bezier(self, values):
        vals = np.array(values)
        if len(vals.shape) != 3 and vals.shape[1:] != (3, 3)

# =============================================================================================================================
# Wrap a Blender object

def wrap(obj):
    obj = WObject.get_object(obj)

    if obj.type == 'MESH':
        return WMeshObject(obj)
    if obj.type == 'CURVE':
        return WCurveObject(obj)
    elif obj.type == 'EMPTY':
        return WEmpty(obj)
    else:
        raise WrapException(
            "Object not wrappable", obj,
            f"No wrapper is implemented for type '{obj.type}'")

# =============================================================================================================================
# Array of WMeshObject

class ArrayOfWObject(wgen.ArrayOfWObject):
    def wrap(self, obj):
        return wrap(obj)
