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

from wrapanime.wrappers.generated_wrappers import WObject, WMesh, WCurve
from wrapanime.wrappers import generated_wrappers as wgen

import wrapanime.cython.bezier as wbez

import importlib
importlib.reload(geo)
importlib.reload(wgen)
importlib.reload(root)


# =============================================================================================================================
# To vector

def to_vector(value, dim=3):
    return root.to_array(value, [dim], f"(to_vector) single value or a {dim}-vector, not {value}")


# =============================================================================================================================
# Empty object wrapper

class WEmpty(WObject):
    def __init__(self, obj):
        obj = WObject.get_object(obj, 'EMPTY')
        super().__init__(obj)

# =============================================================================================================================
# Mesh object wrapper

class WMeshObject(WObject):
    def __init__(self, obj):

        obj = WObject.get_object(obj, 'MESH')

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
# Curve object wrapper

class WCurveObject(WObject):
    def __init__(self, obj):
        obj = WObject.get_object(obj, 'CURVE')
        self.wcurve = WCurve(self.data)
        
    @property
    def bezier_curve(self):
        splines = self.wcurve.wsplines.obj
        spline = splines[0]
        
        bp = spline.bezier_points
        
        count  = len(bp)
        
        points = np.zeros(count*3, np.float).reshape[count, 3]
        
        bp.foreach_get('co',           points[:, 0, :].reshape(count*3))
        bp.foreach_get('handle_left',  points[:, 1, :].reshape(count*3))
        bp.foreach_get('handle_right', points[:, 2, :].reshape(count*3))
        
        return points
    
    
    @bezier_curve.setter
    def bezier_curve(self, points):
    
        points = wbez.check_bezier_curve_points(points)
        count  = points.shape[0]
        
        splines = self.wcurve.wsplines.obj
        splines.clear()
        spline = splines.new('BEZIER')
        
        bp = spline.bezier_points
        
        bp.add(count-len(bp))
        
        bp.foreach_set('co',           points[:, 0, :].reshape(count*3))
        bp.foreach_set('handle_left',  points[:, 1, :].reshape(count*3))
        bp.foreach_set('handle_right', points[:, 2, :].reshape(count*3))
        

    def set_function(self, f, x0, x1, count=100):
        self.bezier_curve = wbez.Mapper.FromFunction(f, x0, x1, count=count).to_3D()

    def set_curve_function(self, f, t0, t1, count=100):
        self.bezier_curve = wbez.BezierCurve.FromFunction(f, t0, t1, count=count).points
        

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

