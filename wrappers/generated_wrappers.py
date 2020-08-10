# ****************************************************************************************************
# Generated 2020-08-10

import numpy as np
from wrapanime.wrappers.root import Wrapper, to_array, WObjectRoot, WMeshRoot, WSplineRoot, WSplinesRoot
from wrapanime.utils.errors import WrapException
import bpy

#================================================================================
# MeshVertex class wrapper

class WMeshVertex(Wrapper):

    @property
    def vertex(self): # The wrapped Blender vertex
        return self.root_wrapper.top_obj.data.vertices[self.windex]

    @property
    def bstruct(self): # The wrapped Blender vertex
        return self.root_wrapper.top_obj.data.vertices[self.windex]

    @property
    def bevel_weight(self): # float
        return self.vertex.bevel_weight

    @bevel_weight.setter
    def bevel_weight(self, value): # float
        self.vertex.bevel_weight = value

    @property
    def co(self): # V3
        return self.vertex.co

    @property
    def x(self):
        return self.vertex.co[0]

    @property
    def y(self):
        return self.vertex.co[1]

    @property
    def z(self):
        return self.vertex.co[2]

    @co.setter
    def co(self, value): # V3
        self.vertex.co = to_array(value, (3,), "co")

    @x.setter
    def x(self, value):
        self.vertex.co[0] = value

    @y.setter
    def y(self, value):
        self.vertex.co[1] = value

    @z.setter
    def z(self, value):
        self.vertex.co[2] = value

    @property
    def hide(self): # bool
        return self.vertex.hide

    @hide.setter
    def hide(self, value): # bool
        self.vertex.hide = value

    @property
    def index(self): # int
        return self.vertex.index

    @property
    def normal(self): # V3
        return self.vertex.normal

    @property
    def nx(self):
        return self.vertex.normal[0]

    @property
    def ny(self):
        return self.vertex.normal[1]

    @property
    def nz(self):
        return self.vertex.normal[2]

    @property
    def undeformed_co(self): # V3
        return self.vertex.undeformed_co

#================================================================================
# Array of WMeshVertices

class WMeshVertices(Wrapper):

    @property
    def vertices(self): # The wrapped Blender vertices
        return self.root_wrapper.top_obj.data.vertices

    def __len__(self):
        return len(self.root_wrapper.top_obj.data.vertices)

    def __getitem__(self, index):
        return WMeshVertex(self.root_wrapper, index=index)

    @property
    def item_class(self):
        return WMeshVertex

    @property
    def bevel_weights(self): # Array of float
        array = np.empty(len(self), np.float)
        self.root_wrapper.top_obj.data.vertices.foreach_get('bevel_weight', array)
        return array

    @bevel_weights.setter
    def bevel_weights(self, values): # Arrayf of float
        self.root_wrapper.top_obj.data.vertices.foreach_set('bevel_weight', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def cos(self): # Array of V3
        array = np.empty(len(self)*3, np.float)
        self.root_wrapper.top_obj.data.vertices.foreach_get('co', array)
        return array.reshape(len(self), 3)

    @cos.setter
    def cos(self, values): # Arrayf of V3
        self.root_wrapper.top_obj.data.vertices.foreach_set('co', to_array(values, (len(self), 3), f'3-vector or array of {len(self)} 3-vectors').reshape(len(self) * 3))

    # xyzw access to cos

    @property
    def xs(self): 
        return self.cos[:, 0]

    @xs.setter
    def xs(self, values):
        cos = self.cos
        cos[:, 0] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.cos = cos

    @property
    def ys(self): 
        return self.cos[:, 1]

    @ys.setter
    def ys(self, values):
        cos = self.cos
        cos[:, 1] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.cos = cos

    @property
    def zs(self): 
        return self.cos[:, 2]

    @zs.setter
    def zs(self, values):
        cos = self.cos
        cos[:, 2] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.cos = cos

    @property
    def hides(self): # Array of bool
        array = np.empty(len(self), np.bool)
        self.root_wrapper.top_obj.data.vertices.foreach_get('hide', array)
        return array

    @hides.setter
    def hides(self, values): # Arrayf of bool
        self.root_wrapper.top_obj.data.vertices.foreach_set('hide', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def indices(self): # Array of int
        array = np.empty(len(self), np.int)
        self.root_wrapper.top_obj.data.vertices.foreach_get('index', array)
        return array

    @property
    def normals(self): # Array of V3
        array = np.empty(len(self)*3, np.float)
        self.root_wrapper.top_obj.data.vertices.foreach_get('normal', array)
        return array.reshape(len(self), 3)

    # xyzw access to normals

    @property
    def nxs(self): 
        return self.normals[:, 0]

    @property
    def nys(self): 
        return self.normals[:, 1]

    @property
    def nzs(self): 
        return self.normals[:, 2]

    @property
    def undeformed_cos(self): # Array of V3
        array = np.empty(len(self)*3, np.float)
        self.root_wrapper.top_obj.data.vertices.foreach_get('undeformed_co', array)
        return array.reshape(len(self), 3)

#================================================================================
# Edge class wrapper

class WEdge(Wrapper):

    @property
    def edge(self): # The wrapped Blender edge
        return self.root_wrapper.top_obj.data.edges[self.windex]

    @property
    def bstruct(self): # The wrapped Blender edge
        return self.root_wrapper.top_obj.data.edges[self.windex]

    @property
    def bevel_weight(self): # float
        return self.edge.bevel_weight

    @bevel_weight.setter
    def bevel_weight(self, value): # float
        self.edge.bevel_weight = value

    @property
    def crease(self): # float
        return self.edge.crease

    @crease.setter
    def crease(self, value): # float
        self.edge.crease = value

    @property
    def hide(self): # bool
        return self.edge.hide

    @hide.setter
    def hide(self, value): # bool
        self.edge.hide = value

    @property
    def index(self): # int
        return self.edge.index

    @property
    def is_loose(self): # bool
        return self.edge.is_loose

    @property
    def select(self): # bool
        return self.edge.select

    @select.setter
    def select(self, value): # bool
        self.edge.select = value

    @property
    def use_edge_sharp(self): # bool
        return self.edge.use_edge_sharp

    @use_edge_sharp.setter
    def use_edge_sharp(self, value): # bool
        self.edge.use_edge_sharp = value

    @property
    def use_seam(self): # bool
        return self.edge.use_seam

    @use_seam.setter
    def use_seam(self, value): # bool
        self.edge.use_seam = value

    @property
    def vertices(self): # array
        return self.edge.vertices

#================================================================================
# Array of WEdges

class WEdges(Wrapper):

    @property
    def edges(self): # The wrapped Blender edges
        return self.root_wrapper.top_obj.data.edges

    def __len__(self):
        return len(self.root_wrapper.top_obj.data.edges)

    def __getitem__(self, index):
        return WEdge(self.root_wrapper, index=index)

    @property
    def item_class(self):
        return WEdge

    @property
    def bevel_weights(self): # Array of float
        array = np.empty(len(self), np.float)
        self.root_wrapper.top_obj.data.edges.foreach_get('bevel_weight', array)
        return array

    @bevel_weights.setter
    def bevel_weights(self, values): # Arrayf of float
        self.root_wrapper.top_obj.data.edges.foreach_set('bevel_weight', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def creases(self): # Array of float
        array = np.empty(len(self), np.float)
        self.root_wrapper.top_obj.data.edges.foreach_get('crease', array)
        return array

    @creases.setter
    def creases(self, values): # Arrayf of float
        self.root_wrapper.top_obj.data.edges.foreach_set('crease', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def hides(self): # Array of bool
        array = np.empty(len(self), np.bool)
        self.root_wrapper.top_obj.data.edges.foreach_get('hide', array)
        return array

    @hides.setter
    def hides(self, values): # Arrayf of bool
        self.root_wrapper.top_obj.data.edges.foreach_set('hide', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def indices(self): # Array of int
        array = np.empty(len(self), np.int)
        self.root_wrapper.top_obj.data.edges.foreach_get('index', array)
        return array

    @property
    def is_looses(self): # Array of bool
        array = np.empty(len(self), np.bool)
        self.root_wrapper.top_obj.data.edges.foreach_get('is_loose', array)
        return array

    @property
    def selects(self): # Array of bool
        array = np.empty(len(self), np.bool)
        self.root_wrapper.top_obj.data.edges.foreach_get('select', array)
        return array

    @selects.setter
    def selects(self, values): # Arrayf of bool
        self.root_wrapper.top_obj.data.edges.foreach_set('select', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def use_edge_sharps(self): # Array of bool
        array = np.empty(len(self), np.bool)
        self.root_wrapper.top_obj.data.edges.foreach_get('use_edge_sharp', array)
        return array

    @use_edge_sharps.setter
    def use_edge_sharps(self, values): # Arrayf of bool
        self.root_wrapper.top_obj.data.edges.foreach_set('use_edge_sharp', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def use_seams(self): # Array of bool
        array = np.empty(len(self), np.bool)
        self.root_wrapper.top_obj.data.edges.foreach_get('use_seam', array)
        return array

    @use_seams.setter
    def use_seams(self, values): # Arrayf of bool
        self.root_wrapper.top_obj.data.edges.foreach_set('use_seam', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

#================================================================================
# Loop class wrapper

class WLoop(Wrapper):

    @property
    def loop(self): # The wrapped Blender loop
        return self.root_wrapper.top_obj.data.loops[self.windex]

    @property
    def bstruct(self): # The wrapped Blender loop
        return self.root_wrapper.top_obj.data.loops[self.windex]

    @property
    def bitangent_sign(self): # float
        return self.loop.bitangent_sign

    @property
    def bitangent(self): # V3
        return self.loop.bitangent

    @property
    def edge_index(self): # int
        return self.loop.edge_index

    @property
    def index(self): # int
        return self.loop.index

    @property
    def normal(self): # V3
        return self.loop.normal

    @property
    def tangent(self): # V3
        return self.loop.tangent

    @property
    def vertex_index(self): # int
        return self.loop.vertex_index

#================================================================================
# Array of WLoops

class WLoops(Wrapper):

    @property
    def loops(self): # The wrapped Blender loops
        return self.root_wrapper.top_obj.data.loops

    def __len__(self):
        return len(self.root_wrapper.top_obj.data.loops)

    def __getitem__(self, index):
        return WLoop(self.root_wrapper, index=index)

    @property
    def item_class(self):
        return WLoop

    @property
    def bitangent_signs(self): # Array of float
        array = np.empty(len(self), np.float)
        self.root_wrapper.top_obj.data.loops.foreach_get('bitangent_sign', array)
        return array

    @property
    def bitangents(self): # Array of V3
        array = np.empty(len(self)*3, np.float)
        self.root_wrapper.top_obj.data.loops.foreach_get('bitangent', array)
        return array.reshape(len(self), 3)

    @property
    def edge_indices(self): # Array of int
        array = np.empty(len(self), np.int)
        self.root_wrapper.top_obj.data.loops.foreach_get('edge_index', array)
        return array

    @property
    def indices(self): # Array of int
        array = np.empty(len(self), np.int)
        self.root_wrapper.top_obj.data.loops.foreach_get('index', array)
        return array

    @property
    def normals(self): # Array of V3
        array = np.empty(len(self)*3, np.float)
        self.root_wrapper.top_obj.data.loops.foreach_get('normal', array)
        return array.reshape(len(self), 3)

    @property
    def tangents(self): # Array of V3
        array = np.empty(len(self)*3, np.float)
        self.root_wrapper.top_obj.data.loops.foreach_get('tangent', array)
        return array.reshape(len(self), 3)

    @property
    def vertex_indices(self): # Array of int
        array = np.empty(len(self), np.int)
        self.root_wrapper.top_obj.data.loops.foreach_get('vertex_index', array)
        return array

#================================================================================
# Polygon class wrapper

class WPolygon(Wrapper):

    @property
    def polygons(self): # The wrapped Blender polygons
        return self.root_wrapper.top_obj.data.polygons[self.windex]

    @property
    def bstruct(self): # The wrapped Blender polygons
        return self.root_wrapper.top_obj.data.polygons[self.windex]

    @property
    def area(self): # float
        return self.polygons.area

    @property
    def center(self): # V3
        return self.polygons.center

    @property
    def hide(self): # bool
        return self.polygons.hide

    @hide.setter
    def hide(self, value): # bool
        self.polygons.hide = value

    @property
    def index(self): # int
        return self.polygons.index

    @property
    def loop_start(self): # int
        return self.polygons.loop_start

    @property
    def loop_total(self): # int
        return self.polygons.loop_total

    @property
    def material_index(self): # int
        return self.polygons.material_index

    @material_index.setter
    def material_index(self, value): # int
        self.polygons.material_index = value

    @property
    def normal(self): # V3
        return self.polygons.normal

    @property
    def use_smooth(self): # bool
        return self.polygons.use_smooth

    @use_smooth.setter
    def use_smooth(self, value): # bool
        self.polygons.use_smooth = value

    @property
    def vertices(self): # array
        return self.polygons.vertices

#================================================================================
# Array of WPolygons

class WPolygons(Wrapper):

    @property
    def polygons(self): # The wrapped Blender polygons
        return self.root_wrapper.top_obj.data.polygons

    def __len__(self):
        return len(self.root_wrapper.top_obj.data.polygons)

    def __getitem__(self, index):
        return WPolygon(self.root_wrapper, index=index)

    @property
    def item_class(self):
        return WPolygon

    @property
    def areas(self): # Array of float
        array = np.empty(len(self), np.float)
        self.root_wrapper.top_obj.data.polygons.foreach_get('area', array)
        return array

    @property
    def centers(self): # Array of V3
        array = np.empty(len(self)*3, np.float)
        self.root_wrapper.top_obj.data.polygons.foreach_get('center', array)
        return array.reshape(len(self), 3)

    @property
    def hides(self): # Array of bool
        array = np.empty(len(self), np.bool)
        self.root_wrapper.top_obj.data.polygons.foreach_get('hide', array)
        return array

    @hides.setter
    def hides(self, values): # Arrayf of bool
        self.root_wrapper.top_obj.data.polygons.foreach_set('hide', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def indices(self): # Array of int
        array = np.empty(len(self), np.int)
        self.root_wrapper.top_obj.data.polygons.foreach_get('index', array)
        return array

    @property
    def loop_starts(self): # Array of int
        array = np.empty(len(self), np.int)
        self.root_wrapper.top_obj.data.polygons.foreach_get('loop_start', array)
        return array

    @property
    def loop_totals(self): # Array of int
        array = np.empty(len(self), np.int)
        self.root_wrapper.top_obj.data.polygons.foreach_get('loop_total', array)
        return array

    @property
    def material_indices(self): # Array of int
        array = np.empty(len(self), np.int)
        self.root_wrapper.top_obj.data.polygons.foreach_get('material_index', array)
        return array

    @material_indices.setter
    def material_indices(self, values): # Arrayf of int
        self.root_wrapper.top_obj.data.polygons.foreach_set('material_index', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def normals(self): # Array of V3
        array = np.empty(len(self)*3, np.float)
        self.root_wrapper.top_obj.data.polygons.foreach_get('normal', array)
        return array.reshape(len(self), 3)

    @property
    def use_smooths(self): # Array of bool
        array = np.empty(len(self), np.bool)
        self.root_wrapper.top_obj.data.polygons.foreach_get('use_smooth', array)
        return array

    @use_smooths.setter
    def use_smooths(self, values): # Arrayf of bool
        self.root_wrapper.top_obj.data.polygons.foreach_set('use_smooth', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

#================================================================================
# Mesh class wrapper

class WMesh(WMeshRoot):

    def __init__(self, root_wrapper):
        super().__init__(root_wrapper)
        self.wvertices = WMeshVertices(self.root_wrapper)
        self.wedges    = WEdges(self.root_wrapper)
        self.wloops    = WLoops(self.root_wrapper)
        self.wpolygons = WPolygons(self.root_wrapper)

    @property
    def mesh(self): # The wrapped Blender mesh
        return self.root_wrapper.top_obj.data

    @property
    def bstruct(self): # The wrapped Blender mesh
        return self.root_wrapper.top_obj.data

    @property
    def auto_smooth_angle(self): # float
        return self.mesh.auto_smooth_angle

    @auto_smooth_angle.setter
    def auto_smooth_angle(self, value): # float
        self.mesh.auto_smooth_angle = value

    @property
    def auto_texspace(self): # bool
        return self.mesh.auto_texspace

    @auto_texspace.setter
    def auto_texspace(self, value): # bool
        self.mesh.auto_texspace = value

    @property
    def use_auto_smooth(self): # bool
        return self.mesh.use_auto_smooth

    @use_auto_smooth.setter
    def use_auto_smooth(self, value): # bool
        self.mesh.use_auto_smooth = value

    @property
    def use_auto_texspace(self): # bool
        return self.mesh.use_auto_texspace

    @use_auto_texspace.setter
    def use_auto_texspace(self, value): # bool
        self.mesh.use_auto_texspace = value

#================================================================================
# BezierSplinePoint class wrapper

class WBezierSplinePoint(Wrapper):

    @property
    def bezier_point(self): # The wrapped Blender bezier_point
        return self.root_wrapper.top_obj.data.splines[self.wowner.windex].bezier_points[self.windex]

    @property
    def bstruct(self): # The wrapped Blender bezier_point
        return self.root_wrapper.top_obj.data.splines[self.wowner.windex].bezier_points[self.windex]

    @property
    def co(self): # V3
        return self.bezier_point.co

    @property
    def x(self):
        return self.bezier_point.co[0]

    @property
    def y(self):
        return self.bezier_point.co[1]

    @property
    def z(self):
        return self.bezier_point.co[2]

    @co.setter
    def co(self, value): # V3
        self.bezier_point.co = to_array(value, (3,), "co")

    @x.setter
    def x(self, value):
        self.bezier_point.co[0] = value

    @y.setter
    def y(self, value):
        self.bezier_point.co[1] = value

    @z.setter
    def z(self, value):
        self.bezier_point.co[2] = value

    @property
    def handle_left_type(self): # str
        return self.bezier_point.handle_left_type

    @handle_left_type.setter
    def handle_left_type(self, value): # str
        self.bezier_point.handle_left_type = value

    @property
    def handle_left(self): # V3
        return self.bezier_point.handle_left

    @property
    def lx(self):
        return self.bezier_point.handle_left[0]

    @property
    def ly(self):
        return self.bezier_point.handle_left[1]

    @property
    def lz(self):
        return self.bezier_point.handle_left[2]

    @handle_left.setter
    def handle_left(self, value): # V3
        self.bezier_point.handle_left = to_array(value, (3,), "handle_left")

    @lx.setter
    def lx(self, value):
        self.bezier_point.handle_left[0] = value

    @ly.setter
    def ly(self, value):
        self.bezier_point.handle_left[1] = value

    @lz.setter
    def lz(self, value):
        self.bezier_point.handle_left[2] = value

    @property
    def handle_right_type(self): # str
        return self.bezier_point.handle_right_type

    @handle_right_type.setter
    def handle_right_type(self, value): # str
        self.bezier_point.handle_right_type = value

    @property
    def handle_right(self): # V3
        return self.bezier_point.handle_right

    @property
    def rx(self):
        return self.bezier_point.handle_right[0]

    @property
    def ry(self):
        return self.bezier_point.handle_right[1]

    @property
    def rz(self):
        return self.bezier_point.handle_right[2]

    @handle_right.setter
    def handle_right(self, value): # V3
        self.bezier_point.handle_right = to_array(value, (3,), "handle_right")

    @rx.setter
    def rx(self, value):
        self.bezier_point.handle_right[0] = value

    @ry.setter
    def ry(self, value):
        self.bezier_point.handle_right[1] = value

    @rz.setter
    def rz(self, value):
        self.bezier_point.handle_right[2] = value

    @property
    def hide(self): # bool
        return self.bezier_point.hide

    @hide.setter
    def hide(self, value): # bool
        self.bezier_point.hide = value

    @property
    def radius(self): # float
        return self.bezier_point.radius

    @radius.setter
    def radius(self, value): # float
        self.bezier_point.radius = value

    @property
    def tilt(self): # int
        return self.bezier_point.tilt

    @tilt.setter
    def tilt(self, value): # int
        self.bezier_point.tilt = value

    @property
    def weight_softbody(self): # float
        return self.bezier_point.weight_softbody

    @weight_softbody.setter
    def weight_softbody(self, value): # float
        self.bezier_point.weight_softbody = value

#================================================================================
# Array of WBezierSplinePoints

class WBezierSplinePoints(Wrapper):

    @property
    def bezier_points(self): # The wrapped Blender bezier_points
        return self.root_wrapper.top_obj.data.splines[self.wowner.windex].bezier_points

    def __len__(self):
        return len(self.root_wrapper.top_obj.data.splines[self.wowner.windex].bezier_points)

    def __getitem__(self, index):
        return WBezierSplinePoint(self.root_wrapper, wowner=self, index=index)

    @property
    def item_class(self):
        return WBezierSplinePoint

    @property
    def cos(self): # Array of V3
        array = np.empty(len(self)*3, np.float)
        self.root_wrapper.top_obj.data.splines[self.wowner.windex].bezier_points.foreach_get('co', array)
        return array.reshape(len(self), 3)

    @cos.setter
    def cos(self, values): # Arrayf of V3
        self.root_wrapper.top_obj.data.splines[self.wowner.windex].bezier_points.foreach_set('co', to_array(values, (len(self), 3), f'3-vector or array of {len(self)} 3-vectors').reshape(len(self) * 3))

    # xyzw access to cos

    @property
    def xs(self): 
        return self.cos[:, 0]

    @xs.setter
    def xs(self, values):
        cos = self.cos
        cos[:, 0] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.cos = cos

    @property
    def ys(self): 
        return self.cos[:, 1]

    @ys.setter
    def ys(self, values):
        cos = self.cos
        cos[:, 1] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.cos = cos

    @property
    def zs(self): 
        return self.cos[:, 2]

    @zs.setter
    def zs(self, values):
        cos = self.cos
        cos[:, 2] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.cos = cos

    @property
    def handle_left_types(self): # Array of str
        array = np.empty(len(self), np.object)
        coll = self.root_wrapper.top_obj.data.splines[self.wowner.windex].bezier_points
        for i in range(len(self)):
            array[i] = coll[i].handle_left_type
        return array

    @handle_left_types.setter
    def handle_left_types(self, values): # Arrayf of str
        array = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        coll = self.root_wrapper.top_obj.data.splines[self.wowner.windex].bezier_points
        for i in range(len(self)):
            coll[i].handle_left_type = array[i]

    @property
    def handle_lefts(self): # Array of V3
        array = np.empty(len(self)*3, np.float)
        self.root_wrapper.top_obj.data.splines[self.wowner.windex].bezier_points.foreach_get('handle_left', array)
        return array.reshape(len(self), 3)

    @handle_lefts.setter
    def handle_lefts(self, values): # Arrayf of V3
        self.root_wrapper.top_obj.data.splines[self.wowner.windex].bezier_points.foreach_set('handle_left', to_array(values, (len(self), 3), f'3-vector or array of {len(self)} 3-vectors').reshape(len(self) * 3))

    # xyzw access to handle_lefts

    @property
    def lxs(self): 
        return self.handle_lefts[:, 0]

    @lxs.setter
    def lxs(self, values):
        handle_lefts = self.handle_lefts
        handle_lefts[:, 0] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.handle_lefts = handle_lefts

    @property
    def lys(self): 
        return self.handle_lefts[:, 1]

    @lys.setter
    def lys(self, values):
        handle_lefts = self.handle_lefts
        handle_lefts[:, 1] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.handle_lefts = handle_lefts

    @property
    def lzs(self): 
        return self.handle_lefts[:, 2]

    @lzs.setter
    def lzs(self, values):
        handle_lefts = self.handle_lefts
        handle_lefts[:, 2] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.handle_lefts = handle_lefts

    @property
    def handle_right_types(self): # Array of str
        array = np.empty(len(self), np.object)
        coll = self.root_wrapper.top_obj.data.splines[self.wowner.windex].bezier_points
        for i in range(len(self)):
            array[i] = coll[i].handle_right_type
        return array

    @handle_right_types.setter
    def handle_right_types(self, values): # Arrayf of str
        array = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        coll = self.root_wrapper.top_obj.data.splines[self.wowner.windex].bezier_points
        for i in range(len(self)):
            coll[i].handle_right_type = array[i]

    @property
    def handle_rights(self): # Array of V3
        array = np.empty(len(self)*3, np.float)
        self.root_wrapper.top_obj.data.splines[self.wowner.windex].bezier_points.foreach_get('handle_right', array)
        return array.reshape(len(self), 3)

    @handle_rights.setter
    def handle_rights(self, values): # Arrayf of V3
        self.root_wrapper.top_obj.data.splines[self.wowner.windex].bezier_points.foreach_set('handle_right', to_array(values, (len(self), 3), f'3-vector or array of {len(self)} 3-vectors').reshape(len(self) * 3))

    # xyzw access to handle_rights

    @property
    def rxs(self): 
        return self.handle_rights[:, 0]

    @rxs.setter
    def rxs(self, values):
        handle_rights = self.handle_rights
        handle_rights[:, 0] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.handle_rights = handle_rights

    @property
    def rys(self): 
        return self.handle_rights[:, 1]

    @rys.setter
    def rys(self, values):
        handle_rights = self.handle_rights
        handle_rights[:, 1] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.handle_rights = handle_rights

    @property
    def rzs(self): 
        return self.handle_rights[:, 2]

    @rzs.setter
    def rzs(self, values):
        handle_rights = self.handle_rights
        handle_rights[:, 2] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.handle_rights = handle_rights

    @property
    def hides(self): # Array of bool
        array = np.empty(len(self), np.bool)
        self.root_wrapper.top_obj.data.splines[self.wowner.windex].bezier_points.foreach_get('hide', array)
        return array

    @hides.setter
    def hides(self, values): # Arrayf of bool
        self.root_wrapper.top_obj.data.splines[self.wowner.windex].bezier_points.foreach_set('hide', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def radius_s(self): # Array of float
        array = np.empty(len(self), np.float)
        self.root_wrapper.top_obj.data.splines[self.wowner.windex].bezier_points.foreach_get('radius', array)
        return array

    @radius_s.setter
    def radius_s(self, values): # Arrayf of float
        self.root_wrapper.top_obj.data.splines[self.wowner.windex].bezier_points.foreach_set('radius', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def tilts(self): # Array of int
        array = np.empty(len(self), np.int)
        self.root_wrapper.top_obj.data.splines[self.wowner.windex].bezier_points.foreach_get('tilt', array)
        return array

    @tilts.setter
    def tilts(self, values): # Arrayf of int
        self.root_wrapper.top_obj.data.splines[self.wowner.windex].bezier_points.foreach_set('tilt', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def weight_softbodys(self): # Array of float
        array = np.empty(len(self), np.float)
        self.root_wrapper.top_obj.data.splines[self.wowner.windex].bezier_points.foreach_get('weight_softbody', array)
        return array

    @weight_softbodys.setter
    def weight_softbodys(self, values): # Arrayf of float
        self.root_wrapper.top_obj.data.splines[self.wowner.windex].bezier_points.foreach_set('weight_softbody', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def bpoints(self):
        points = np.empty((len(self), 3*3), np.float).reshape(len(self), 3, 3)
        points[:, 0] = self.handle_lefts
        points[:, 1] = self.cos
        points[:, 2] = self.handle_rights
        return points.reshape(len(self), 3*3)
    
    @bpoints.setter
    def bpoints(self, bpoints):
        a = np.array(bpoints)
        if a.size != len(self)*3*3:
            raise WrapException(
                    "Set Bezier points error: the length of the points array is incorrect",
                    f"Need: {len(self)} triplets of 3-vectors: {len(self)*3*3}",
                    f"Received: array {a.shape} of size {a.size}"
                    )
            
        np.reshape(a, (len(self), 3, 3))
        self.handle_lefts  = a[:, 0]
        self.cos           = a[:, 1]
        self.handle_rights = a[:, 2]
    
#================================================================================
# SplinePoint class wrapper

class WSplinePoint(Wrapper):

    @property
    def point(self): # The wrapped Blender point
        return self.root_wrapper.top_obj.data.splines[self.wowner.windex].points[self.windex]

    @property
    def bstruct(self): # The wrapped Blender point
        return self.root_wrapper.top_obj.data.splines[self.wowner.windex].points[self.windex]

    @property
    def co(self): # V4
        return self.point.co

    @property
    def x(self):
        return self.point.co[0]

    @property
    def y(self):
        return self.point.co[1]

    @property
    def z(self):
        return self.point.co[2]

    @property
    def w(self):
        return self.point.co[3]

    @co.setter
    def co(self, value): # V4
        self.point.co = to_array(value, (4,), "co")

    @x.setter
    def x(self, value):
        self.point.co[0] = value

    @y.setter
    def y(self, value):
        self.point.co[1] = value

    @z.setter
    def z(self, value):
        self.point.co[2] = value

    @w.setter
    def w(self, value):
        self.point.co[3] = value

    @property
    def radius(self): # float
        return self.point.radius

    @radius.setter
    def radius(self, value): # float
        self.point.radius = value

    @property
    def tilt(self): # float
        return self.point.tilt

    @tilt.setter
    def tilt(self, value): # float
        self.point.tilt = value

    @property
    def weight_softbody(self): # float
        return self.point.weight_softbody

    @weight_softbody.setter
    def weight_softbody(self, value): # float
        self.point.weight_softbody = value

    @property
    def weight(self): # float
        return self.point.weight

    @weight.setter
    def weight(self, value): # float
        self.point.weight = value

#================================================================================
# Array of WSplinePoints

class WSplinePoints(Wrapper):

    @property
    def points(self): # The wrapped Blender points
        return self.root_wrapper.top_obj.data.splines[self.wowner.windex].points

    def __len__(self):
        return len(self.root_wrapper.top_obj.data.splines[self.wowner.windex].points)

    def __getitem__(self, index):
        return WSplinePoint(self.root_wrapper, wowner=self, index=index)

    @property
    def item_class(self):
        return WSplinePoint

    @property
    def cos(self): # Array of V4
        array = np.empty(len(self)*4, np.float)
        self.root_wrapper.top_obj.data.splines[self.wowner.windex].points.foreach_get('co', array)
        return array.reshape(len(self), 4)

    @cos.setter
    def cos(self, values): # Arrayf of V4
        self.root_wrapper.top_obj.data.splines[self.wowner.windex].points.foreach_set('co', to_array(values, (len(self), 4), f'4-vector or array of {len(self)} 4-vectors').reshape(len(self) * 4))

    # xyzw access to cos

    @property
    def xs(self): 
        return self.cos[:, 0]

    @xs.setter
    def xs(self, values):
        cos = self.cos
        cos[:, 0] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.cos = cos

    @property
    def ys(self): 
        return self.cos[:, 1]

    @ys.setter
    def ys(self, values):
        cos = self.cos
        cos[:, 1] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.cos = cos

    @property
    def zs(self): 
        return self.cos[:, 2]

    @zs.setter
    def zs(self, values):
        cos = self.cos
        cos[:, 2] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.cos = cos

    @property
    def ws(self): 
        return self.cos[:, 3]

    @ws.setter
    def ws(self, values):
        cos = self.cos
        cos[:, 3] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.cos = cos

    @property
    def radius_s(self): # Array of float
        array = np.empty(len(self), np.float)
        self.root_wrapper.top_obj.data.splines[self.wowner.windex].points.foreach_get('radius', array)
        return array

    @radius_s.setter
    def radius_s(self, values): # Arrayf of float
        self.root_wrapper.top_obj.data.splines[self.wowner.windex].points.foreach_set('radius', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def tilts(self): # Array of float
        array = np.empty(len(self), np.float)
        self.root_wrapper.top_obj.data.splines[self.wowner.windex].points.foreach_get('tilt', array)
        return array

    @tilts.setter
    def tilts(self, values): # Arrayf of float
        self.root_wrapper.top_obj.data.splines[self.wowner.windex].points.foreach_set('tilt', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def weight_softbodys(self): # Array of float
        array = np.empty(len(self), np.float)
        self.root_wrapper.top_obj.data.splines[self.wowner.windex].points.foreach_get('weight_softbody', array)
        return array

    @weight_softbodys.setter
    def weight_softbodys(self, values): # Arrayf of float
        self.root_wrapper.top_obj.data.splines[self.wowner.windex].points.foreach_set('weight_softbody', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def weights(self): # Array of float
        array = np.empty(len(self), np.float)
        self.root_wrapper.top_obj.data.splines[self.wowner.windex].points.foreach_get('weight', array)
        return array

    @weights.setter
    def weights(self, values): # Arrayf of float
        self.root_wrapper.top_obj.data.splines[self.wowner.windex].points.foreach_set('weight', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

#================================================================================
# Spline class wrapper

class WSpline(WSplineRoot):

    def __init__(self, root_wrapper, index):
        super().__init__(root_wrapper, index=index)
        self.wbezier_points = WBezierSplinePoints(self.root_wrapper, self, index)
        self.wpoints        = WSplinePoints(self.root_wrapper, self, index)

    @property
    def spline(self): # The wrapped Blender spline
        return self.root_wrapper.top_obj.data.splines[self.windex]

    @property
    def bstruct(self): # The wrapped Blender spline
        return self.root_wrapper.top_obj.data.splines[self.windex]

    @property
    def character_index(self): # int
        return self.spline.character_index

    @character_index.setter
    def character_index(self, value): # int
        self.spline.character_index = value

    @property
    def material_index(self): # int
        return self.spline.material_index

    @material_index.setter
    def material_index(self, value): # int
        self.spline.material_index = value

    @property
    def order_u(self): # int
        return self.spline.order_u

    @order_u.setter
    def order_u(self, value): # int
        self.spline.order_u = value

    @property
    def order_v(self): # int
        return self.spline.order_v

    @order_v.setter
    def order_v(self, value): # int
        self.spline.order_v = value

    @property
    def point_count_u(self): # int
        return self.spline.point_count_u

    @point_count_u.setter
    def point_count_u(self, value): # int
        self.spline.point_count_u = value

    @property
    def point_count_v(self): # int
        return self.spline.point_count_v

    @point_count_v.setter
    def point_count_v(self, value): # int
        self.spline.point_count_v = value

    @property
    def radius_interpolation(self): # str
        return self.spline.radius_interpolation

    @radius_interpolation.setter
    def radius_interpolation(self, value): # str
        self.spline.radius_interpolation = value

    @property
    def resolution_u(self): # int
        return self.spline.resolution_u

    @resolution_u.setter
    def resolution_u(self, value): # int
        self.spline.resolution_u = value

    @property
    def resolution_v(self): # int
        return self.spline.resolution_v

    @resolution_v.setter
    def resolution_v(self, value): # int
        self.spline.resolution_v = value

    @property
    def type(self): # str
        return self.spline.type

    @type.setter
    def type(self, value): # str
        self.spline.type = value

    @property
    def use_bezier_u(self): # bool
        return self.spline.use_bezier_u

    @use_bezier_u.setter
    def use_bezier_u(self, value): # bool
        self.spline.use_bezier_u = value

    @property
    def use_bezier_v(self): # bool
        return self.spline.use_bezier_v

    @use_bezier_v.setter
    def use_bezier_v(self, value): # bool
        self.spline.use_bezier_v = value

    @property
    def use_cyclic_u(self): # bool
        return self.spline.use_cyclic_u

    @use_cyclic_u.setter
    def use_cyclic_u(self, value): # bool
        self.spline.use_cyclic_u = value

    @property
    def use_cyclic_v(self): # bool
        return self.spline.use_cyclic_v

    @use_cyclic_v.setter
    def use_cyclic_v(self, value): # bool
        self.spline.use_cyclic_v = value

    @property
    def use_endpoint_u(self): # bool
        return self.spline.use_endpoint_u

    @use_endpoint_u.setter
    def use_endpoint_u(self, value): # bool
        self.spline.use_endpoint_u = value

    @property
    def use_endpoint_v(self): # bool
        return self.spline.use_endpoint_v

    @use_endpoint_v.setter
    def use_endpoint_v(self, value): # bool
        self.spline.use_endpoint_v = value

    @property
    def use_smooth(self): # bool
        return self.spline.use_smooth

    @use_smooth.setter
    def use_smooth(self, value): # bool
        self.spline.use_smooth = value

#================================================================================
# Array of WSplines

class WSplines(WSplinesRoot):

    @property
    def splines(self): # The wrapped Blender splines
        return self.root_wrapper.top_obj.data.splines

    def __len__(self):
        return len(self.root_wrapper.top_obj.data.splines)

    def __getitem__(self, index):
        return WSpline(self.root_wrapper, index=index)

    @property
    def item_class(self):
        return WSpline

    @property
    def character_indices(self): # Array of int
        array = np.empty(len(self), np.int)
        self.root_wrapper.top_obj.data.splines.foreach_get('character_index', array)
        return array

    @character_indices.setter
    def character_indices(self, values): # Arrayf of int
        self.root_wrapper.top_obj.data.splines.foreach_set('character_index', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def material_indices(self): # Array of int
        array = np.empty(len(self), np.int)
        self.root_wrapper.top_obj.data.splines.foreach_get('material_index', array)
        return array

    @material_indices.setter
    def material_indices(self, values): # Arrayf of int
        self.root_wrapper.top_obj.data.splines.foreach_set('material_index', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def order_us(self): # Array of int
        array = np.empty(len(self), np.int)
        self.root_wrapper.top_obj.data.splines.foreach_get('order_u', array)
        return array

    @order_us.setter
    def order_us(self, values): # Arrayf of int
        self.root_wrapper.top_obj.data.splines.foreach_set('order_u', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def order_vs(self): # Array of int
        array = np.empty(len(self), np.int)
        self.root_wrapper.top_obj.data.splines.foreach_get('order_v', array)
        return array

    @order_vs.setter
    def order_vs(self, values): # Arrayf of int
        self.root_wrapper.top_obj.data.splines.foreach_set('order_v', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def point_count_us(self): # Array of int
        array = np.empty(len(self), np.int)
        self.root_wrapper.top_obj.data.splines.foreach_get('point_count_u', array)
        return array

    @point_count_us.setter
    def point_count_us(self, values): # Arrayf of int
        self.root_wrapper.top_obj.data.splines.foreach_set('point_count_u', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def point_count_vs(self): # Array of int
        array = np.empty(len(self), np.int)
        self.root_wrapper.top_obj.data.splines.foreach_get('point_count_v', array)
        return array

    @point_count_vs.setter
    def point_count_vs(self, values): # Arrayf of int
        self.root_wrapper.top_obj.data.splines.foreach_set('point_count_v', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def radius_interpolations(self): # Array of str
        array = np.empty(len(self), np.object)
        coll = self.root_wrapper.top_obj.data.splines
        for i in range(len(self)):
            array[i] = coll[i].radius_interpolation
        return array

    @radius_interpolations.setter
    def radius_interpolations(self, values): # Arrayf of str
        array = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        coll = self.root_wrapper.top_obj.data.splines
        for i in range(len(self)):
            coll[i].radius_interpolation = array[i]

    @property
    def resolution_us(self): # Array of int
        array = np.empty(len(self), np.int)
        self.root_wrapper.top_obj.data.splines.foreach_get('resolution_u', array)
        return array

    @resolution_us.setter
    def resolution_us(self, values): # Arrayf of int
        self.root_wrapper.top_obj.data.splines.foreach_set('resolution_u', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def resolution_vs(self): # Array of int
        array = np.empty(len(self), np.int)
        self.root_wrapper.top_obj.data.splines.foreach_get('resolution_v', array)
        return array

    @resolution_vs.setter
    def resolution_vs(self, values): # Arrayf of int
        self.root_wrapper.top_obj.data.splines.foreach_set('resolution_v', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def types(self): # Array of str
        array = np.empty(len(self), np.object)
        coll = self.root_wrapper.top_obj.data.splines
        for i in range(len(self)):
            array[i] = coll[i].type
        return array

    @types.setter
    def types(self, values): # Arrayf of str
        array = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        coll = self.root_wrapper.top_obj.data.splines
        for i in range(len(self)):
            coll[i].type = array[i]

    @property
    def use_bezier_us(self): # Array of bool
        array = np.empty(len(self), np.bool)
        self.root_wrapper.top_obj.data.splines.foreach_get('use_bezier_u', array)
        return array

    @use_bezier_us.setter
    def use_bezier_us(self, values): # Arrayf of bool
        self.root_wrapper.top_obj.data.splines.foreach_set('use_bezier_u', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def use_bezier_vs(self): # Array of bool
        array = np.empty(len(self), np.bool)
        self.root_wrapper.top_obj.data.splines.foreach_get('use_bezier_v', array)
        return array

    @use_bezier_vs.setter
    def use_bezier_vs(self, values): # Arrayf of bool
        self.root_wrapper.top_obj.data.splines.foreach_set('use_bezier_v', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def use_cyclic_us(self): # Array of bool
        array = np.empty(len(self), np.bool)
        self.root_wrapper.top_obj.data.splines.foreach_get('use_cyclic_u', array)
        return array

    @use_cyclic_us.setter
    def use_cyclic_us(self, values): # Arrayf of bool
        self.root_wrapper.top_obj.data.splines.foreach_set('use_cyclic_u', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def use_cyclic_vs(self): # Array of bool
        array = np.empty(len(self), np.bool)
        self.root_wrapper.top_obj.data.splines.foreach_get('use_cyclic_v', array)
        return array

    @use_cyclic_vs.setter
    def use_cyclic_vs(self, values): # Arrayf of bool
        self.root_wrapper.top_obj.data.splines.foreach_set('use_cyclic_v', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def use_endpoint_us(self): # Array of bool
        array = np.empty(len(self), np.bool)
        self.root_wrapper.top_obj.data.splines.foreach_get('use_endpoint_u', array)
        return array

    @use_endpoint_us.setter
    def use_endpoint_us(self, values): # Arrayf of bool
        self.root_wrapper.top_obj.data.splines.foreach_set('use_endpoint_u', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def use_endpoint_vs(self): # Array of bool
        array = np.empty(len(self), np.bool)
        self.root_wrapper.top_obj.data.splines.foreach_get('use_endpoint_v', array)
        return array

    @use_endpoint_vs.setter
    def use_endpoint_vs(self, values): # Arrayf of bool
        self.root_wrapper.top_obj.data.splines.foreach_set('use_endpoint_v', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def use_smooths(self): # Array of bool
        array = np.empty(len(self), np.bool)
        self.root_wrapper.top_obj.data.splines.foreach_get('use_smooth', array)
        return array

    @use_smooths.setter
    def use_smooths(self, values): # Arrayf of bool
        self.root_wrapper.top_obj.data.splines.foreach_set('use_smooth', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

#================================================================================
# Curve class wrapper

class WCurve(Wrapper):

    def __init__(self, root_wrapper):
        super().__init__(root_wrapper)
        self.wsplines = WSplines(self.root_wrapper)

    @property
    def curve(self): # The wrapped Blender curve
        return self.root_wrapper.top_obj.data

    @property
    def bstruct(self): # The wrapped Blender curve
        return self.root_wrapper.top_obj.data

    @property
    def bevel_depth(self): # float
        return self.curve.bevel_depth

    @bevel_depth.setter
    def bevel_depth(self, value): # float
        self.curve.bevel_depth = value

    @property
    def bevel_factor_end(self): # float
        return self.curve.bevel_factor_end

    @bevel_factor_end.setter
    def bevel_factor_end(self, value): # float
        self.curve.bevel_factor_end = value

    @property
    def bevel_factor_start(self): # float
        return self.curve.bevel_factor_start

    @bevel_factor_start.setter
    def bevel_factor_start(self, value): # float
        self.curve.bevel_factor_start = value

    @property
    def bevel_object(self): # object
        return self.curve.bevel_object

    @bevel_object.setter
    def bevel_object(self, value): # object
        self.curve.bevel_object = value

    @property
    def bevel_resolution(self): # int
        return self.curve.bevel_resolution

    @bevel_resolution.setter
    def bevel_resolution(self, value): # int
        self.curve.bevel_resolution = value

    @property
    def dimensions(self): # str
        return self.curve.dimensions

    @dimensions.setter
    def dimensions(self, value): # str
        self.curve.dimensions = value

    @property
    def eval_time(self): # float
        return self.curve.eval_time

    @eval_time.setter
    def eval_time(self, value): # float
        self.curve.eval_time = value

    @property
    def extrude(self): # float
        return self.curve.extrude

    @extrude.setter
    def extrude(self, value): # float
        self.curve.extrude = value

    @property
    def fill_mode(self): # str
        return self.curve.fill_mode

    @fill_mode.setter
    def fill_mode(self, value): # str
        self.curve.fill_mode = value

    @property
    def offset(self): # float
        return self.curve.offset

    @offset.setter
    def offset(self, value): # float
        self.curve.offset = value

    @property
    def path_duration(self): # int
        return self.curve.path_duration

    @path_duration.setter
    def path_duration(self, value): # int
        self.curve.path_duration = value

    @property
    def render_resolution_u(self): # int
        return self.curve.render_resolution_u

    @render_resolution_u.setter
    def render_resolution_u(self, value): # int
        self.curve.render_resolution_u = value

    @property
    def render_resolution_v(self): # int
        return self.curve.render_resolution_v

    @render_resolution_v.setter
    def render_resolution_v(self, value): # int
        self.curve.render_resolution_v = value

    @property
    def resolution_u(self): # int
        return self.curve.resolution_u

    @resolution_u.setter
    def resolution_u(self, value): # int
        self.curve.resolution_u = value

    @property
    def resolution_v(self): # int
        return self.curve.resolution_v

    @resolution_v.setter
    def resolution_v(self, value): # int
        self.curve.resolution_v = value

    @property
    def taper_object(self): # object
        return self.curve.taper_object

    @taper_object.setter
    def taper_object(self, value): # object
        self.curve.taper_object = value

    @property
    def twist_smooth(self): # float
        return self.curve.twist_smooth

    @twist_smooth.setter
    def twist_smooth(self, value): # float
        self.curve.twist_smooth = value

    @property
    def use_auto_texspace(self): # bool
        return self.curve.use_auto_texspace

    @use_auto_texspace.setter
    def use_auto_texspace(self, value): # bool
        self.curve.use_auto_texspace = value

    @property
    def use_deform_bounds(self): # bool
        return self.curve.use_deform_bounds

    @use_deform_bounds.setter
    def use_deform_bounds(self, value): # bool
        self.curve.use_deform_bounds = value

    @property
    def use_fill_caps(self): # bool
        return self.curve.use_fill_caps

    @use_fill_caps.setter
    def use_fill_caps(self, value): # bool
        self.curve.use_fill_caps = value

    @property
    def use_fill_deform(self): # bool
        return self.curve.use_fill_deform

    @use_fill_deform.setter
    def use_fill_deform(self, value): # bool
        self.curve.use_fill_deform = value

    @property
    def use_map_taper(self): # bool
        return self.curve.use_map_taper

    @use_map_taper.setter
    def use_map_taper(self, value): # bool
        self.curve.use_map_taper = value

    @property
    def use_path_follow(self): # bool
        return self.curve.use_path_follow

    @use_path_follow.setter
    def use_path_follow(self, value): # bool
        self.curve.use_path_follow = value

    @property
    def use_path(self): # bool
        return self.curve.use_path

    @use_path.setter
    def use_path(self, value): # bool
        self.curve.use_path = value

    @property
    def use_radius(self): # bool
        return self.curve.use_radius

    @use_radius.setter
    def use_radius(self, value): # bool
        self.curve.use_radius = value

    @property
    def use_stretch(self): # bool
        return self.curve.use_stretch

    @use_stretch.setter
    def use_stretch(self, value): # bool
        self.curve.use_stretch = value

    @property
    def t0(self): # float
        return self.curve.bevel_factor_end

    @t0.setter
    def t0(self, value): # float
        self.curve.bevel_factor_end = value

    @property
    def t1(self): # float
        return self.curve.bevel_factor_start

    @t1.setter
    def t1(self, value): # float
        self.curve.bevel_factor_start = value

#================================================================================
# Object class wrapper

class WObject(WObjectRoot):

    def __init__(self, name):
        super().__init__(self)
        self.name = name

    @property
    def top_obj(self):
        return bpy.data.objects[self.name]

    @property
    def object(self): # The wrapped Blender object
        return bpy.data.objects[self.name]

    @property
    def bstruct(self): # The wrapped Blender object
        return bpy.data.objects[self.name]

    @property
    def active_material_index(self): # int
        return self.object.active_material_index

    @active_material_index.setter
    def active_material_index(self, value): # int
        self.object.active_material_index = value

    @property
    def active_shape_key_index(self): # int
        return self.object.active_shape_key_index

    @active_shape_key_index.setter
    def active_shape_key_index(self, value): # int
        self.object.active_shape_key_index = value

    @property
    def bound_box(self): # bbox
        return self.object.bound_box

    @bound_box.setter
    def bound_box(self, value): # bbox
        self.object.bound_box = value

    @property
    def color(self): # V4
        return self.object.color

    @color.setter
    def color(self, value): # V4
        self.object.color = to_array(value, (4,), "color")

    @property
    def delta_location(self): # V3
        return self.object.delta_location

    @delta_location.setter
    def delta_location(self, value): # V3
        self.object.delta_location = to_array(value, (3,), "delta_location")

    @property
    def delta_rotation_euler(self): # V3
        return self.object.delta_rotation_euler

    @delta_rotation_euler.setter
    def delta_rotation_euler(self, value): # V3
        self.object.delta_rotation_euler = to_array(value, (3,), "delta_rotation_euler")

    @property
    def delta_rotation_quaternion(self): # V4
        return self.object.delta_rotation_quaternion

    @delta_rotation_quaternion.setter
    def delta_rotation_quaternion(self, value): # V4
        self.object.delta_rotation_quaternion = to_array(value, (4,), "delta_rotation_quaternion")

    @property
    def delta_scale(self): # V3
        return self.object.delta_scale

    @delta_scale.setter
    def delta_scale(self, value): # V3
        self.object.delta_scale = to_array(value, (3,), "delta_scale")

    @property
    def empty_display_size(self): # float
        return self.object.empty_display_size

    @empty_display_size.setter
    def empty_display_size(self, value): # float
        self.object.empty_display_size = value

    @property
    def empty_image_offset(self): # V2
        return self.object.empty_image_offset

    @empty_image_offset.setter
    def empty_image_offset(self, value): # V2
        self.object.empty_image_offset = to_array(value, (2,), "empty_image_offset")

    @property
    def dimensions(self): # V3
        return self.object.dimensions

    @dimensions.setter
    def dimensions(self, value): # V3
        self.object.dimensions = to_array(value, (3,), "dimensions")

    @property
    def hide_render(self): # bool
        return self.object.hide_render

    @hide_render.setter
    def hide_render(self, value): # bool
        self.object.hide_render = value

    @property
    def hide_select(self): # bool
        return self.object.hide_select

    @hide_select.setter
    def hide_select(self, value): # bool
        self.object.hide_select = value

    @property
    def hide_viewport(self): # bool
        return self.object.hide_viewport

    @hide_viewport.setter
    def hide_viewport(self, value): # bool
        self.object.hide_viewport = value

    @property
    def instance_faces_scale(self): # float
        return self.object.instance_faces_scale

    @instance_faces_scale.setter
    def instance_faces_scale(self, value): # float
        self.object.instance_faces_scale = value

    @property
    def location(self): # V3
        return self.object.location

    @property
    def x(self):
        return self.object.location[0]

    @property
    def y(self):
        return self.object.location[1]

    @property
    def z(self):
        return self.object.location[2]

    @location.setter
    def location(self, value): # V3
        self.object.location = to_array(value, (3,), "location")

    @x.setter
    def x(self, value):
        self.object.location[0] = value

    @y.setter
    def y(self, value):
        self.object.location[1] = value

    @z.setter
    def z(self, value):
        self.object.location[2] = value

    @property
    def lock_scale(self): # bool
        return self.object.lock_scale

    @lock_scale.setter
    def lock_scale(self, value): # bool
        self.object.lock_scale = value

    @property
    def matrix_basis(self): # M4
        return self.object.matrix_basis

    @property
    def matrix_local(self): # M4
        return self.object.matrix_local

    @property
    def matrix_parent_inverse(self): # M4
        return self.object.matrix_parent_inverse

    @property
    def matrix_world(self): # M4
        return self.object.matrix_world

    @property
    def pass_index(self): # int
        return self.object.pass_index

    @pass_index.setter
    def pass_index(self, value): # int
        self.object.pass_index = value

    @property
    def rotation_euler(self): # V3
        return self.object.rotation_euler

    @property
    def rx(self):
        return self.object.rotation_euler[0]

    @property
    def ry(self):
        return self.object.rotation_euler[1]

    @property
    def rz(self):
        return self.object.rotation_euler[2]

    @rotation_euler.setter
    def rotation_euler(self, value): # V3
        self.object.rotation_euler = to_array(value, (3,), "rotation_euler")

    @rx.setter
    def rx(self, value):
        self.object.rotation_euler[0] = value

    @ry.setter
    def ry(self, value):
        self.object.rotation_euler[1] = value

    @rz.setter
    def rz(self, value):
        self.object.rotation_euler[2] = value

    @property
    def rotation_mode(self): # str
        return self.object.rotation_mode

    @rotation_mode.setter
    def rotation_mode(self, value): # str
        self.object.rotation_mode = value

    @property
    def rotation_quaternion(self): # V4
        return self.object.rotation_quaternion

    @rotation_quaternion.setter
    def rotation_quaternion(self, value): # V4
        self.object.rotation_quaternion = to_array(value, (4,), "rotation_quaternion")

    @property
    def scale(self): # V3
        return self.object.scale

    @property
    def sx(self):
        return self.object.scale[0]

    @property
    def sy(self):
        return self.object.scale[1]

    @property
    def sz(self):
        return self.object.scale[2]

    @scale.setter
    def scale(self, value): # V3
        self.object.scale = to_array(value, (3,), "scale")

    @sx.setter
    def sx(self, value):
        self.object.scale[0] = value

    @sy.setter
    def sy(self, value):
        self.object.scale[1] = value

    @sz.setter
    def sz(self, value):
        self.object.scale[2] = value

    @property
    def show_all_edges(self): # bool
        return self.object.show_all_edges

    @show_all_edges.setter
    def show_all_edges(self, value): # bool
        self.object.show_all_edges = value

    @property
    def show_axis(self): # bool
        return self.object.show_axis

    @show_axis.setter
    def show_axis(self, value): # bool
        self.object.show_axis = value

    @property
    def show_bounds(self): # bool
        return self.object.show_bounds

    @show_bounds.setter
    def show_bounds(self, value): # bool
        self.object.show_bounds = value

    @property
    def show_empty_image_only_axis_aligned(self): # bool
        return self.object.show_empty_image_only_axis_aligned

    @show_empty_image_only_axis_aligned.setter
    def show_empty_image_only_axis_aligned(self, value): # bool
        self.object.show_empty_image_only_axis_aligned = value

    @property
    def show_empty_image_orthographic(self): # bool
        return self.object.show_empty_image_orthographic

    @show_empty_image_orthographic.setter
    def show_empty_image_orthographic(self, value): # bool
        self.object.show_empty_image_orthographic = value

    @property
    def show_empty_image_perspective(self): # bool
        return self.object.show_empty_image_perspective

    @show_empty_image_perspective.setter
    def show_empty_image_perspective(self, value): # bool
        self.object.show_empty_image_perspective = value

    @property
    def show_in_front(self): # bool
        return self.object.show_in_front

    @show_in_front.setter
    def show_in_front(self, value): # bool
        self.object.show_in_front = value

    @property
    def show_instancer_for_render(self): # bool
        return self.object.show_instancer_for_render

    @show_instancer_for_render.setter
    def show_instancer_for_render(self, value): # bool
        self.object.show_instancer_for_render = value

    @property
    def show_instancer_for_viewport(self): # bool
        return self.object.show_instancer_for_viewport

    @show_instancer_for_viewport.setter
    def show_instancer_for_viewport(self, value): # bool
        self.object.show_instancer_for_viewport = value

    @property
    def show_name(self): # bool
        return self.object.show_name

    @show_name.setter
    def show_name(self, value): # bool
        self.object.show_name = value

    @property
    def show_only_shape_key(self): # bool
        return self.object.show_only_shape_key

    @show_only_shape_key.setter
    def show_only_shape_key(self, value): # bool
        self.object.show_only_shape_key = value

    @property
    def show_texture_space(self): # bool
        return self.object.show_texture_space

    @show_texture_space.setter
    def show_texture_space(self, value): # bool
        self.object.show_texture_space = value

    @property
    def show_transparent(self): # bool
        return self.object.show_transparent

    @show_transparent.setter
    def show_transparent(self, value): # bool
        self.object.show_transparent = value

    @property
    def show_wire(self): # bool
        return self.object.show_wire

    @show_wire.setter
    def show_wire(self, value): # bool
        self.object.show_wire = value

    @property
    def track_axis(self): # str
        return self.object.track_axis

    @track_axis.setter
    def track_axis(self, value): # str
        self.object.track_axis = value

    @property
    def type(self): # str
        return self.object.type

    @property
    def up_axis(self): # str
        return self.object.up_axis

    @up_axis.setter
    def up_axis(self, value): # str
        self.object.up_axis = value

    @property
    def use_empty_image_alpha(self): # bool
        return self.object.use_empty_image_alpha

    @use_empty_image_alpha.setter
    def use_empty_image_alpha(self, value): # bool
        self.object.use_empty_image_alpha = value

    @property
    def use_instance_faces_scale(self): # bool
        return self.object.use_instance_faces_scale

    @use_instance_faces_scale.setter
    def use_instance_faces_scale(self, value): # bool
        self.object.use_instance_faces_scale = value

    @property
    def use_instance_vertices_rotation(self): # bool
        return self.object.use_instance_vertices_rotation

    @use_instance_vertices_rotation.setter
    def use_instance_vertices_rotation(self, value): # bool
        self.object.use_instance_vertices_rotation = value

    @property
    def use_shape_key_edit_mode(self): # bool
        return self.object.use_shape_key_edit_mode

    @use_shape_key_edit_mode.setter
    def use_shape_key_edit_mode(self, value): # bool
        self.object.use_shape_key_edit_mode = value

#================================================================================
# Array of WObjects

class WObjects(Wrapper):
    def __init__(self, name):
        super().__init__(self)
        self.name = name

    @property
    def top_obj(self):
        return bpy.data.collections[self.name]

    @property
    def objects(self): # The wrapped Blender objects
        return bpy.data.collections[self.name].objects

    def __len__(self):
        return len(bpy.data.collections[self.name].objects)

    def __getitem__(self, index):
        return WObject(bpy.data.collections[self.name].objects[index].name)

    @property
    def item_class(self):
        return WObject

    @property
    def active_material_indices(self): # Array of int
        array = np.empty(len(self), np.int)
        bpy.data.collections[self.name].objects.foreach_get('active_material_index', array)
        return array

    @active_material_indices.setter
    def active_material_indices(self, values): # Arrayf of int
        bpy.data.collections[self.name].objects.foreach_set('active_material_index', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def active_shape_key_indices(self): # Array of int
        array = np.empty(len(self), np.int)
        bpy.data.collections[self.name].objects.foreach_get('active_shape_key_index', array)
        return array

    @active_shape_key_indices.setter
    def active_shape_key_indices(self, values): # Arrayf of int
        bpy.data.collections[self.name].objects.foreach_set('active_shape_key_index', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def colors(self): # Array of V4
        array = np.empty(len(self)*4, np.float)
        bpy.data.collections[self.name].objects.foreach_get('color', array)
        return array.reshape(len(self), 4)

    @colors.setter
    def colors(self, values): # Arrayf of V4
        bpy.data.collections[self.name].objects.foreach_set('color', to_array(values, (len(self), 4), f'4-vector or array of {len(self)} 4-vectors').reshape(len(self) * 4))

    @property
    def delta_locations(self): # Array of V3
        array = np.empty(len(self)*3, np.float)
        bpy.data.collections[self.name].objects.foreach_get('delta_location', array)
        return array.reshape(len(self), 3)

    @delta_locations.setter
    def delta_locations(self, values): # Arrayf of V3
        bpy.data.collections[self.name].objects.foreach_set('delta_location', to_array(values, (len(self), 3), f'3-vector or array of {len(self)} 3-vectors').reshape(len(self) * 3))

    @property
    def delta_rotation_eulers(self): # Array of V3
        array = np.empty(len(self)*3, np.float)
        bpy.data.collections[self.name].objects.foreach_get('delta_rotation_euler', array)
        return array.reshape(len(self), 3)

    @delta_rotation_eulers.setter
    def delta_rotation_eulers(self, values): # Arrayf of V3
        bpy.data.collections[self.name].objects.foreach_set('delta_rotation_euler', to_array(values, (len(self), 3), f'3-vector or array of {len(self)} 3-vectors').reshape(len(self) * 3))

    @property
    def delta_rotation_quaternions(self): # Array of V4
        array = np.empty(len(self)*4, np.float)
        bpy.data.collections[self.name].objects.foreach_get('delta_rotation_quaternion', array)
        return array.reshape(len(self), 4)

    @delta_rotation_quaternions.setter
    def delta_rotation_quaternions(self, values): # Arrayf of V4
        bpy.data.collections[self.name].objects.foreach_set('delta_rotation_quaternion', to_array(values, (len(self), 4), f'4-vector or array of {len(self)} 4-vectors').reshape(len(self) * 4))

    @property
    def delta_scales(self): # Array of V3
        array = np.empty(len(self)*3, np.float)
        bpy.data.collections[self.name].objects.foreach_get('delta_scale', array)
        return array.reshape(len(self), 3)

    @delta_scales.setter
    def delta_scales(self, values): # Arrayf of V3
        bpy.data.collections[self.name].objects.foreach_set('delta_scale', to_array(values, (len(self), 3), f'3-vector or array of {len(self)} 3-vectors').reshape(len(self) * 3))

    @property
    def empty_display_sizes(self): # Array of float
        array = np.empty(len(self), np.float)
        bpy.data.collections[self.name].objects.foreach_get('empty_display_size', array)
        return array

    @empty_display_sizes.setter
    def empty_display_sizes(self, values): # Arrayf of float
        bpy.data.collections[self.name].objects.foreach_set('empty_display_size', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def empty_image_offsets(self): # Array of V2
        array = np.empty(len(self)*2, np.float)
        bpy.data.collections[self.name].objects.foreach_get('empty_image_offset', array)
        return array.reshape(len(self), 2)

    @empty_image_offsets.setter
    def empty_image_offsets(self, values): # Arrayf of V2
        bpy.data.collections[self.name].objects.foreach_set('empty_image_offset', to_array(values, (len(self), 2), f'2-vector or array of {len(self)} 2-vectors').reshape(len(self) * 2))

    @property
    def dimensions_s(self): # Array of V3
        array = np.empty(len(self)*3, np.float)
        bpy.data.collections[self.name].objects.foreach_get('dimensions', array)
        return array.reshape(len(self), 3)

    @dimensions_s.setter
    def dimensions_s(self, values): # Arrayf of V3
        bpy.data.collections[self.name].objects.foreach_set('dimensions', to_array(values, (len(self), 3), f'3-vector or array of {len(self)} 3-vectors').reshape(len(self) * 3))

    @property
    def hide_renders(self): # Array of bool
        array = np.empty(len(self), np.bool)
        bpy.data.collections[self.name].objects.foreach_get('hide_render', array)
        return array

    @hide_renders.setter
    def hide_renders(self, values): # Arrayf of bool
        bpy.data.collections[self.name].objects.foreach_set('hide_render', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def hide_selects(self): # Array of bool
        array = np.empty(len(self), np.bool)
        bpy.data.collections[self.name].objects.foreach_get('hide_select', array)
        return array

    @hide_selects.setter
    def hide_selects(self, values): # Arrayf of bool
        bpy.data.collections[self.name].objects.foreach_set('hide_select', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def hide_viewports(self): # Array of bool
        array = np.empty(len(self), np.bool)
        bpy.data.collections[self.name].objects.foreach_get('hide_viewport', array)
        return array

    @hide_viewports.setter
    def hide_viewports(self, values): # Arrayf of bool
        bpy.data.collections[self.name].objects.foreach_set('hide_viewport', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def instance_faces_scales(self): # Array of float
        array = np.empty(len(self), np.float)
        bpy.data.collections[self.name].objects.foreach_get('instance_faces_scale', array)
        return array

    @instance_faces_scales.setter
    def instance_faces_scales(self, values): # Arrayf of float
        bpy.data.collections[self.name].objects.foreach_set('instance_faces_scale', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def locations(self): # Array of V3
        array = np.empty(len(self)*3, np.float)
        bpy.data.collections[self.name].objects.foreach_get('location', array)
        return array.reshape(len(self), 3)

    @locations.setter
    def locations(self, values): # Arrayf of V3
        bpy.data.collections[self.name].objects.foreach_set('location', to_array(values, (len(self), 3), f'3-vector or array of {len(self)} 3-vectors').reshape(len(self) * 3))

    # xyzw access to locations

    @property
    def xs(self): 
        return self.locations[:, 0]

    @xs.setter
    def xs(self, values):
        locations = self.locations
        locations[:, 0] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.locations = locations

    @property
    def ys(self): 
        return self.locations[:, 1]

    @ys.setter
    def ys(self, values):
        locations = self.locations
        locations[:, 1] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.locations = locations

    @property
    def zs(self): 
        return self.locations[:, 2]

    @zs.setter
    def zs(self, values):
        locations = self.locations
        locations[:, 2] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.locations = locations

    @property
    def lock_scales(self): # Array of bool
        array = np.empty(len(self), np.bool)
        bpy.data.collections[self.name].objects.foreach_get('lock_scale', array)
        return array

    @lock_scales.setter
    def lock_scales(self, values): # Arrayf of bool
        bpy.data.collections[self.name].objects.foreach_set('lock_scale', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def pass_indices(self): # Array of int
        array = np.empty(len(self), np.int)
        bpy.data.collections[self.name].objects.foreach_get('pass_index', array)
        return array

    @pass_indices.setter
    def pass_indices(self, values): # Arrayf of int
        bpy.data.collections[self.name].objects.foreach_set('pass_index', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def rotation_eulers(self): # Array of V3
        array = np.empty(len(self)*3, np.float)
        bpy.data.collections[self.name].objects.foreach_get('rotation_euler', array)
        return array.reshape(len(self), 3)

    @rotation_eulers.setter
    def rotation_eulers(self, values): # Arrayf of V3
        bpy.data.collections[self.name].objects.foreach_set('rotation_euler', to_array(values, (len(self), 3), f'3-vector or array of {len(self)} 3-vectors').reshape(len(self) * 3))

    # xyzw access to rotation_eulers

    @property
    def rxs(self): 
        return self.rotation_eulers[:, 0]

    @rxs.setter
    def rxs(self, values):
        rotation_eulers = self.rotation_eulers
        rotation_eulers[:, 0] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.rotation_eulers = rotation_eulers

    @property
    def rys(self): 
        return self.rotation_eulers[:, 1]

    @rys.setter
    def rys(self, values):
        rotation_eulers = self.rotation_eulers
        rotation_eulers[:, 1] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.rotation_eulers = rotation_eulers

    @property
    def rzs(self): 
        return self.rotation_eulers[:, 2]

    @rzs.setter
    def rzs(self, values):
        rotation_eulers = self.rotation_eulers
        rotation_eulers[:, 2] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.rotation_eulers = rotation_eulers

    @property
    def rotation_modes(self): # Array of str
        array = np.empty(len(self), np.object)
        coll = bpy.data.collections[self.name].objects
        for i in range(len(self)):
            array[i] = coll[i].rotation_mode
        return array

    @rotation_modes.setter
    def rotation_modes(self, values): # Arrayf of str
        array = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        coll = bpy.data.collections[self.name].objects
        for i in range(len(self)):
            coll[i].rotation_mode = array[i]

    @property
    def rotation_quaternions(self): # Array of V4
        array = np.empty(len(self)*4, np.float)
        bpy.data.collections[self.name].objects.foreach_get('rotation_quaternion', array)
        return array.reshape(len(self), 4)

    @rotation_quaternions.setter
    def rotation_quaternions(self, values): # Arrayf of V4
        bpy.data.collections[self.name].objects.foreach_set('rotation_quaternion', to_array(values, (len(self), 4), f'4-vector or array of {len(self)} 4-vectors').reshape(len(self) * 4))

    @property
    def scales(self): # Array of V3
        array = np.empty(len(self)*3, np.float)
        bpy.data.collections[self.name].objects.foreach_get('scale', array)
        return array.reshape(len(self), 3)

    @scales.setter
    def scales(self, values): # Arrayf of V3
        bpy.data.collections[self.name].objects.foreach_set('scale', to_array(values, (len(self), 3), f'3-vector or array of {len(self)} 3-vectors').reshape(len(self) * 3))

    # xyzw access to scales

    @property
    def sxs(self): 
        return self.scales[:, 0]

    @sxs.setter
    def sxs(self, values):
        scales = self.scales
        scales[:, 0] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.scales = scales

    @property
    def sys(self): 
        return self.scales[:, 1]

    @sys.setter
    def sys(self, values):
        scales = self.scales
        scales[:, 1] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.scales = scales

    @property
    def szs(self): 
        return self.scales[:, 2]

    @szs.setter
    def szs(self, values):
        scales = self.scales
        scales[:, 2] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.scales = scales

    @property
    def show_all_edges_s(self): # Array of bool
        array = np.empty(len(self), np.bool)
        bpy.data.collections[self.name].objects.foreach_get('show_all_edges', array)
        return array

    @show_all_edges_s.setter
    def show_all_edges_s(self, values): # Arrayf of bool
        bpy.data.collections[self.name].objects.foreach_set('show_all_edges', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def show_axis_s(self): # Array of bool
        array = np.empty(len(self), np.bool)
        bpy.data.collections[self.name].objects.foreach_get('show_axis', array)
        return array

    @show_axis_s.setter
    def show_axis_s(self, values): # Arrayf of bool
        bpy.data.collections[self.name].objects.foreach_set('show_axis', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def show_bounds_s(self): # Array of bool
        array = np.empty(len(self), np.bool)
        bpy.data.collections[self.name].objects.foreach_get('show_bounds', array)
        return array

    @show_bounds_s.setter
    def show_bounds_s(self, values): # Arrayf of bool
        bpy.data.collections[self.name].objects.foreach_set('show_bounds', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def show_empty_image_only_axis_aligneds(self): # Array of bool
        array = np.empty(len(self), np.bool)
        bpy.data.collections[self.name].objects.foreach_get('show_empty_image_only_axis_aligned', array)
        return array

    @show_empty_image_only_axis_aligneds.setter
    def show_empty_image_only_axis_aligneds(self, values): # Arrayf of bool
        bpy.data.collections[self.name].objects.foreach_set('show_empty_image_only_axis_aligned', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def show_empty_image_orthographics(self): # Array of bool
        array = np.empty(len(self), np.bool)
        bpy.data.collections[self.name].objects.foreach_get('show_empty_image_orthographic', array)
        return array

    @show_empty_image_orthographics.setter
    def show_empty_image_orthographics(self, values): # Arrayf of bool
        bpy.data.collections[self.name].objects.foreach_set('show_empty_image_orthographic', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def show_empty_image_perspectives(self): # Array of bool
        array = np.empty(len(self), np.bool)
        bpy.data.collections[self.name].objects.foreach_get('show_empty_image_perspective', array)
        return array

    @show_empty_image_perspectives.setter
    def show_empty_image_perspectives(self, values): # Arrayf of bool
        bpy.data.collections[self.name].objects.foreach_set('show_empty_image_perspective', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def show_in_fronts(self): # Array of bool
        array = np.empty(len(self), np.bool)
        bpy.data.collections[self.name].objects.foreach_get('show_in_front', array)
        return array

    @show_in_fronts.setter
    def show_in_fronts(self, values): # Arrayf of bool
        bpy.data.collections[self.name].objects.foreach_set('show_in_front', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def show_instancer_for_renders(self): # Array of bool
        array = np.empty(len(self), np.bool)
        bpy.data.collections[self.name].objects.foreach_get('show_instancer_for_render', array)
        return array

    @show_instancer_for_renders.setter
    def show_instancer_for_renders(self, values): # Arrayf of bool
        bpy.data.collections[self.name].objects.foreach_set('show_instancer_for_render', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def show_instancer_for_viewports(self): # Array of bool
        array = np.empty(len(self), np.bool)
        bpy.data.collections[self.name].objects.foreach_get('show_instancer_for_viewport', array)
        return array

    @show_instancer_for_viewports.setter
    def show_instancer_for_viewports(self, values): # Arrayf of bool
        bpy.data.collections[self.name].objects.foreach_set('show_instancer_for_viewport', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def show_names(self): # Array of bool
        array = np.empty(len(self), np.bool)
        bpy.data.collections[self.name].objects.foreach_get('show_name', array)
        return array

    @show_names.setter
    def show_names(self, values): # Arrayf of bool
        bpy.data.collections[self.name].objects.foreach_set('show_name', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def show_only_shape_keys(self): # Array of bool
        array = np.empty(len(self), np.bool)
        bpy.data.collections[self.name].objects.foreach_get('show_only_shape_key', array)
        return array

    @show_only_shape_keys.setter
    def show_only_shape_keys(self, values): # Arrayf of bool
        bpy.data.collections[self.name].objects.foreach_set('show_only_shape_key', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def show_texture_spaces(self): # Array of bool
        array = np.empty(len(self), np.bool)
        bpy.data.collections[self.name].objects.foreach_get('show_texture_space', array)
        return array

    @show_texture_spaces.setter
    def show_texture_spaces(self, values): # Arrayf of bool
        bpy.data.collections[self.name].objects.foreach_set('show_texture_space', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def show_transparents(self): # Array of bool
        array = np.empty(len(self), np.bool)
        bpy.data.collections[self.name].objects.foreach_get('show_transparent', array)
        return array

    @show_transparents.setter
    def show_transparents(self, values): # Arrayf of bool
        bpy.data.collections[self.name].objects.foreach_set('show_transparent', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def show_wires(self): # Array of bool
        array = np.empty(len(self), np.bool)
        bpy.data.collections[self.name].objects.foreach_get('show_wire', array)
        return array

    @show_wires.setter
    def show_wires(self, values): # Arrayf of bool
        bpy.data.collections[self.name].objects.foreach_set('show_wire', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def track_axis_s(self): # Array of str
        array = np.empty(len(self), np.object)
        coll = bpy.data.collections[self.name].objects
        for i in range(len(self)):
            array[i] = coll[i].track_axis
        return array

    @track_axis_s.setter
    def track_axis_s(self, values): # Arrayf of str
        array = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        coll = bpy.data.collections[self.name].objects
        for i in range(len(self)):
            coll[i].track_axis = array[i]

    @property
    def types(self): # Array of str
        array = np.empty(len(self), np.object)
        coll = bpy.data.collections[self.name].objects
        for i in range(len(self)):
            array[i] = coll[i].type
        return array

    @property
    def up_axis_s(self): # Array of str
        array = np.empty(len(self), np.object)
        coll = bpy.data.collections[self.name].objects
        for i in range(len(self)):
            array[i] = coll[i].up_axis
        return array

    @up_axis_s.setter
    def up_axis_s(self, values): # Arrayf of str
        array = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        coll = bpy.data.collections[self.name].objects
        for i in range(len(self)):
            coll[i].up_axis = array[i]

    @property
    def use_empty_image_alphas(self): # Array of bool
        array = np.empty(len(self), np.bool)
        bpy.data.collections[self.name].objects.foreach_get('use_empty_image_alpha', array)
        return array

    @use_empty_image_alphas.setter
    def use_empty_image_alphas(self, values): # Arrayf of bool
        bpy.data.collections[self.name].objects.foreach_set('use_empty_image_alpha', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def use_instance_faces_scales(self): # Array of bool
        array = np.empty(len(self), np.bool)
        bpy.data.collections[self.name].objects.foreach_get('use_instance_faces_scale', array)
        return array

    @use_instance_faces_scales.setter
    def use_instance_faces_scales(self, values): # Arrayf of bool
        bpy.data.collections[self.name].objects.foreach_set('use_instance_faces_scale', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def use_instance_vertices_rotations(self): # Array of bool
        array = np.empty(len(self), np.bool)
        bpy.data.collections[self.name].objects.foreach_get('use_instance_vertices_rotation', array)
        return array

    @use_instance_vertices_rotations.setter
    def use_instance_vertices_rotations(self, values): # Arrayf of bool
        bpy.data.collections[self.name].objects.foreach_set('use_instance_vertices_rotation', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def use_shape_key_edit_modes(self): # Array of bool
        array = np.empty(len(self), np.bool)
        bpy.data.collections[self.name].objects.foreach_get('use_shape_key_edit_mode', array)
        return array

    @use_shape_key_edit_modes.setter
    def use_shape_key_edit_modes(self, values): # Arrayf of bool
        bpy.data.collections[self.name].objects.foreach_set('use_shape_key_edit_mode', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    def orient(self, axis):
        _axis                 = to_array(axis, (len(self), 3), f'np.float or array of {len(self)} np.float')
        for (_i_obj, _i_axis, ) in zip(self, _axis, ):
            _i_obj.orient(_i_axis)

    def track_to(self, location):
        _location             = to_array(location, (len(self), 3), f'np.float or array of {len(self)} np.float')
        for (_i_obj, _i_location, ) in zip(self, _location, ):
            _i_obj.track_to(_i_location)

    def distances(self, location):
        _location             = to_array(location, (len(self), 3), f'np.float or array of {len(self)} np.float')
        _res                  = np.empty(len(self), np.float)
        for (i, _i_obj, _i_location, ) in zip(range(len(self)), self, _location, ):
            _res[i] = _i_obj.distance(_i_location)
        return _res

#================================================================================
# Texture class wrapper

class WTexture(Wrapper):

    def __init__(self, name):
        super().__init__(self)
        self.name = name

    @property
    def top_obj(self):
        return bpy.data.textures[self.name]

    @property
    def texture(self): # The wrapped Blender texture
        return bpy.data.textures[self.name]

    @property
    def bstruct(self): # The wrapped Blender texture
        return bpy.data.textures[self.name]

    @property
    def cloud_type(self): # str
        return self.texture.cloud_type

    @cloud_type.setter
    def cloud_type(self, value): # str
        self.texture.cloud_type = value

    @property
    def contrast(self): # float
        return self.texture.contrast

    @contrast.setter
    def contrast(self, value): # float
        self.texture.contrast = value

    @property
    def factor_blue(self): # float
        return self.texture.factor_blue

    @factor_blue.setter
    def factor_blue(self, value): # float
        self.texture.factor_blue = value

    @property
    def factor_green(self): # float
        return self.texture.factor_green

    @factor_green.setter
    def factor_green(self, value): # float
        self.texture.factor_green = value

    @property
    def factor_red(self): # float
        return self.texture.factor_red

    @factor_red.setter
    def factor_red(self, value): # float
        self.texture.factor_red = value

    @property
    def intensity(self): # float
        return self.texture.intensity

    @intensity.setter
    def intensity(self, value): # float
        self.texture.intensity = value

    @property
    def is_embedded_data(self): # bool
        return self.texture.is_embedded_data

    @property
    def is_evaluated(self): # bool
        return self.texture.is_evaluated

    @property
    def is_library_indirect(self): # bool
        return self.texture.is_library_indirect

    @property
    def nabla(self): # float
        return self.texture.nabla

    @nabla.setter
    def nabla(self, value): # float
        self.texture.nabla = value

    @property
    def name_full(self): # str
        return self.texture.name_full

    @property
    def noise_basis(self): # str
        return self.texture.noise_basis

    @noise_basis.setter
    def noise_basis(self, value): # str
        self.texture.noise_basis = value

    @property
    def noise_depth(self): # int
        return self.texture.noise_depth

    @noise_depth.setter
    def noise_depth(self, value): # int
        self.texture.noise_depth = value

    @property
    def noise_scale(self): # float
        return self.texture.noise_scale

    @noise_scale.setter
    def noise_scale(self, value): # float
        self.texture.noise_scale = value

    @property
    def noise_type(self): # str
        return self.texture.noise_type

    @noise_type.setter
    def noise_type(self, value): # str
        self.texture.noise_type = value

    @property
    def saturation(self): # float
        return self.texture.saturation

    @saturation.setter
    def saturation(self, value): # float
        self.texture.saturation = value

    @property
    def tag(self): # bool
        return self.texture.tag

    @tag.setter
    def tag(self, value): # bool
        self.texture.tag = value

    @property
    def type(self): # str
        return self.texture.type

    @type.setter
    def type(self, value): # str
        self.texture.type = value

    @property
    def use_clamp(self): # bool
        return self.texture.use_clamp

    @use_clamp.setter
    def use_clamp(self, value): # bool
        self.texture.use_clamp = value

    @property
    def use_color_ramp(self): # bool
        return self.texture.use_color_ramp

    @use_color_ramp.setter
    def use_color_ramp(self, value): # bool
        self.texture.use_color_ramp = value

    @property
    def use_fake_user(self): # bool
        return self.texture.use_fake_user

    @use_fake_user.setter
    def use_fake_user(self, value): # bool
        self.texture.use_fake_user = value

    @property
    def use_nodes(self): # bool
        return self.texture.use_nodes

    @use_nodes.setter
    def use_nodes(self, value): # bool
        self.texture.use_nodes = value

    @property
    def use_preview_alpha(self): # bool
        return self.texture.use_preview_alpha

    @use_preview_alpha.setter
    def use_preview_alpha(self, value): # bool
        self.texture.use_preview_alpha = value

#================================================================================
# KeyFrame class wrapper

class WKeyFrame(Wrapper):

    @property
    def key_frame(self): # The wrapped Blender key_frame
        return self.root_wrapper.top_obj

    @property
    def bstruct(self): # The wrapped Blender key_frame
        return self.root_wrapper.top_obj

    @property
    def amplitude(self): # float
        return self.key_frame.amplitude

    @amplitude.setter
    def amplitude(self, value): # float
        self.key_frame.amplitude = value

    @property
    def back(self): # float
        return self.key_frame.back

    @back.setter
    def back(self, value): # float
        self.key_frame.back = value

    @property
    def co(self): # V2
        return self.key_frame.co

    @property
    def x(self):
        return self.key_frame.co[0]

    @property
    def y(self):
        return self.key_frame.co[1]

    @co.setter
    def co(self, value): # V2
        self.key_frame.co = to_array(value, (2,), "co")

    @x.setter
    def x(self, value):
        self.key_frame.co[0] = value

    @y.setter
    def y(self, value):
        self.key_frame.co[1] = value

    @property
    def easing(self): # str
        return self.key_frame.easing

    @easing.setter
    def easing(self, value): # str
        self.key_frame.easing = value

    @property
    def handle_left(self): # V2
        return self.key_frame.handle_left

    @property
    def lx(self):
        return self.key_frame.handle_left[0]

    @property
    def ly(self):
        return self.key_frame.handle_left[1]

    @handle_left.setter
    def handle_left(self, value): # V2
        self.key_frame.handle_left = to_array(value, (2,), "handle_left")

    @lx.setter
    def lx(self, value):
        self.key_frame.handle_left[0] = value

    @ly.setter
    def ly(self, value):
        self.key_frame.handle_left[1] = value

    @property
    def handle_left_type(self): # str
        return self.key_frame.handle_left_type

    @handle_left_type.setter
    def handle_left_type(self, value): # str
        self.key_frame.handle_left_type = value

    @property
    def handle_right(self): # V2
        return self.key_frame.handle_right

    @property
    def rx(self):
        return self.key_frame.handle_right[0]

    @property
    def ry(self):
        return self.key_frame.handle_right[1]

    @handle_right.setter
    def handle_right(self, value): # V2
        self.key_frame.handle_right = to_array(value, (2,), "handle_right")

    @rx.setter
    def rx(self, value):
        self.key_frame.handle_right[0] = value

    @ry.setter
    def ry(self, value):
        self.key_frame.handle_right[1] = value

    @property
    def handle_right_type(self): # str
        return self.key_frame.handle_right_type

    @handle_right_type.setter
    def handle_right_type(self, value): # str
        self.key_frame.handle_right_type = value

    @property
    def interpolation(self): # str
        return self.key_frame.interpolation

    @interpolation.setter
    def interpolation(self, value): # str
        self.key_frame.interpolation = value

    @property
    def period(self): # float
        return self.key_frame.period

    @period.setter
    def period(self, value): # float
        self.key_frame.period = value

    @property
    def select_control_point(self): # bool
        return self.key_frame.select_control_point

    @select_control_point.setter
    def select_control_point(self, value): # bool
        self.key_frame.select_control_point = value

    @property
    def select_left_handle(self): # bool
        return self.key_frame.select_left_handle

    @select_left_handle.setter
    def select_left_handle(self, value): # bool
        self.key_frame.select_left_handle = value

    @property
    def select_right_handle(self): # bool
        return self.key_frame.select_right_handle

    @select_right_handle.setter
    def select_right_handle(self, value): # bool
        self.key_frame.select_right_handle = value

#================================================================================
# Array of WKeyFrames

class WKeyFrames(Wrapper):

    @property
    def key_frames(self): # The wrapped Blender key_frames
        return self.root_wrapper.top_objbpy.data.textures[self.name]

    def __len__(self):
        return len(self.root_wrapper.top_objbpy.data.textures[self.name])

    def __getitem__(self, index):
        return WKeyFrame(self.root_wrapper, index=index)

    @property
    def item_class(self):
        return WKeyFrame

    @property
    def amplitudes(self): # Array of float
        array = np.empty(len(self), np.float)
        self.root_wrapper.top_objbpy.data.textures[self.name].foreach_get('amplitude', array)
        return array

    @amplitudes.setter
    def amplitudes(self, values): # Arrayf of float
        self.root_wrapper.top_objbpy.data.textures[self.name].foreach_set('amplitude', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def backs(self): # Array of float
        array = np.empty(len(self), np.float)
        self.root_wrapper.top_objbpy.data.textures[self.name].foreach_get('back', array)
        return array

    @backs.setter
    def backs(self, values): # Arrayf of float
        self.root_wrapper.top_objbpy.data.textures[self.name].foreach_set('back', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def cos(self): # Array of V2
        array = np.empty(len(self)*2, np.float)
        self.root_wrapper.top_objbpy.data.textures[self.name].foreach_get('co', array)
        return array.reshape(len(self), 2)

    @cos.setter
    def cos(self, values): # Arrayf of V2
        self.root_wrapper.top_objbpy.data.textures[self.name].foreach_set('co', to_array(values, (len(self), 2), f'2-vector or array of {len(self)} 2-vectors').reshape(len(self) * 2))

    # xyzw access to cos

    @property
    def xs(self): 
        return self.cos[:, 0]

    @xs.setter
    def xs(self, values):
        cos = self.cos
        cos[:, 0] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.cos = cos

    @property
    def ys(self): 
        return self.cos[:, 1]

    @ys.setter
    def ys(self, values):
        cos = self.cos
        cos[:, 1] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.cos = cos

    @property
    def easings(self): # Array of str
        array = np.empty(len(self), np.object)
        coll = self.root_wrapper.top_objbpy.data.textures[self.name]
        for i in range(len(self)):
            array[i] = coll[i].easing
        return array

    @easings.setter
    def easings(self, values): # Arrayf of str
        array = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        coll = self.root_wrapper.top_objbpy.data.textures[self.name]
        for i in range(len(self)):
            coll[i].easing = array[i]

    @property
    def handle_lefts(self): # Array of V2
        array = np.empty(len(self)*2, np.float)
        self.root_wrapper.top_objbpy.data.textures[self.name].foreach_get('handle_left', array)
        return array.reshape(len(self), 2)

    @handle_lefts.setter
    def handle_lefts(self, values): # Arrayf of V2
        self.root_wrapper.top_objbpy.data.textures[self.name].foreach_set('handle_left', to_array(values, (len(self), 2), f'2-vector or array of {len(self)} 2-vectors').reshape(len(self) * 2))

    # xyzw access to handle_lefts

    @property
    def lxs(self): 
        return self.handle_lefts[:, 0]

    @lxs.setter
    def lxs(self, values):
        handle_lefts = self.handle_lefts
        handle_lefts[:, 0] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.handle_lefts = handle_lefts

    @property
    def lys(self): 
        return self.handle_lefts[:, 1]

    @lys.setter
    def lys(self, values):
        handle_lefts = self.handle_lefts
        handle_lefts[:, 1] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.handle_lefts = handle_lefts

    @property
    def handle_left_types(self): # Array of str
        array = np.empty(len(self), np.object)
        coll = self.root_wrapper.top_objbpy.data.textures[self.name]
        for i in range(len(self)):
            array[i] = coll[i].handle_left_type
        return array

    @handle_left_types.setter
    def handle_left_types(self, values): # Arrayf of str
        array = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        coll = self.root_wrapper.top_objbpy.data.textures[self.name]
        for i in range(len(self)):
            coll[i].handle_left_type = array[i]

    @property
    def handle_rights(self): # Array of V2
        array = np.empty(len(self)*2, np.float)
        self.root_wrapper.top_objbpy.data.textures[self.name].foreach_get('handle_right', array)
        return array.reshape(len(self), 2)

    @handle_rights.setter
    def handle_rights(self, values): # Arrayf of V2
        self.root_wrapper.top_objbpy.data.textures[self.name].foreach_set('handle_right', to_array(values, (len(self), 2), f'2-vector or array of {len(self)} 2-vectors').reshape(len(self) * 2))

    # xyzw access to handle_rights

    @property
    def rxs(self): 
        return self.handle_rights[:, 0]

    @rxs.setter
    def rxs(self, values):
        handle_rights = self.handle_rights
        handle_rights[:, 0] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.handle_rights = handle_rights

    @property
    def rys(self): 
        return self.handle_rights[:, 1]

    @rys.setter
    def rys(self, values):
        handle_rights = self.handle_rights
        handle_rights[:, 1] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.handle_rights = handle_rights

    @property
    def handle_right_types(self): # Array of str
        array = np.empty(len(self), np.object)
        coll = self.root_wrapper.top_objbpy.data.textures[self.name]
        for i in range(len(self)):
            array[i] = coll[i].handle_right_type
        return array

    @handle_right_types.setter
    def handle_right_types(self, values): # Arrayf of str
        array = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        coll = self.root_wrapper.top_objbpy.data.textures[self.name]
        for i in range(len(self)):
            coll[i].handle_right_type = array[i]

    @property
    def interpolations(self): # Array of str
        array = np.empty(len(self), np.object)
        coll = self.root_wrapper.top_objbpy.data.textures[self.name]
        for i in range(len(self)):
            array[i] = coll[i].interpolation
        return array

    @interpolations.setter
    def interpolations(self, values): # Arrayf of str
        array = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        coll = self.root_wrapper.top_objbpy.data.textures[self.name]
        for i in range(len(self)):
            coll[i].interpolation = array[i]

    @property
    def periods(self): # Array of float
        array = np.empty(len(self), np.float)
        self.root_wrapper.top_objbpy.data.textures[self.name].foreach_get('period', array)
        return array

    @periods.setter
    def periods(self, values): # Arrayf of float
        self.root_wrapper.top_objbpy.data.textures[self.name].foreach_set('period', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def select_control_points(self): # Array of bool
        array = np.empty(len(self), np.bool)
        self.root_wrapper.top_objbpy.data.textures[self.name].foreach_get('select_control_point', array)
        return array

    @select_control_points.setter
    def select_control_points(self, values): # Arrayf of bool
        self.root_wrapper.top_objbpy.data.textures[self.name].foreach_set('select_control_point', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def select_left_handles(self): # Array of bool
        array = np.empty(len(self), np.bool)
        self.root_wrapper.top_objbpy.data.textures[self.name].foreach_get('select_left_handle', array)
        return array

    @select_left_handles.setter
    def select_left_handles(self, values): # Arrayf of bool
        self.root_wrapper.top_objbpy.data.textures[self.name].foreach_set('select_left_handle', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def select_right_handles(self): # Array of bool
        array = np.empty(len(self), np.bool)
        self.root_wrapper.top_objbpy.data.textures[self.name].foreach_get('select_right_handle', array)
        return array

    @select_right_handles.setter
    def select_right_handles(self, values): # Arrayf of bool
        self.root_wrapper.top_objbpy.data.textures[self.name].foreach_set('select_right_handle', to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors'))

    @property
    def bpoints(self):
        points = np.empty((len(self), 2*3), np.float).reshape(len(self), 3, 2)
        points[:, 0] = self.handle_lefts
        points[:, 1] = self.cos
        points[:, 2] = self.handle_rights
        return points.reshape(len(self), 2*3)
    
    @bpoints.setter
    def bpoints(self, bpoints):
        a = np.array(bpoints)
        if a.size != len(self)*2*3:
            raise WrapException(
                    "Set Bezier points error: the length of the points array is incorrect",
                    f"Need: {len(self)} triplets of 2-vectors: {len(self)*2*3}",
                    f"Received: array {a.shape} of size {a.size}"
                    )
            
        np.reshape(a, (len(self), 3, 2))
        self.handle_lefts  = a[:, 0]
        self.cos           = a[:, 1]
        self.handle_rights = a[:, 2]
    
