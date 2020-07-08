# ****************************************************************************************************
# Generated 2020-07-07

import numpy as np
from wrapanime.root.root import ArrayOf, Wrapper, CollWrapper, to_array, WObjectRoot

#================================================================================
# MeshVertex class wrapper

class WMeshVertex(Wrapper):

    @property
    def bevel_weight(self): # float
        return self.obj.bevel_weight

    @bevel_weight.setter
    def bevel_weight(self, value): # float
        self.obj.bevel_weight = value

    @property
    def co(self): # V3
        return self.obj.co

    @property
    def x(self):
        return self.obj.co[0]

    @property
    def y(self):
        return self.obj.co[1]

    @property
    def z(self):
        return self.obj.co[2]

    @co.setter
    def co(self, value): # V3
        self.obj.co = value

    @x.setter
    def x(self, value):
        self.obj.co[0] = value

    @y.setter
    def y(self, value):
        self.obj.co[1] = value

    @z.setter
    def z(self, value):
        self.obj.co[2] = value

    @property
    def hide(self): # bool
        return self.obj.hide

    @hide.setter
    def hide(self, value): # bool
        self.obj.hide = value

    @property
    def index(self): # int
        return self.obj.index

    @property
    def normal(self): # V3
        return self.obj.normal

    @property
    def x(self):
        return self.obj.normal[0]

    @property
    def y(self):
        return self.obj.normal[1]

    @property
    def z(self):
        return self.obj.normal[2]

    @property
    def undeformed_co(self): # V3
        return self.obj.undeformed_co

#================================================================================
# Array of WMeshVertices

class WMeshVertices(CollWrapper):
    def __init__(self, coll, wowner):
        super().__init__(coll, wowner, WMeshVertex)
        self._cache_bevel_weights           = None
        self._cache_cos                     = None
        self._cache_hides                   = None
        self._cache_indices                 = None
        self._cache_normals                 = None
        self._cache_undeformed_cos          = None

    def erase_cache(self):
        super().erase_cache()
        self._cache_bevel_weights           = None
        self._cache_cos                     = None
        self._cache_hides                   = None
        self._cache_indices                 = None
        self._cache_normals                 = None
        self._cache_undeformed_cos          = None

    @property
    def bevel_weights(self): # Array of float
        if self._cache_bevel_weights is None:
            self._cache_bevel_weights = np.empty(len(self), np.float)
            self.coll.foreach_get('bevel_weight', self._cache_bevel_weights)
        return self._cache_bevel_weights

    @bevel_weights.setter
    def bevel_weights(self, values): # Arrayf of float
        self._cache_bevel_weights = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('bevel_weight', self._cache_bevel_weights)

    @property
    def cos(self): # Array of V3
        if self._cache_cos is None:
            self._cache_cos = np.empty(len(self)*3, np.float)
            self.coll.foreach_get('co', self._cache_cos)
            self._cache_cos = self._cache_cos.reshape(len(self), 3)
        return self._cache_cos

    @cos.setter
    def cos(self, values): # Arrayf of V3
        self._cache_cos = to_array(values, (len(self), 3), f'3-vector or array of {len(self)} 3-vectors')
        self.coll.foreach_set('co', self._cache_cos.reshape(len(self) * 3))

    # xyzw access to cos

    @property
    def xs(self): 
        return self.cos[:, 0]

    @xs.setter
    def xs(self, values):
        self.cos[:, 0] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.cos = self._cache_cos

    @property
    def ys(self): 
        return self.cos[:, 1]

    @ys.setter
    def ys(self, values):
        self.cos[:, 1] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.cos = self._cache_cos

    @property
    def zs(self): 
        return self.cos[:, 2]

    @zs.setter
    def zs(self, values):
        self.cos[:, 2] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.cos = self._cache_cos

    @property
    def hides(self): # Array of bool
        if self._cache_hides is None:
            self._cache_hides = np.empty(len(self), np.bool)
            self.coll.foreach_get('hide', self._cache_hides)
        return self._cache_hides

    @hides.setter
    def hides(self, values): # Arrayf of bool
        self._cache_hides = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('hide', self._cache_hides)

    @property
    def indices(self): # Array of int
        if self._cache_indices is None:
            self._cache_indices = np.empty(len(self), np.int)
            self.coll.foreach_get('index', self._cache_indices)
        return self._cache_indices

    @property
    def normals(self): # Array of V3
        if self._cache_normals is None:
            self._cache_normals = np.empty(len(self)*3, np.float)
            self.coll.foreach_get('normal', self._cache_normals)
            self._cache_normals = self._cache_normals.reshape(len(self), 3)
        return self._cache_normals

    @property
    def undeformed_cos(self): # Array of V3
        if self._cache_undeformed_cos is None:
            self._cache_undeformed_cos = np.empty(len(self)*3, np.float)
            self.coll.foreach_get('undeformed_co', self._cache_undeformed_cos)
            self._cache_undeformed_cos = self._cache_undeformed_cos.reshape(len(self), 3)
        return self._cache_undeformed_cos

#================================================================================
# Edge class wrapper

class WEdge(Wrapper):

    @property
    def bevel_weight(self): # float
        return self.obj.bevel_weight

    @bevel_weight.setter
    def bevel_weight(self, value): # float
        self.obj.bevel_weight = value

    @property
    def crease(self): # float
        return self.obj.crease

    @crease.setter
    def crease(self, value): # float
        self.obj.crease = value

    @property
    def hide(self): # bool
        return self.obj.hide

    @hide.setter
    def hide(self, value): # bool
        self.obj.hide = value

    @property
    def index(self): # int
        return self.obj.index

    @property
    def is_loose(self): # bool
        return self.obj.is_loose

    @property
    def select(self): # bool
        return self.obj.select

    @select.setter
    def select(self, value): # bool
        self.obj.select = value

    @property
    def use_edge_sharp(self): # bool
        return self.obj.use_edge_sharp

    @use_edge_sharp.setter
    def use_edge_sharp(self, value): # bool
        self.obj.use_edge_sharp = value

    @property
    def use_seam(self): # bool
        return self.obj.use_seam

    @use_seam.setter
    def use_seam(self, value): # bool
        self.obj.use_seam = value

    @property
    def vertices(self): # array
        return self.obj.vertices

#================================================================================
# Array of WEdges

class WEdges(CollWrapper):
    def __init__(self, coll, wowner):
        super().__init__(coll, wowner, WEdge)
        self._cache_bevel_weights           = None
        self._cache_creases                 = None
        self._cache_hides                   = None
        self._cache_indices                 = None
        self._cache_is_looses               = None
        self._cache_selects                 = None
        self._cache_use_edge_sharps         = None
        self._cache_use_seams               = None

    def erase_cache(self):
        super().erase_cache()
        self._cache_bevel_weights           = None
        self._cache_creases                 = None
        self._cache_hides                   = None
        self._cache_indices                 = None
        self._cache_is_looses               = None
        self._cache_selects                 = None
        self._cache_use_edge_sharps         = None
        self._cache_use_seams               = None

    @property
    def bevel_weights(self): # Array of float
        if self._cache_bevel_weights is None:
            self._cache_bevel_weights = np.empty(len(self), np.float)
            self.coll.foreach_get('bevel_weight', self._cache_bevel_weights)
        return self._cache_bevel_weights

    @bevel_weights.setter
    def bevel_weights(self, values): # Arrayf of float
        self._cache_bevel_weights = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('bevel_weight', self._cache_bevel_weights)

    @property
    def creases(self): # Array of float
        if self._cache_creases is None:
            self._cache_creases = np.empty(len(self), np.float)
            self.coll.foreach_get('crease', self._cache_creases)
        return self._cache_creases

    @creases.setter
    def creases(self, values): # Arrayf of float
        self._cache_creases = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('crease', self._cache_creases)

    @property
    def hides(self): # Array of bool
        if self._cache_hides is None:
            self._cache_hides = np.empty(len(self), np.bool)
            self.coll.foreach_get('hide', self._cache_hides)
        return self._cache_hides

    @hides.setter
    def hides(self, values): # Arrayf of bool
        self._cache_hides = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('hide', self._cache_hides)

    @property
    def indices(self): # Array of int
        if self._cache_indices is None:
            self._cache_indices = np.empty(len(self), np.int)
            self.coll.foreach_get('index', self._cache_indices)
        return self._cache_indices

    @property
    def is_looses(self): # Array of bool
        if self._cache_is_looses is None:
            self._cache_is_looses = np.empty(len(self), np.bool)
            self.coll.foreach_get('is_loose', self._cache_is_looses)
        return self._cache_is_looses

    @property
    def selects(self): # Array of bool
        if self._cache_selects is None:
            self._cache_selects = np.empty(len(self), np.bool)
            self.coll.foreach_get('select', self._cache_selects)
        return self._cache_selects

    @selects.setter
    def selects(self, values): # Arrayf of bool
        self._cache_selects = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('select', self._cache_selects)

    @property
    def use_edge_sharps(self): # Array of bool
        if self._cache_use_edge_sharps is None:
            self._cache_use_edge_sharps = np.empty(len(self), np.bool)
            self.coll.foreach_get('use_edge_sharp', self._cache_use_edge_sharps)
        return self._cache_use_edge_sharps

    @use_edge_sharps.setter
    def use_edge_sharps(self, values): # Arrayf of bool
        self._cache_use_edge_sharps = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('use_edge_sharp', self._cache_use_edge_sharps)

    @property
    def use_seams(self): # Array of bool
        if self._cache_use_seams is None:
            self._cache_use_seams = np.empty(len(self), np.bool)
            self.coll.foreach_get('use_seam', self._cache_use_seams)
        return self._cache_use_seams

    @use_seams.setter
    def use_seams(self, values): # Arrayf of bool
        self._cache_use_seams = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('use_seam', self._cache_use_seams)

#================================================================================
# Loop class wrapper

class WLoop(Wrapper):

    @property
    def bitangent_sign(self): # float
        return self.obj.bitangent_sign

    @property
    def bitangent(self): # V3
        return self.obj.bitangent

    @property
    def edge_index(self): # int
        return self.obj.edge_index

    @property
    def index(self): # int
        return self.obj.index

    @property
    def normal(self): # V3
        return self.obj.normal

    @property
    def tangent(self): # V3
        return self.obj.tangent

    @property
    def vertex_index(self): # int
        return self.obj.vertex_index

#================================================================================
# Array of WLoops

class WLoops(CollWrapper):
    def __init__(self, coll, wowner):
        super().__init__(coll, wowner, WLoop)
        self._cache_bitangent_signs         = None
        self._cache_bitangents              = None
        self._cache_edge_indices            = None
        self._cache_indices                 = None
        self._cache_normals                 = None
        self._cache_tangents                = None
        self._cache_vertex_indices          = None

    def erase_cache(self):
        super().erase_cache()
        self._cache_bitangent_signs         = None
        self._cache_bitangents              = None
        self._cache_edge_indices            = None
        self._cache_indices                 = None
        self._cache_normals                 = None
        self._cache_tangents                = None
        self._cache_vertex_indices          = None

    @property
    def bitangent_signs(self): # Array of float
        if self._cache_bitangent_signs is None:
            self._cache_bitangent_signs = np.empty(len(self), np.float)
            self.coll.foreach_get('bitangent_sign', self._cache_bitangent_signs)
        return self._cache_bitangent_signs

    @property
    def bitangents(self): # Array of V3
        if self._cache_bitangents is None:
            self._cache_bitangents = np.empty(len(self)*3, np.float)
            self.coll.foreach_get('bitangent', self._cache_bitangents)
            self._cache_bitangents = self._cache_bitangents.reshape(len(self), 3)
        return self._cache_bitangents

    @property
    def edge_indices(self): # Array of int
        if self._cache_edge_indices is None:
            self._cache_edge_indices = np.empty(len(self), np.int)
            self.coll.foreach_get('edge_index', self._cache_edge_indices)
        return self._cache_edge_indices

    @property
    def indices(self): # Array of int
        if self._cache_indices is None:
            self._cache_indices = np.empty(len(self), np.int)
            self.coll.foreach_get('index', self._cache_indices)
        return self._cache_indices

    @property
    def normals(self): # Array of V3
        if self._cache_normals is None:
            self._cache_normals = np.empty(len(self)*3, np.float)
            self.coll.foreach_get('normal', self._cache_normals)
            self._cache_normals = self._cache_normals.reshape(len(self), 3)
        return self._cache_normals

    @property
    def tangents(self): # Array of V3
        if self._cache_tangents is None:
            self._cache_tangents = np.empty(len(self)*3, np.float)
            self.coll.foreach_get('tangent', self._cache_tangents)
            self._cache_tangents = self._cache_tangents.reshape(len(self), 3)
        return self._cache_tangents

    @property
    def vertex_indices(self): # Array of int
        if self._cache_vertex_indices is None:
            self._cache_vertex_indices = np.empty(len(self), np.int)
            self.coll.foreach_get('vertex_index', self._cache_vertex_indices)
        return self._cache_vertex_indices

#================================================================================
# Polygon class wrapper

class WPolygon(Wrapper):

    @property
    def area(self): # float
        return self.obj.area

    @property
    def center(self): # V3
        return self.obj.center

    @property
    def hide(self): # bool
        return self.obj.hide

    @hide.setter
    def hide(self, value): # bool
        self.obj.hide = value

    @property
    def index(self): # int
        return self.obj.index

    @property
    def loop_start(self): # int
        return self.obj.loop_start

    @property
    def loop_total(self): # int
        return self.obj.loop_total

    @property
    def material_index(self): # int
        return self.obj.material_index

    @material_index.setter
    def material_index(self, value): # int
        self.obj.material_index = value

    @property
    def normal(self): # V3
        return self.obj.normal

    @property
    def use_smooth(self): # bool
        return self.obj.use_smooth

    @use_smooth.setter
    def use_smooth(self, value): # bool
        self.obj.use_smooth = value

    @property
    def vertices(self): # array
        return self.obj.vertices

#================================================================================
# Array of WPolygons

class WPolygons(CollWrapper):
    def __init__(self, coll, wowner):
        super().__init__(coll, wowner, WPolygon)
        self._cache_areas                   = None
        self._cache_centers                 = None
        self._cache_hides                   = None
        self._cache_indices                 = None
        self._cache_loop_starts             = None
        self._cache_loop_totals             = None
        self._cache_material_indices        = None
        self._cache_normals                 = None
        self._cache_use_smooths             = None

    def erase_cache(self):
        super().erase_cache()
        self._cache_areas                   = None
        self._cache_centers                 = None
        self._cache_hides                   = None
        self._cache_indices                 = None
        self._cache_loop_starts             = None
        self._cache_loop_totals             = None
        self._cache_material_indices        = None
        self._cache_normals                 = None
        self._cache_use_smooths             = None

    @property
    def areas(self): # Array of float
        if self._cache_areas is None:
            self._cache_areas = np.empty(len(self), np.float)
            self.coll.foreach_get('area', self._cache_areas)
        return self._cache_areas

    @property
    def centers(self): # Array of V3
        if self._cache_centers is None:
            self._cache_centers = np.empty(len(self)*3, np.float)
            self.coll.foreach_get('center', self._cache_centers)
            self._cache_centers = self._cache_centers.reshape(len(self), 3)
        return self._cache_centers

    @property
    def hides(self): # Array of bool
        if self._cache_hides is None:
            self._cache_hides = np.empty(len(self), np.bool)
            self.coll.foreach_get('hide', self._cache_hides)
        return self._cache_hides

    @hides.setter
    def hides(self, values): # Arrayf of bool
        self._cache_hides = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('hide', self._cache_hides)

    @property
    def indices(self): # Array of int
        if self._cache_indices is None:
            self._cache_indices = np.empty(len(self), np.int)
            self.coll.foreach_get('index', self._cache_indices)
        return self._cache_indices

    @property
    def loop_starts(self): # Array of int
        if self._cache_loop_starts is None:
            self._cache_loop_starts = np.empty(len(self), np.int)
            self.coll.foreach_get('loop_start', self._cache_loop_starts)
        return self._cache_loop_starts

    @property
    def loop_totals(self): # Array of int
        if self._cache_loop_totals is None:
            self._cache_loop_totals = np.empty(len(self), np.int)
            self.coll.foreach_get('loop_total', self._cache_loop_totals)
        return self._cache_loop_totals

    @property
    def material_indices(self): # Array of int
        if self._cache_material_indices is None:
            self._cache_material_indices = np.empty(len(self), np.int)
            self.coll.foreach_get('material_index', self._cache_material_indices)
        return self._cache_material_indices

    @material_indices.setter
    def material_indices(self, values): # Arrayf of int
        self._cache_material_indices = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('material_index', self._cache_material_indices)

    @property
    def normals(self): # Array of V3
        if self._cache_normals is None:
            self._cache_normals = np.empty(len(self)*3, np.float)
            self.coll.foreach_get('normal', self._cache_normals)
            self._cache_normals = self._cache_normals.reshape(len(self), 3)
        return self._cache_normals

    @property
    def use_smooths(self): # Array of bool
        if self._cache_use_smooths is None:
            self._cache_use_smooths = np.empty(len(self), np.bool)
            self.coll.foreach_get('use_smooth', self._cache_use_smooths)
        return self._cache_use_smooths

    @use_smooths.setter
    def use_smooths(self, values): # Arrayf of bool
        self._cache_use_smooths = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('use_smooth', self._cache_use_smooths)

#================================================================================
# Mesh class wrapper

class WMesh(Wrapper):

    def __init__(self, obj, wowner):
        super().__init__(self, obj, wowner)
        self.wvertices = WMeshVertices(self.obj.vertices, self)
        self.wedges    = WEdges(self.obj.edges, self)
        self.wloops    = WLoops(self.obj.loops, self)
        self.wpolygons = WPolygons(self.obj.polygons, self)

    @property
    def auto_smooth_angle(self): # float
        return self.obj.auto_smooth_angle

    @auto_smooth_angle.setter
    def auto_smooth_angle(self, value): # float
        self.obj.auto_smooth_angle = value

    @property
    def auto_texspace(self): # bool
        return self.obj.auto_texspace

    @auto_texspace.setter
    def auto_texspace(self, value): # bool
        self.obj.auto_texspace = value

    @property
    def use_auto_smooth(self): # bool
        return self.obj.use_auto_smooth

    @use_auto_smooth.setter
    def use_auto_smooth(self, value): # bool
        self.obj.use_auto_smooth = value

    @property
    def use_auto_texspace(self): # bool
        return self.obj.use_auto_texspace

    @use_auto_texspace.setter
    def use_auto_texspace(self, value): # bool
        self.obj.use_auto_texspace = value

#================================================================================
# Array of WMeshs

class WMeshs(CollWrapper):
    def __init__(self, coll, wowner):
        super().__init__(coll, wowner, WMesh)
        self._cache_auto_smooth_angles      = None
        self._cache_auto_texspaces          = None
        self._cache_use_auto_smooths        = None
        self._cache_use_auto_texspaces      = None

    def erase_cache(self):
        super().erase_cache()
        self._cache_auto_smooth_angles      = None
        self._cache_auto_texspaces          = None
        self._cache_use_auto_smooths        = None
        self._cache_use_auto_texspaces      = None

    @property
    def auto_smooth_angles(self): # Array of float
        if self._cache_auto_smooth_angles is None:
            self._cache_auto_smooth_angles = np.empty(len(self), np.float)
            self.coll.foreach_get('auto_smooth_angle', self._cache_auto_smooth_angles)
        return self._cache_auto_smooth_angles

    @auto_smooth_angles.setter
    def auto_smooth_angles(self, values): # Arrayf of float
        self._cache_auto_smooth_angles = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('auto_smooth_angle', self._cache_auto_smooth_angles)

    @property
    def auto_texspaces(self): # Array of bool
        if self._cache_auto_texspaces is None:
            self._cache_auto_texspaces = np.empty(len(self), np.bool)
            self.coll.foreach_get('auto_texspace', self._cache_auto_texspaces)
        return self._cache_auto_texspaces

    @auto_texspaces.setter
    def auto_texspaces(self, values): # Arrayf of bool
        self._cache_auto_texspaces = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('auto_texspace', self._cache_auto_texspaces)

    @property
    def use_auto_smooths(self): # Array of bool
        if self._cache_use_auto_smooths is None:
            self._cache_use_auto_smooths = np.empty(len(self), np.bool)
            self.coll.foreach_get('use_auto_smooth', self._cache_use_auto_smooths)
        return self._cache_use_auto_smooths

    @use_auto_smooths.setter
    def use_auto_smooths(self, values): # Arrayf of bool
        self._cache_use_auto_smooths = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('use_auto_smooth', self._cache_use_auto_smooths)

    @property
    def use_auto_texspaces(self): # Array of bool
        if self._cache_use_auto_texspaces is None:
            self._cache_use_auto_texspaces = np.empty(len(self), np.bool)
            self.coll.foreach_get('use_auto_texspace', self._cache_use_auto_texspaces)
        return self._cache_use_auto_texspaces

    @use_auto_texspaces.setter
    def use_auto_texspaces(self, values): # Arrayf of bool
        self._cache_use_auto_texspaces = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('use_auto_texspace', self._cache_use_auto_texspaces)

#================================================================================
# SplineBezierPoint class wrapper

class WSplineBezierPoint(Wrapper):

    @property
    def co(self): # V3
        return self.obj.co

    @property
    def x(self):
        return self.obj.co[0]

    @property
    def y(self):
        return self.obj.co[1]

    @property
    def z(self):
        return self.obj.co[2]

    @co.setter
    def co(self, value): # V3
        self.obj.co = value

    @x.setter
    def x(self, value):
        self.obj.co[0] = value

    @y.setter
    def y(self, value):
        self.obj.co[1] = value

    @z.setter
    def z(self, value):
        self.obj.co[2] = value

    @property
    def handle_left_type(self): # str
        return self.obj.handle_left_type

    @handle_left_type.setter
    def handle_left_type(self, value): # str
        self.obj.handle_left_type = value

    @property
    def handle_left(self): # V3
        return self.obj.handle_left

    @property
    def lx(self):
        return self.obj.handle_left[0]

    @property
    def ly(self):
        return self.obj.handle_left[1]

    @property
    def lz(self):
        return self.obj.handle_left[2]

    @handle_left.setter
    def handle_left(self, value): # V3
        self.obj.handle_left = value

    @lx.setter
    def lx(self, value):
        self.obj.handle_left[0] = value

    @ly.setter
    def ly(self, value):
        self.obj.handle_left[1] = value

    @lz.setter
    def lz(self, value):
        self.obj.handle_left[2] = value

    @property
    def handle_right_type(self): # str
        return self.obj.handle_right_type

    @handle_right_type.setter
    def handle_right_type(self, value): # str
        self.obj.handle_right_type = value

    @property
    def handle_right(self): # V3
        return self.obj.handle_right

    @property
    def rx(self):
        return self.obj.handle_right[0]

    @property
    def ry(self):
        return self.obj.handle_right[1]

    @property
    def rz(self):
        return self.obj.handle_right[2]

    @handle_right.setter
    def handle_right(self, value): # V3
        self.obj.handle_right = value

    @rx.setter
    def rx(self, value):
        self.obj.handle_right[0] = value

    @ry.setter
    def ry(self, value):
        self.obj.handle_right[1] = value

    @rz.setter
    def rz(self, value):
        self.obj.handle_right[2] = value

    @property
    def hide(self): # bool
        return self.obj.hide

    @hide.setter
    def hide(self, value): # bool
        self.obj.hide = value

    @property
    def radius(self): # float
        return self.obj.radius

    @radius.setter
    def radius(self, value): # float
        self.obj.radius = value

    @property
    def tilt(self): # int
        return self.obj.tilt

    @tilt.setter
    def tilt(self, value): # int
        self.obj.tilt = value

    @property
    def weight_softbody(self): # float
        return self.obj.weight_softbody

    @weight_softbody.setter
    def weight_softbody(self, value): # float
        self.obj.weight_softbody = value

#================================================================================
# Array of WSplineBezierPoints

class WSplineBezierPoints(CollWrapper):
    def __init__(self, coll, wowner):
        super().__init__(coll, wowner, WSplineBezierPoint)
        self._cache_cos                     = None
        self._cache_handle_left_types       = None
        self._cache_handle_lefts            = None
        self._cache_handle_right_types      = None
        self._cache_handle_rights           = None
        self._cache_hides                   = None
        self._cache_radius_s                = None
        self._cache_tilts                   = None
        self._cache_weight_softbodys        = None

    def erase_cache(self):
        super().erase_cache()
        self._cache_cos                     = None
        self._cache_handle_left_types       = None
        self._cache_handle_lefts            = None
        self._cache_handle_right_types      = None
        self._cache_handle_rights           = None
        self._cache_hides                   = None
        self._cache_radius_s                = None
        self._cache_tilts                   = None
        self._cache_weight_softbodys        = None

    @property
    def cos(self): # Array of V3
        if self._cache_cos is None:
            self._cache_cos = np.empty(len(self)*3, np.float)
            self.coll.foreach_get('co', self._cache_cos)
            self._cache_cos = self._cache_cos.reshape(len(self), 3)
        return self._cache_cos

    @cos.setter
    def cos(self, values): # Arrayf of V3
        self._cache_cos = to_array(values, (len(self), 3), f'3-vector or array of {len(self)} 3-vectors')
        self.coll.foreach_set('co', self._cache_cos.reshape(len(self) * 3))

    # xyzw access to cos

    @property
    def xs(self): 
        return self.cos[:, 0]

    @xs.setter
    def xs(self, values):
        self.cos[:, 0] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.cos = self._cache_cos

    @property
    def ys(self): 
        return self.cos[:, 1]

    @ys.setter
    def ys(self, values):
        self.cos[:, 1] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.cos = self._cache_cos

    @property
    def zs(self): 
        return self.cos[:, 2]

    @zs.setter
    def zs(self, values):
        self.cos[:, 2] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.cos = self._cache_cos

    @property
    def handle_left_types(self): # Array of str
        if self._cache_handle_left_types is None:
            self._cache_handle_left_types = np.empty(len(self), np.object)
            for i in range(len(self)):
                self._cache_handle_left_types[i] = self[i].handle_left_type
        return self._cache_handle_left_types

    @handle_left_types.setter
    def handle_left_types(self, values): # Arrayf of str
        self._cache_handle_left_types = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].handle_left_type = self._cache_handle_left_types[i]

    @property
    def handle_lefts(self): # Array of V3
        if self._cache_handle_lefts is None:
            self._cache_handle_lefts = np.empty(len(self)*3, np.float)
            self.coll.foreach_get('handle_left', self._cache_handle_lefts)
            self._cache_handle_lefts = self._cache_handle_lefts.reshape(len(self), 3)
        return self._cache_handle_lefts

    @handle_lefts.setter
    def handle_lefts(self, values): # Arrayf of V3
        self._cache_handle_lefts = to_array(values, (len(self), 3), f'3-vector or array of {len(self)} 3-vectors')
        self.coll.foreach_set('handle_left', self._cache_handle_lefts.reshape(len(self) * 3))

    # xyzw access to handle_lefts

    @property
    def lxs(self): 
        return self.handle_lefts[:, 0]

    @lxs.setter
    def lxs(self, values):
        self.handle_lefts[:, 0] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.handle_lefts = self._cache_handle_lefts

    @property
    def lys(self): 
        return self.handle_lefts[:, 1]

    @lys.setter
    def lys(self, values):
        self.handle_lefts[:, 1] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.handle_lefts = self._cache_handle_lefts

    @property
    def lzs(self): 
        return self.handle_lefts[:, 2]

    @lzs.setter
    def lzs(self, values):
        self.handle_lefts[:, 2] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.handle_lefts = self._cache_handle_lefts

    @property
    def handle_right_types(self): # Array of str
        if self._cache_handle_right_types is None:
            self._cache_handle_right_types = np.empty(len(self), np.object)
            for i in range(len(self)):
                self._cache_handle_right_types[i] = self[i].handle_right_type
        return self._cache_handle_right_types

    @handle_right_types.setter
    def handle_right_types(self, values): # Arrayf of str
        self._cache_handle_right_types = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].handle_right_type = self._cache_handle_right_types[i]

    @property
    def handle_rights(self): # Array of V3
        if self._cache_handle_rights is None:
            self._cache_handle_rights = np.empty(len(self)*3, np.float)
            self.coll.foreach_get('handle_right', self._cache_handle_rights)
            self._cache_handle_rights = self._cache_handle_rights.reshape(len(self), 3)
        return self._cache_handle_rights

    @handle_rights.setter
    def handle_rights(self, values): # Arrayf of V3
        self._cache_handle_rights = to_array(values, (len(self), 3), f'3-vector or array of {len(self)} 3-vectors')
        self.coll.foreach_set('handle_right', self._cache_handle_rights.reshape(len(self) * 3))

    # xyzw access to handle_rights

    @property
    def rxs(self): 
        return self.handle_rights[:, 0]

    @rxs.setter
    def rxs(self, values):
        self.handle_rights[:, 0] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.handle_rights = self._cache_handle_rights

    @property
    def rys(self): 
        return self.handle_rights[:, 1]

    @rys.setter
    def rys(self, values):
        self.handle_rights[:, 1] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.handle_rights = self._cache_handle_rights

    @property
    def rzs(self): 
        return self.handle_rights[:, 2]

    @rzs.setter
    def rzs(self, values):
        self.handle_rights[:, 2] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.handle_rights = self._cache_handle_rights

    @property
    def hides(self): # Array of bool
        if self._cache_hides is None:
            self._cache_hides = np.empty(len(self), np.bool)
            self.coll.foreach_get('hide', self._cache_hides)
        return self._cache_hides

    @hides.setter
    def hides(self, values): # Arrayf of bool
        self._cache_hides = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('hide', self._cache_hides)

    @property
    def radius_s(self): # Array of float
        if self._cache_radius_s is None:
            self._cache_radius_s = np.empty(len(self), np.float)
            self.coll.foreach_get('radius', self._cache_radius_s)
        return self._cache_radius_s

    @radius_s.setter
    def radius_s(self, values): # Arrayf of float
        self._cache_radius_s = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('radius', self._cache_radius_s)

    @property
    def tilts(self): # Array of int
        if self._cache_tilts is None:
            self._cache_tilts = np.empty(len(self), np.int)
            self.coll.foreach_get('tilt', self._cache_tilts)
        return self._cache_tilts

    @tilts.setter
    def tilts(self, values): # Arrayf of int
        self._cache_tilts = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('tilt', self._cache_tilts)

    @property
    def weight_softbodys(self): # Array of float
        if self._cache_weight_softbodys is None:
            self._cache_weight_softbodys = np.empty(len(self), np.float)
            self.coll.foreach_get('weight_softbody', self._cache_weight_softbodys)
        return self._cache_weight_softbodys

    @weight_softbodys.setter
    def weight_softbodys(self, values): # Arrayf of float
        self._cache_weight_softbodys = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('weight_softbody', self._cache_weight_softbodys)

#================================================================================
# SplinePoint class wrapper

class WSplinePoint(Wrapper):

    @property
    def co(self): # V3
        return self.obj.co

    @property
    def x(self):
        return self.obj.co[0]

    @property
    def y(self):
        return self.obj.co[1]

    @property
    def z(self):
        return self.obj.co[2]

    @co.setter
    def co(self, value): # V3
        self.obj.co = value

    @x.setter
    def x(self, value):
        self.obj.co[0] = value

    @y.setter
    def y(self, value):
        self.obj.co[1] = value

    @z.setter
    def z(self, value):
        self.obj.co[2] = value

    @property
    def radius(self): # float
        return self.obj.radius

    @radius.setter
    def radius(self, value): # float
        self.obj.radius = value

    @property
    def tilt(self): # float
        return self.obj.tilt

    @tilt.setter
    def tilt(self, value): # float
        self.obj.tilt = value

    @property
    def weight_softbody(self): # float
        return self.obj.weight_softbody

    @weight_softbody.setter
    def weight_softbody(self, value): # float
        self.obj.weight_softbody = value

    @property
    def weight(self): # float
        return self.obj.weight

    @weight.setter
    def weight(self, value): # float
        self.obj.weight = value

#================================================================================
# Array of WSplinePoints

class WSplinePoints(CollWrapper):
    def __init__(self, coll, wowner):
        super().__init__(coll, wowner, WSplinePoint)
        self._cache_cos                     = None
        self._cache_radius_s                = None
        self._cache_tilts                   = None
        self._cache_weight_softbodys        = None
        self._cache_weights                 = None

    def erase_cache(self):
        super().erase_cache()
        self._cache_cos                     = None
        self._cache_radius_s                = None
        self._cache_tilts                   = None
        self._cache_weight_softbodys        = None
        self._cache_weights                 = None

    @property
    def cos(self): # Array of V3
        if self._cache_cos is None:
            self._cache_cos = np.empty(len(self)*3, np.float)
            self.coll.foreach_get('co', self._cache_cos)
            self._cache_cos = self._cache_cos.reshape(len(self), 3)
        return self._cache_cos

    @cos.setter
    def cos(self, values): # Arrayf of V3
        self._cache_cos = to_array(values, (len(self), 3), f'3-vector or array of {len(self)} 3-vectors')
        self.coll.foreach_set('co', self._cache_cos.reshape(len(self) * 3))

    # xyzw access to cos

    @property
    def xs(self): 
        return self.cos[:, 0]

    @xs.setter
    def xs(self, values):
        self.cos[:, 0] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.cos = self._cache_cos

    @property
    def ys(self): 
        return self.cos[:, 1]

    @ys.setter
    def ys(self, values):
        self.cos[:, 1] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.cos = self._cache_cos

    @property
    def zs(self): 
        return self.cos[:, 2]

    @zs.setter
    def zs(self, values):
        self.cos[:, 2] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.cos = self._cache_cos

    @property
    def radius_s(self): # Array of float
        if self._cache_radius_s is None:
            self._cache_radius_s = np.empty(len(self), np.float)
            self.coll.foreach_get('radius', self._cache_radius_s)
        return self._cache_radius_s

    @radius_s.setter
    def radius_s(self, values): # Arrayf of float
        self._cache_radius_s = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('radius', self._cache_radius_s)

    @property
    def tilts(self): # Array of float
        if self._cache_tilts is None:
            self._cache_tilts = np.empty(len(self), np.float)
            self.coll.foreach_get('tilt', self._cache_tilts)
        return self._cache_tilts

    @tilts.setter
    def tilts(self, values): # Arrayf of float
        self._cache_tilts = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('tilt', self._cache_tilts)

    @property
    def weight_softbodys(self): # Array of float
        if self._cache_weight_softbodys is None:
            self._cache_weight_softbodys = np.empty(len(self), np.float)
            self.coll.foreach_get('weight_softbody', self._cache_weight_softbodys)
        return self._cache_weight_softbodys

    @weight_softbodys.setter
    def weight_softbodys(self, values): # Arrayf of float
        self._cache_weight_softbodys = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('weight_softbody', self._cache_weight_softbodys)

    @property
    def weights(self): # Array of float
        if self._cache_weights is None:
            self._cache_weights = np.empty(len(self), np.float)
            self.coll.foreach_get('weight', self._cache_weights)
        return self._cache_weights

    @weights.setter
    def weights(self, values): # Arrayf of float
        self._cache_weights = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('weight', self._cache_weights)

#================================================================================
# Spline class wrapper

class WSpline(Wrapper):

    def __init__(self, obj, wowner):
        super().__init__(self, obj, wowner)
        self.wbezier_points = WBezierSplinePoints(self.obj.bezier_points, self)
        self.wpoints        = WBezierSplinePoints(self.obj.points, self)

    @property
    def character_index(self): # int
        return self.obj.character_index

    @character_index.setter
    def character_index(self, value): # int
        self.obj.character_index = value

    @property
    def material_index(self): # int
        return self.obj.material_index

    @material_index.setter
    def material_index(self, value): # int
        self.obj.material_index = value

    @property
    def order_u(self): # int
        return self.obj.order_u

    @order_u.setter
    def order_u(self, value): # int
        self.obj.order_u = value

    @property
    def order_v(self): # int
        return self.obj.order_v

    @order_v.setter
    def order_v(self, value): # int
        self.obj.order_v = value

    @property
    def point_count_u(self): # int
        return self.obj.point_count_u

    @point_count_u.setter
    def point_count_u(self, value): # int
        self.obj.point_count_u = value

    @property
    def point_count_v(self): # int
        return self.obj.point_count_v

    @point_count_v.setter
    def point_count_v(self, value): # int
        self.obj.point_count_v = value

    @property
    def radius_interpolation(self): # str
        return self.obj.radius_interpolation

    @radius_interpolation.setter
    def radius_interpolation(self, value): # str
        self.obj.radius_interpolation = value

    @property
    def resolution_u(self): # int
        return self.obj.resolution_u

    @resolution_u.setter
    def resolution_u(self, value): # int
        self.obj.resolution_u = value

    @property
    def resolution_v(self): # int
        return self.obj.resolution_v

    @resolution_v.setter
    def resolution_v(self, value): # int
        self.obj.resolution_v = value

    @property
    def type(self): # str
        return self.obj.type

    @type.setter
    def type(self, value): # str
        self.obj.type = value

    @property
    def use_bezier_u(self): # bool
        return self.obj.use_bezier_u

    @use_bezier_u.setter
    def use_bezier_u(self, value): # bool
        self.obj.use_bezier_u = value

    @property
    def use_bezier_v(self): # bool
        return self.obj.use_bezier_v

    @use_bezier_v.setter
    def use_bezier_v(self, value): # bool
        self.obj.use_bezier_v = value

    @property
    def use_cyclic_u(self): # bool
        return self.obj.use_cyclic_u

    @use_cyclic_u.setter
    def use_cyclic_u(self, value): # bool
        self.obj.use_cyclic_u = value

    @property
    def use_cyclic_v(self): # bool
        return self.obj.use_cyclic_v

    @use_cyclic_v.setter
    def use_cyclic_v(self, value): # bool
        self.obj.use_cyclic_v = value

    @property
    def use_endpoint_u(self): # bool
        return self.obj.use_endpoint_u

    @use_endpoint_u.setter
    def use_endpoint_u(self, value): # bool
        self.obj.use_endpoint_u = value

    @property
    def use_endpoint_v(self): # bool
        return self.obj.use_endpoint_v

    @use_endpoint_v.setter
    def use_endpoint_v(self, value): # bool
        self.obj.use_endpoint_v = value

    @property
    def use_smooth(self): # bool
        return self.obj.use_smooth

    @use_smooth.setter
    def use_smooth(self, value): # bool
        self.obj.use_smooth = value

#================================================================================
# Array of WSplines

class WSplines(CollWrapper):
    def __init__(self, coll, wowner):
        super().__init__(coll, wowner, WSpline)
        self._cache_character_indices       = None
        self._cache_material_indices        = None
        self._cache_order_us                = None
        self._cache_order_vs                = None
        self._cache_point_count_us          = None
        self._cache_point_count_vs          = None
        self._cache_radius_interpolations   = None
        self._cache_resolution_us           = None
        self._cache_resolution_vs           = None
        self._cache_types                   = None
        self._cache_use_bezier_us           = None
        self._cache_use_bezier_vs           = None
        self._cache_use_cyclic_us           = None
        self._cache_use_cyclic_vs           = None
        self._cache_use_endpoint_us         = None
        self._cache_use_endpoint_vs         = None
        self._cache_use_smooths             = None

    def erase_cache(self):
        super().erase_cache()
        self._cache_character_indices       = None
        self._cache_material_indices        = None
        self._cache_order_us                = None
        self._cache_order_vs                = None
        self._cache_point_count_us          = None
        self._cache_point_count_vs          = None
        self._cache_radius_interpolations   = None
        self._cache_resolution_us           = None
        self._cache_resolution_vs           = None
        self._cache_types                   = None
        self._cache_use_bezier_us           = None
        self._cache_use_bezier_vs           = None
        self._cache_use_cyclic_us           = None
        self._cache_use_cyclic_vs           = None
        self._cache_use_endpoint_us         = None
        self._cache_use_endpoint_vs         = None
        self._cache_use_smooths             = None

    @property
    def character_indices(self): # Array of int
        if self._cache_character_indices is None:
            self._cache_character_indices = np.empty(len(self), np.int)
            self.coll.foreach_get('character_index', self._cache_character_indices)
        return self._cache_character_indices

    @character_indices.setter
    def character_indices(self, values): # Arrayf of int
        self._cache_character_indices = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('character_index', self._cache_character_indices)

    @property
    def material_indices(self): # Array of int
        if self._cache_material_indices is None:
            self._cache_material_indices = np.empty(len(self), np.int)
            self.coll.foreach_get('material_index', self._cache_material_indices)
        return self._cache_material_indices

    @material_indices.setter
    def material_indices(self, values): # Arrayf of int
        self._cache_material_indices = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('material_index', self._cache_material_indices)

    @property
    def order_us(self): # Array of int
        if self._cache_order_us is None:
            self._cache_order_us = np.empty(len(self), np.int)
            self.coll.foreach_get('order_u', self._cache_order_us)
        return self._cache_order_us

    @order_us.setter
    def order_us(self, values): # Arrayf of int
        self._cache_order_us = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('order_u', self._cache_order_us)

    @property
    def order_vs(self): # Array of int
        if self._cache_order_vs is None:
            self._cache_order_vs = np.empty(len(self), np.int)
            self.coll.foreach_get('order_v', self._cache_order_vs)
        return self._cache_order_vs

    @order_vs.setter
    def order_vs(self, values): # Arrayf of int
        self._cache_order_vs = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('order_v', self._cache_order_vs)

    @property
    def point_count_us(self): # Array of int
        if self._cache_point_count_us is None:
            self._cache_point_count_us = np.empty(len(self), np.int)
            self.coll.foreach_get('point_count_u', self._cache_point_count_us)
        return self._cache_point_count_us

    @point_count_us.setter
    def point_count_us(self, values): # Arrayf of int
        self._cache_point_count_us = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('point_count_u', self._cache_point_count_us)

    @property
    def point_count_vs(self): # Array of int
        if self._cache_point_count_vs is None:
            self._cache_point_count_vs = np.empty(len(self), np.int)
            self.coll.foreach_get('point_count_v', self._cache_point_count_vs)
        return self._cache_point_count_vs

    @point_count_vs.setter
    def point_count_vs(self, values): # Arrayf of int
        self._cache_point_count_vs = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('point_count_v', self._cache_point_count_vs)

    @property
    def radius_interpolations(self): # Array of str
        if self._cache_radius_interpolations is None:
            self._cache_radius_interpolations = np.empty(len(self), np.object)
            for i in range(len(self)):
                self._cache_radius_interpolations[i] = self[i].radius_interpolation
        return self._cache_radius_interpolations

    @radius_interpolations.setter
    def radius_interpolations(self, values): # Arrayf of str
        self._cache_radius_interpolations = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].radius_interpolation = self._cache_radius_interpolations[i]

    @property
    def resolution_us(self): # Array of int
        if self._cache_resolution_us is None:
            self._cache_resolution_us = np.empty(len(self), np.int)
            self.coll.foreach_get('resolution_u', self._cache_resolution_us)
        return self._cache_resolution_us

    @resolution_us.setter
    def resolution_us(self, values): # Arrayf of int
        self._cache_resolution_us = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('resolution_u', self._cache_resolution_us)

    @property
    def resolution_vs(self): # Array of int
        if self._cache_resolution_vs is None:
            self._cache_resolution_vs = np.empty(len(self), np.int)
            self.coll.foreach_get('resolution_v', self._cache_resolution_vs)
        return self._cache_resolution_vs

    @resolution_vs.setter
    def resolution_vs(self, values): # Arrayf of int
        self._cache_resolution_vs = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('resolution_v', self._cache_resolution_vs)

    @property
    def types(self): # Array of str
        if self._cache_types is None:
            self._cache_types = np.empty(len(self), np.object)
            for i in range(len(self)):
                self._cache_types[i] = self[i].type
        return self._cache_types

    @types.setter
    def types(self, values): # Arrayf of str
        self._cache_types = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].type = self._cache_types[i]

    @property
    def use_bezier_us(self): # Array of bool
        if self._cache_use_bezier_us is None:
            self._cache_use_bezier_us = np.empty(len(self), np.bool)
            self.coll.foreach_get('use_bezier_u', self._cache_use_bezier_us)
        return self._cache_use_bezier_us

    @use_bezier_us.setter
    def use_bezier_us(self, values): # Arrayf of bool
        self._cache_use_bezier_us = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('use_bezier_u', self._cache_use_bezier_us)

    @property
    def use_bezier_vs(self): # Array of bool
        if self._cache_use_bezier_vs is None:
            self._cache_use_bezier_vs = np.empty(len(self), np.bool)
            self.coll.foreach_get('use_bezier_v', self._cache_use_bezier_vs)
        return self._cache_use_bezier_vs

    @use_bezier_vs.setter
    def use_bezier_vs(self, values): # Arrayf of bool
        self._cache_use_bezier_vs = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('use_bezier_v', self._cache_use_bezier_vs)

    @property
    def use_cyclic_us(self): # Array of bool
        if self._cache_use_cyclic_us is None:
            self._cache_use_cyclic_us = np.empty(len(self), np.bool)
            self.coll.foreach_get('use_cyclic_u', self._cache_use_cyclic_us)
        return self._cache_use_cyclic_us

    @use_cyclic_us.setter
    def use_cyclic_us(self, values): # Arrayf of bool
        self._cache_use_cyclic_us = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('use_cyclic_u', self._cache_use_cyclic_us)

    @property
    def use_cyclic_vs(self): # Array of bool
        if self._cache_use_cyclic_vs is None:
            self._cache_use_cyclic_vs = np.empty(len(self), np.bool)
            self.coll.foreach_get('use_cyclic_v', self._cache_use_cyclic_vs)
        return self._cache_use_cyclic_vs

    @use_cyclic_vs.setter
    def use_cyclic_vs(self, values): # Arrayf of bool
        self._cache_use_cyclic_vs = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('use_cyclic_v', self._cache_use_cyclic_vs)

    @property
    def use_endpoint_us(self): # Array of bool
        if self._cache_use_endpoint_us is None:
            self._cache_use_endpoint_us = np.empty(len(self), np.bool)
            self.coll.foreach_get('use_endpoint_u', self._cache_use_endpoint_us)
        return self._cache_use_endpoint_us

    @use_endpoint_us.setter
    def use_endpoint_us(self, values): # Arrayf of bool
        self._cache_use_endpoint_us = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('use_endpoint_u', self._cache_use_endpoint_us)

    @property
    def use_endpoint_vs(self): # Array of bool
        if self._cache_use_endpoint_vs is None:
            self._cache_use_endpoint_vs = np.empty(len(self), np.bool)
            self.coll.foreach_get('use_endpoint_v', self._cache_use_endpoint_vs)
        return self._cache_use_endpoint_vs

    @use_endpoint_vs.setter
    def use_endpoint_vs(self, values): # Arrayf of bool
        self._cache_use_endpoint_vs = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('use_endpoint_v', self._cache_use_endpoint_vs)

    @property
    def use_smooths(self): # Array of bool
        if self._cache_use_smooths is None:
            self._cache_use_smooths = np.empty(len(self), np.bool)
            self.coll.foreach_get('use_smooth', self._cache_use_smooths)
        return self._cache_use_smooths

    @use_smooths.setter
    def use_smooths(self, values): # Arrayf of bool
        self._cache_use_smooths = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('use_smooth', self._cache_use_smooths)

#================================================================================
# Curve class wrapper

class WCurve(Wrapper):

    def __init__(self, obj, wowner):
        super().__init__(self, obj, wowner)
        self.wsplines = WSplines(self.obj.splines, self)

    @property
    def bevel_depth(self): # float
        return self.obj.bevel_depth

    @bevel_depth.setter
    def bevel_depth(self, value): # float
        self.obj.bevel_depth = value

    @property
    def bevel_factor_end(self): # float
        return self.obj.bevel_factor_end

    @bevel_factor_end.setter
    def bevel_factor_end(self, value): # float
        self.obj.bevel_factor_end = value

    @property
    def bevel_factor_start(self): # float
        return self.obj.bevel_factor_start

    @bevel_factor_start.setter
    def bevel_factor_start(self, value): # float
        self.obj.bevel_factor_start = value

    @property
    def bevel_object(self): # object
        return self.obj.bevel_object

    @bevel_object.setter
    def bevel_object(self, value): # object
        self.obj.bevel_object = value

    @property
    def bevel_resolution(self): # int
        return self.obj.bevel_resolution

    @bevel_resolution.setter
    def bevel_resolution(self, value): # int
        self.obj.bevel_resolution = value

    @property
    def dimensions(self): # str
        return self.obj.dimensions

    @dimensions.setter
    def dimensions(self, value): # str
        self.obj.dimensions = value

    @property
    def eval_time(self): # float
        return self.obj.eval_time

    @eval_time.setter
    def eval_time(self, value): # float
        self.obj.eval_time = value

    @property
    def extrude(self): # float
        return self.obj.extrude

    @extrude.setter
    def extrude(self, value): # float
        self.obj.extrude = value

    @property
    def fill_mode(self): # str
        return self.obj.fill_mode

    @fill_mode.setter
    def fill_mode(self, value): # str
        self.obj.fill_mode = value

    @property
    def offset(self): # float
        return self.obj.offset

    @offset.setter
    def offset(self, value): # float
        self.obj.offset = value

    @property
    def path_duration(self): # int
        return self.obj.path_duration

    @path_duration.setter
    def path_duration(self, value): # int
        self.obj.path_duration = value

    @property
    def render_resolution_u(self): # int
        return self.obj.render_resolution_u

    @render_resolution_u.setter
    def render_resolution_u(self, value): # int
        self.obj.render_resolution_u = value

    @property
    def render_resolution_v(self): # int
        return self.obj.render_resolution_v

    @render_resolution_v.setter
    def render_resolution_v(self, value): # int
        self.obj.render_resolution_v = value

    @property
    def resolution_u(self): # int
        return self.obj.resolution_u

    @resolution_u.setter
    def resolution_u(self, value): # int
        self.obj.resolution_u = value

    @property
    def resolution_v(self): # int
        return self.obj.resolution_v

    @resolution_v.setter
    def resolution_v(self, value): # int
        self.obj.resolution_v = value

    @property
    def taper_object(self): # object
        return self.obj.taper_object

    @taper_object.setter
    def taper_object(self, value): # object
        self.obj.taper_object = value

    @property
    def twist_smooth(self): # float
        return self.obj.twist_smooth

    @twist_smooth.setter
    def twist_smooth(self, value): # float
        self.obj.twist_smooth = value

    @property
    def use_auto_texspace(self): # bool
        return self.obj.use_auto_texspace

    @use_auto_texspace.setter
    def use_auto_texspace(self, value): # bool
        self.obj.use_auto_texspace = value

    @property
    def use_deform_bounds(self): # bool
        return self.obj.use_deform_bounds

    @use_deform_bounds.setter
    def use_deform_bounds(self, value): # bool
        self.obj.use_deform_bounds = value

    @property
    def use_fill_caps(self): # bool
        return self.obj.use_fill_caps

    @use_fill_caps.setter
    def use_fill_caps(self, value): # bool
        self.obj.use_fill_caps = value

    @property
    def use_fill_deform(self): # bool
        return self.obj.use_fill_deform

    @use_fill_deform.setter
    def use_fill_deform(self, value): # bool
        self.obj.use_fill_deform = value

    @property
    def use_map_taper(self): # bool
        return self.obj.use_map_taper

    @use_map_taper.setter
    def use_map_taper(self, value): # bool
        self.obj.use_map_taper = value

    @property
    def use_path_follow(self): # bool
        return self.obj.use_path_follow

    @use_path_follow.setter
    def use_path_follow(self, value): # bool
        self.obj.use_path_follow = value

    @property
    def use_path(self): # bool
        return self.obj.use_path

    @use_path.setter
    def use_path(self, value): # bool
        self.obj.use_path = value

    @property
    def use_radius(self): # bool
        return self.obj.use_radius

    @use_radius.setter
    def use_radius(self, value): # bool
        self.obj.use_radius = value

    @property
    def use_stretch(self): # bool
        return self.obj.use_stretch

    @use_stretch.setter
    def use_stretch(self, value): # bool
        self.obj.use_stretch = value

#================================================================================
# Array of WCurves

class WCurves(CollWrapper):
    def __init__(self, coll, wowner):
        super().__init__(coll, wowner, WCurve)
        self._cache_bevel_depths            = None
        self._cache_bevel_factor_ends       = None
        self._cache_bevel_factor_starts     = None
        self._cache_bevel_resolutions       = None
        self._cache_dimensions_s            = None
        self._cache_eval_times              = None
        self._cache_extrudes                = None
        self._cache_fill_modes              = None
        self._cache_offsets                 = None
        self._cache_path_durations          = None
        self._cache_render_resolution_us    = None
        self._cache_render_resolution_vs    = None
        self._cache_resolution_us           = None
        self._cache_resolution_vs           = None
        self._cache_twist_smooths           = None
        self._cache_use_auto_texspaces      = None
        self._cache_use_deform_bounds_s     = None
        self._cache_use_fill_caps_s         = None
        self._cache_use_fill_deforms        = None
        self._cache_use_map_tapers          = None
        self._cache_use_path_follows        = None
        self._cache_use_paths               = None
        self._cache_use_radius_s            = None
        self._cache_use_stretchs            = None

    def erase_cache(self):
        super().erase_cache()
        self._cache_bevel_depths            = None
        self._cache_bevel_factor_ends       = None
        self._cache_bevel_factor_starts     = None
        self._cache_bevel_resolutions       = None
        self._cache_dimensions_s            = None
        self._cache_eval_times              = None
        self._cache_extrudes                = None
        self._cache_fill_modes              = None
        self._cache_offsets                 = None
        self._cache_path_durations          = None
        self._cache_render_resolution_us    = None
        self._cache_render_resolution_vs    = None
        self._cache_resolution_us           = None
        self._cache_resolution_vs           = None
        self._cache_twist_smooths           = None
        self._cache_use_auto_texspaces      = None
        self._cache_use_deform_bounds_s     = None
        self._cache_use_fill_caps_s         = None
        self._cache_use_fill_deforms        = None
        self._cache_use_map_tapers          = None
        self._cache_use_path_follows        = None
        self._cache_use_paths               = None
        self._cache_use_radius_s            = None
        self._cache_use_stretchs            = None

    @property
    def bevel_depths(self): # Array of float
        if self._cache_bevel_depths is None:
            self._cache_bevel_depths = np.empty(len(self), np.float)
            self.coll.foreach_get('bevel_depth', self._cache_bevel_depths)
        return self._cache_bevel_depths

    @bevel_depths.setter
    def bevel_depths(self, values): # Arrayf of float
        self._cache_bevel_depths = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('bevel_depth', self._cache_bevel_depths)

    @property
    def bevel_factor_ends(self): # Array of float
        if self._cache_bevel_factor_ends is None:
            self._cache_bevel_factor_ends = np.empty(len(self), np.float)
            self.coll.foreach_get('bevel_factor_end', self._cache_bevel_factor_ends)
        return self._cache_bevel_factor_ends

    @bevel_factor_ends.setter
    def bevel_factor_ends(self, values): # Arrayf of float
        self._cache_bevel_factor_ends = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('bevel_factor_end', self._cache_bevel_factor_ends)

    @property
    def bevel_factor_starts(self): # Array of float
        if self._cache_bevel_factor_starts is None:
            self._cache_bevel_factor_starts = np.empty(len(self), np.float)
            self.coll.foreach_get('bevel_factor_start', self._cache_bevel_factor_starts)
        return self._cache_bevel_factor_starts

    @bevel_factor_starts.setter
    def bevel_factor_starts(self, values): # Arrayf of float
        self._cache_bevel_factor_starts = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('bevel_factor_start', self._cache_bevel_factor_starts)

    @property
    def bevel_resolutions(self): # Array of int
        if self._cache_bevel_resolutions is None:
            self._cache_bevel_resolutions = np.empty(len(self), np.int)
            self.coll.foreach_get('bevel_resolution', self._cache_bevel_resolutions)
        return self._cache_bevel_resolutions

    @bevel_resolutions.setter
    def bevel_resolutions(self, values): # Arrayf of int
        self._cache_bevel_resolutions = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('bevel_resolution', self._cache_bevel_resolutions)

    @property
    def dimensions_s(self): # Array of str
        if self._cache_dimensions_s is None:
            self._cache_dimensions_s = np.empty(len(self), np.object)
            for i in range(len(self)):
                self._cache_dimensions_s[i] = self[i].dimensions
        return self._cache_dimensions_s

    @dimensions_s.setter
    def dimensions_s(self, values): # Arrayf of str
        self._cache_dimensions_s = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].dimensions = self._cache_dimensions_s[i]

    @property
    def eval_times(self): # Array of float
        if self._cache_eval_times is None:
            self._cache_eval_times = np.empty(len(self), np.float)
            self.coll.foreach_get('eval_time', self._cache_eval_times)
        return self._cache_eval_times

    @eval_times.setter
    def eval_times(self, values): # Arrayf of float
        self._cache_eval_times = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('eval_time', self._cache_eval_times)

    @property
    def extrudes(self): # Array of float
        if self._cache_extrudes is None:
            self._cache_extrudes = np.empty(len(self), np.float)
            self.coll.foreach_get('extrude', self._cache_extrudes)
        return self._cache_extrudes

    @extrudes.setter
    def extrudes(self, values): # Arrayf of float
        self._cache_extrudes = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('extrude', self._cache_extrudes)

    @property
    def fill_modes(self): # Array of str
        if self._cache_fill_modes is None:
            self._cache_fill_modes = np.empty(len(self), np.object)
            for i in range(len(self)):
                self._cache_fill_modes[i] = self[i].fill_mode
        return self._cache_fill_modes

    @fill_modes.setter
    def fill_modes(self, values): # Arrayf of str
        self._cache_fill_modes = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].fill_mode = self._cache_fill_modes[i]

    @property
    def offsets(self): # Array of float
        if self._cache_offsets is None:
            self._cache_offsets = np.empty(len(self), np.float)
            self.coll.foreach_get('offset', self._cache_offsets)
        return self._cache_offsets

    @offsets.setter
    def offsets(self, values): # Arrayf of float
        self._cache_offsets = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('offset', self._cache_offsets)

    @property
    def path_durations(self): # Array of int
        if self._cache_path_durations is None:
            self._cache_path_durations = np.empty(len(self), np.int)
            self.coll.foreach_get('path_duration', self._cache_path_durations)
        return self._cache_path_durations

    @path_durations.setter
    def path_durations(self, values): # Arrayf of int
        self._cache_path_durations = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('path_duration', self._cache_path_durations)

    @property
    def render_resolution_us(self): # Array of int
        if self._cache_render_resolution_us is None:
            self._cache_render_resolution_us = np.empty(len(self), np.int)
            self.coll.foreach_get('render_resolution_u', self._cache_render_resolution_us)
        return self._cache_render_resolution_us

    @render_resolution_us.setter
    def render_resolution_us(self, values): # Arrayf of int
        self._cache_render_resolution_us = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('render_resolution_u', self._cache_render_resolution_us)

    @property
    def render_resolution_vs(self): # Array of int
        if self._cache_render_resolution_vs is None:
            self._cache_render_resolution_vs = np.empty(len(self), np.int)
            self.coll.foreach_get('render_resolution_v', self._cache_render_resolution_vs)
        return self._cache_render_resolution_vs

    @render_resolution_vs.setter
    def render_resolution_vs(self, values): # Arrayf of int
        self._cache_render_resolution_vs = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('render_resolution_v', self._cache_render_resolution_vs)

    @property
    def resolution_us(self): # Array of int
        if self._cache_resolution_us is None:
            self._cache_resolution_us = np.empty(len(self), np.int)
            self.coll.foreach_get('resolution_u', self._cache_resolution_us)
        return self._cache_resolution_us

    @resolution_us.setter
    def resolution_us(self, values): # Arrayf of int
        self._cache_resolution_us = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('resolution_u', self._cache_resolution_us)

    @property
    def resolution_vs(self): # Array of int
        if self._cache_resolution_vs is None:
            self._cache_resolution_vs = np.empty(len(self), np.int)
            self.coll.foreach_get('resolution_v', self._cache_resolution_vs)
        return self._cache_resolution_vs

    @resolution_vs.setter
    def resolution_vs(self, values): # Arrayf of int
        self._cache_resolution_vs = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('resolution_v', self._cache_resolution_vs)

    @property
    def twist_smooths(self): # Array of float
        if self._cache_twist_smooths is None:
            self._cache_twist_smooths = np.empty(len(self), np.float)
            self.coll.foreach_get('twist_smooth', self._cache_twist_smooths)
        return self._cache_twist_smooths

    @twist_smooths.setter
    def twist_smooths(self, values): # Arrayf of float
        self._cache_twist_smooths = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('twist_smooth', self._cache_twist_smooths)

    @property
    def use_auto_texspaces(self): # Array of bool
        if self._cache_use_auto_texspaces is None:
            self._cache_use_auto_texspaces = np.empty(len(self), np.bool)
            self.coll.foreach_get('use_auto_texspace', self._cache_use_auto_texspaces)
        return self._cache_use_auto_texspaces

    @use_auto_texspaces.setter
    def use_auto_texspaces(self, values): # Arrayf of bool
        self._cache_use_auto_texspaces = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('use_auto_texspace', self._cache_use_auto_texspaces)

    @property
    def use_deform_bounds_s(self): # Array of bool
        if self._cache_use_deform_bounds_s is None:
            self._cache_use_deform_bounds_s = np.empty(len(self), np.bool)
            self.coll.foreach_get('use_deform_bounds', self._cache_use_deform_bounds_s)
        return self._cache_use_deform_bounds_s

    @use_deform_bounds_s.setter
    def use_deform_bounds_s(self, values): # Arrayf of bool
        self._cache_use_deform_bounds_s = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('use_deform_bounds', self._cache_use_deform_bounds_s)

    @property
    def use_fill_caps_s(self): # Array of bool
        if self._cache_use_fill_caps_s is None:
            self._cache_use_fill_caps_s = np.empty(len(self), np.bool)
            self.coll.foreach_get('use_fill_caps', self._cache_use_fill_caps_s)
        return self._cache_use_fill_caps_s

    @use_fill_caps_s.setter
    def use_fill_caps_s(self, values): # Arrayf of bool
        self._cache_use_fill_caps_s = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('use_fill_caps', self._cache_use_fill_caps_s)

    @property
    def use_fill_deforms(self): # Array of bool
        if self._cache_use_fill_deforms is None:
            self._cache_use_fill_deforms = np.empty(len(self), np.bool)
            self.coll.foreach_get('use_fill_deform', self._cache_use_fill_deforms)
        return self._cache_use_fill_deforms

    @use_fill_deforms.setter
    def use_fill_deforms(self, values): # Arrayf of bool
        self._cache_use_fill_deforms = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('use_fill_deform', self._cache_use_fill_deforms)

    @property
    def use_map_tapers(self): # Array of bool
        if self._cache_use_map_tapers is None:
            self._cache_use_map_tapers = np.empty(len(self), np.bool)
            self.coll.foreach_get('use_map_taper', self._cache_use_map_tapers)
        return self._cache_use_map_tapers

    @use_map_tapers.setter
    def use_map_tapers(self, values): # Arrayf of bool
        self._cache_use_map_tapers = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('use_map_taper', self._cache_use_map_tapers)

    @property
    def use_path_follows(self): # Array of bool
        if self._cache_use_path_follows is None:
            self._cache_use_path_follows = np.empty(len(self), np.bool)
            self.coll.foreach_get('use_path_follow', self._cache_use_path_follows)
        return self._cache_use_path_follows

    @use_path_follows.setter
    def use_path_follows(self, values): # Arrayf of bool
        self._cache_use_path_follows = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('use_path_follow', self._cache_use_path_follows)

    @property
    def use_paths(self): # Array of bool
        if self._cache_use_paths is None:
            self._cache_use_paths = np.empty(len(self), np.bool)
            self.coll.foreach_get('use_path', self._cache_use_paths)
        return self._cache_use_paths

    @use_paths.setter
    def use_paths(self, values): # Arrayf of bool
        self._cache_use_paths = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('use_path', self._cache_use_paths)

    @property
    def use_radius_s(self): # Array of bool
        if self._cache_use_radius_s is None:
            self._cache_use_radius_s = np.empty(len(self), np.bool)
            self.coll.foreach_get('use_radius', self._cache_use_radius_s)
        return self._cache_use_radius_s

    @use_radius_s.setter
    def use_radius_s(self, values): # Arrayf of bool
        self._cache_use_radius_s = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('use_radius', self._cache_use_radius_s)

    @property
    def use_stretchs(self): # Array of bool
        if self._cache_use_stretchs is None:
            self._cache_use_stretchs = np.empty(len(self), np.bool)
            self.coll.foreach_get('use_stretch', self._cache_use_stretchs)
        return self._cache_use_stretchs

    @use_stretchs.setter
    def use_stretchs(self, values): # Arrayf of bool
        self._cache_use_stretchs = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        self.coll.foreach_set('use_stretch', self._cache_use_stretchs)

#================================================================================
# Object class wrapper

class WObject(WObjectRoot):

    @property
    def active_material_index(self): # int
        return self.obj.active_material_index

    @active_material_index.setter
    def active_material_index(self, value): # int
        self.obj.active_material_index = value

    @property
    def active_shape_key_index(self): # int
        return self.obj.active_shape_key_index

    @active_shape_key_index.setter
    def active_shape_key_index(self, value): # int
        self.obj.active_shape_key_index = value

    @property
    def bound_box(self): # bbox
        return self.obj.bound_box

    @bound_box.setter
    def bound_box(self, value): # bbox
        self.obj.bound_box = value

    @property
    def color(self): # V4
        return self.obj.color

    @color.setter
    def color(self, value): # V4
        self.obj.color = value

    @property
    def delta_location(self): # V3
        return self.obj.delta_location

    @delta_location.setter
    def delta_location(self, value): # V3
        self.obj.delta_location = value

    @property
    def delta_rotation_euler(self): # V3
        return self.obj.delta_rotation_euler

    @delta_rotation_euler.setter
    def delta_rotation_euler(self, value): # V3
        self.obj.delta_rotation_euler = value

    @property
    def delta_rotation_quaternion(self): # V4
        return self.obj.delta_rotation_quaternion

    @delta_rotation_quaternion.setter
    def delta_rotation_quaternion(self, value): # V4
        self.obj.delta_rotation_quaternion = value

    @property
    def delta_scale(self): # V3
        return self.obj.delta_scale

    @delta_scale.setter
    def delta_scale(self, value): # V3
        self.obj.delta_scale = value

    @property
    def empty_display_size(self): # float
        return self.obj.empty_display_size

    @empty_display_size.setter
    def empty_display_size(self, value): # float
        self.obj.empty_display_size = value

    @property
    def empty_image_offset(self): # V2
        return self.obj.empty_image_offset

    @empty_image_offset.setter
    def empty_image_offset(self, value): # V2
        self.obj.empty_image_offset = value

    @property
    def dimensions(self): # V3
        return self.obj.dimensions

    @dimensions.setter
    def dimensions(self, value): # V3
        self.obj.dimensions = value

    @property
    def hide_render(self): # bool
        return self.obj.hide_render

    @hide_render.setter
    def hide_render(self, value): # bool
        self.obj.hide_render = value

    @property
    def hide_select(self): # bool
        return self.obj.hide_select

    @hide_select.setter
    def hide_select(self, value): # bool
        self.obj.hide_select = value

    @property
    def hide_viewport(self): # bool
        return self.obj.hide_viewport

    @hide_viewport.setter
    def hide_viewport(self, value): # bool
        self.obj.hide_viewport = value

    @property
    def instance_faces_scale(self): # float
        return self.obj.instance_faces_scale

    @instance_faces_scale.setter
    def instance_faces_scale(self, value): # float
        self.obj.instance_faces_scale = value

    @property
    def location(self): # V3
        return self.obj.location

    @property
    def x(self):
        return self.obj.location[0]

    @property
    def y(self):
        return self.obj.location[1]

    @property
    def z(self):
        return self.obj.location[2]

    @location.setter
    def location(self, value): # V3
        self.obj.location = value

    @x.setter
    def x(self, value):
        self.obj.location[0] = value

    @y.setter
    def y(self, value):
        self.obj.location[1] = value

    @z.setter
    def z(self, value):
        self.obj.location[2] = value

    @property
    def lock_scale(self): # bool
        return self.obj.lock_scale

    @lock_scale.setter
    def lock_scale(self, value): # bool
        self.obj.lock_scale = value

    @property
    def matrix_basis(self): # M4
        return self.obj.matrix_basis

    @property
    def matrix_local(self): # M4
        return self.obj.matrix_local

    @property
    def matrix_parent_inverse(self): # M4
        return self.obj.matrix_parent_inverse

    @property
    def matrix_world(self): # M4
        return self.obj.matrix_world

    @property
    def pass_index(self): # int
        return self.obj.pass_index

    @pass_index.setter
    def pass_index(self, value): # int
        self.obj.pass_index = value

    @property
    def rotation_euler(self): # V3
        return self.obj.rotation_euler

    @property
    def rx(self):
        return self.obj.rotation_euler[0]

    @property
    def ry(self):
        return self.obj.rotation_euler[1]

    @property
    def rz(self):
        return self.obj.rotation_euler[2]

    @rotation_euler.setter
    def rotation_euler(self, value): # V3
        self.obj.rotation_euler = value

    @rx.setter
    def rx(self, value):
        self.obj.rotation_euler[0] = value

    @ry.setter
    def ry(self, value):
        self.obj.rotation_euler[1] = value

    @rz.setter
    def rz(self, value):
        self.obj.rotation_euler[2] = value

    @property
    def rotation_mode(self): # str
        return self.obj.rotation_mode

    @rotation_mode.setter
    def rotation_mode(self, value): # str
        self.obj.rotation_mode = value

    @property
    def rotation_quaternion(self): # V4
        return self.obj.rotation_quaternion

    @rotation_quaternion.setter
    def rotation_quaternion(self, value): # V4
        self.obj.rotation_quaternion = value

    @property
    def scale(self): # V3
        return self.obj.scale

    @property
    def scx(self):
        return self.obj.scale[0]

    @property
    def scy(self):
        return self.obj.scale[1]

    @property
    def scz(self):
        return self.obj.scale[2]

    @scale.setter
    def scale(self, value): # V3
        self.obj.scale = value

    @scx.setter
    def scx(self, value):
        self.obj.scale[0] = value

    @scy.setter
    def scy(self, value):
        self.obj.scale[1] = value

    @scz.setter
    def scz(self, value):
        self.obj.scale[2] = value

    @property
    def show_all_edges(self): # bool
        return self.obj.show_all_edges

    @show_all_edges.setter
    def show_all_edges(self, value): # bool
        self.obj.show_all_edges = value

    @property
    def show_axis(self): # bool
        return self.obj.show_axis

    @show_axis.setter
    def show_axis(self, value): # bool
        self.obj.show_axis = value

    @property
    def show_bounds(self): # bool
        return self.obj.show_bounds

    @show_bounds.setter
    def show_bounds(self, value): # bool
        self.obj.show_bounds = value

    @property
    def show_empty_image_only_axis_aligned(self): # bool
        return self.obj.show_empty_image_only_axis_aligned

    @show_empty_image_only_axis_aligned.setter
    def show_empty_image_only_axis_aligned(self, value): # bool
        self.obj.show_empty_image_only_axis_aligned = value

    @property
    def show_empty_image_orthographic(self): # bool
        return self.obj.show_empty_image_orthographic

    @show_empty_image_orthographic.setter
    def show_empty_image_orthographic(self, value): # bool
        self.obj.show_empty_image_orthographic = value

    @property
    def show_empty_image_perspective(self): # bool
        return self.obj.show_empty_image_perspective

    @show_empty_image_perspective.setter
    def show_empty_image_perspective(self, value): # bool
        self.obj.show_empty_image_perspective = value

    @property
    def show_in_front(self): # bool
        return self.obj.show_in_front

    @show_in_front.setter
    def show_in_front(self, value): # bool
        self.obj.show_in_front = value

    @property
    def show_instancer_for_render(self): # bool
        return self.obj.show_instancer_for_render

    @show_instancer_for_render.setter
    def show_instancer_for_render(self, value): # bool
        self.obj.show_instancer_for_render = value

    @property
    def show_instancer_for_viewport(self): # bool
        return self.obj.show_instancer_for_viewport

    @show_instancer_for_viewport.setter
    def show_instancer_for_viewport(self, value): # bool
        self.obj.show_instancer_for_viewport = value

    @property
    def show_name(self): # bool
        return self.obj.show_name

    @show_name.setter
    def show_name(self, value): # bool
        self.obj.show_name = value

    @property
    def show_only_shape_key(self): # bool
        return self.obj.show_only_shape_key

    @show_only_shape_key.setter
    def show_only_shape_key(self, value): # bool
        self.obj.show_only_shape_key = value

    @property
    def show_texture_space(self): # bool
        return self.obj.show_texture_space

    @show_texture_space.setter
    def show_texture_space(self, value): # bool
        self.obj.show_texture_space = value

    @property
    def show_transparent(self): # bool
        return self.obj.show_transparent

    @show_transparent.setter
    def show_transparent(self, value): # bool
        self.obj.show_transparent = value

    @property
    def show_wire(self): # bool
        return self.obj.show_wire

    @show_wire.setter
    def show_wire(self, value): # bool
        self.obj.show_wire = value

    @property
    def track_axis(self): # str
        return self.obj.track_axis

    @track_axis.setter
    def track_axis(self, value): # str
        self.obj.track_axis = value

    @property
    def type(self): # str
        return self.obj.type

    @property
    def up_axis(self): # str
        return self.obj.up_axis

    @up_axis.setter
    def up_axis(self, value): # str
        self.obj.up_axis = value

    @property
    def use_empty_image_alpha(self): # bool
        return self.obj.use_empty_image_alpha

    @use_empty_image_alpha.setter
    def use_empty_image_alpha(self, value): # bool
        self.obj.use_empty_image_alpha = value

    @property
    def use_instance_faces_scale(self): # bool
        return self.obj.use_instance_faces_scale

    @use_instance_faces_scale.setter
    def use_instance_faces_scale(self, value): # bool
        self.obj.use_instance_faces_scale = value

    @property
    def use_instance_vertices_rotation(self): # bool
        return self.obj.use_instance_vertices_rotation

    @use_instance_vertices_rotation.setter
    def use_instance_vertices_rotation(self, value): # bool
        self.obj.use_instance_vertices_rotation = value

    @property
    def use_shape_key_edit_mode(self): # bool
        return self.obj.use_shape_key_edit_mode

    @use_shape_key_edit_mode.setter
    def use_shape_key_edit_mode(self, value): # bool
        self.obj.use_shape_key_edit_mode = value

#================================================================================
# Array of WObjects

class WObjects(ArrayOf):
    def __init__(self, wowner):
        super().__init__(wowner, WObject)
        self._cache_active_material_indices = None
        self._cache_active_shape_key_indices = None
        self._cache_colors                  = None
        self._cache_delta_locations         = None
        self._cache_delta_rotation_eulers   = None
        self._cache_delta_rotation_quaternions = None
        self._cache_delta_scales            = None
        self._cache_empty_display_sizes     = None
        self._cache_empty_image_offsets     = None
        self._cache_dimensions_s            = None
        self._cache_hide_renders            = None
        self._cache_hide_selects            = None
        self._cache_hide_viewports          = None
        self._cache_instance_faces_scales   = None
        self._cache_locations               = None
        self._cache_lock_scales             = None
        self._cache_pass_indices            = None
        self._cache_rotation_eulers         = None
        self._cache_rotation_modes          = None
        self._cache_rotation_quaternions    = None
        self._cache_scales                  = None
        self._cache_show_all_edges_s        = None
        self._cache_show_axis_s             = None
        self._cache_show_bounds_s           = None
        self._cache_show_empty_image_only_axis_aligneds = None
        self._cache_show_empty_image_orthographics = None
        self._cache_show_empty_image_perspectives = None
        self._cache_show_in_fronts          = None
        self._cache_show_instancer_for_renders = None
        self._cache_show_instancer_for_viewports = None
        self._cache_show_names              = None
        self._cache_show_only_shape_keys    = None
        self._cache_show_texture_spaces     = None
        self._cache_show_transparents       = None
        self._cache_show_wires              = None
        self._cache_track_axis_s            = None
        self._cache_types                   = None
        self._cache_up_axis_s               = None
        self._cache_use_empty_image_alphas  = None
        self._cache_use_instance_faces_scales = None
        self._cache_use_instance_vertices_rotations = None
        self._cache_use_shape_key_edit_modes = None
        self._cache_quaternions             = None
        self._cache_hides                   = None

    def erase_cache(self):
        super().erase_cache()
        self._cache_active_material_indices = None
        self._cache_active_shape_key_indices = None
        self._cache_colors                  = None
        self._cache_delta_locations         = None
        self._cache_delta_rotation_eulers   = None
        self._cache_delta_rotation_quaternions = None
        self._cache_delta_scales            = None
        self._cache_empty_display_sizes     = None
        self._cache_empty_image_offsets     = None
        self._cache_dimensions_s            = None
        self._cache_hide_renders            = None
        self._cache_hide_selects            = None
        self._cache_hide_viewports          = None
        self._cache_instance_faces_scales   = None
        self._cache_locations               = None
        self._cache_lock_scales             = None
        self._cache_pass_indices            = None
        self._cache_rotation_eulers         = None
        self._cache_rotation_modes          = None
        self._cache_rotation_quaternions    = None
        self._cache_scales                  = None
        self._cache_show_all_edges_s        = None
        self._cache_show_axis_s             = None
        self._cache_show_bounds_s           = None
        self._cache_show_empty_image_only_axis_aligneds = None
        self._cache_show_empty_image_orthographics = None
        self._cache_show_empty_image_perspectives = None
        self._cache_show_in_fronts          = None
        self._cache_show_instancer_for_renders = None
        self._cache_show_instancer_for_viewports = None
        self._cache_show_names              = None
        self._cache_show_only_shape_keys    = None
        self._cache_show_texture_spaces     = None
        self._cache_show_transparents       = None
        self._cache_show_wires              = None
        self._cache_track_axis_s            = None
        self._cache_types                   = None
        self._cache_up_axis_s               = None
        self._cache_use_empty_image_alphas  = None
        self._cache_use_instance_faces_scales = None
        self._cache_use_instance_vertices_rotations = None
        self._cache_use_shape_key_edit_modes = None
        self._cache_quaternions             = None
        self._cache_hides                   = None

    @property
    def active_material_indices(self): # Array of int
        if self._cache_active_material_indices is None:
            self._cache_active_material_indices = np.empty(len(self), np.int)
            for i in range(len(self)):
                self._cache_active_material_indices[i] = self[i].active_material_index
        return self._cache_active_material_indices

    @active_material_indices.setter
    def active_material_indices(self, values): # Arrayf of int
        self._cache_active_material_indices = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].active_material_index = self._cache_active_material_indices[i]

    @property
    def active_shape_key_indices(self): # Array of int
        if self._cache_active_shape_key_indices is None:
            self._cache_active_shape_key_indices = np.empty(len(self), np.int)
            for i in range(len(self)):
                self._cache_active_shape_key_indices[i] = self[i].active_shape_key_index
        return self._cache_active_shape_key_indices

    @active_shape_key_indices.setter
    def active_shape_key_indices(self, values): # Arrayf of int
        self._cache_active_shape_key_indices = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].active_shape_key_index = self._cache_active_shape_key_indices[i]

    @property
    def colors(self): # Array of V4
        if self._cache_colors is None:
            self._cache_colors = np.empty(len(self)*4, np.float).reshape(len(self), 4)
            for i in range(len(self)):
                self._cache_colors[i] = self[i].color
        return self._cache_colors

    @colors.setter
    def colors(self, values): # Arrayf of V4
        self._cache_colors = to_array(values, (len(self), 4), f'4-vector or array of {len(self)} 4-vectors')
        for i in range(len(self)):
            self[i].color = self._cache_colors[i]

    @property
    def delta_locations(self): # Array of V3
        if self._cache_delta_locations is None:
            self._cache_delta_locations = np.empty(len(self)*3, np.float).reshape(len(self), 3)
            for i in range(len(self)):
                self._cache_delta_locations[i] = self[i].delta_location
        return self._cache_delta_locations

    @delta_locations.setter
    def delta_locations(self, values): # Arrayf of V3
        self._cache_delta_locations = to_array(values, (len(self), 3), f'3-vector or array of {len(self)} 3-vectors')
        for i in range(len(self)):
            self[i].delta_location = self._cache_delta_locations[i]

    @property
    def delta_rotation_eulers(self): # Array of V3
        if self._cache_delta_rotation_eulers is None:
            self._cache_delta_rotation_eulers = np.empty(len(self)*3, np.float).reshape(len(self), 3)
            for i in range(len(self)):
                self._cache_delta_rotation_eulers[i] = self[i].delta_rotation_euler
        return self._cache_delta_rotation_eulers

    @delta_rotation_eulers.setter
    def delta_rotation_eulers(self, values): # Arrayf of V3
        self._cache_delta_rotation_eulers = to_array(values, (len(self), 3), f'3-vector or array of {len(self)} 3-vectors')
        for i in range(len(self)):
            self[i].delta_rotation_euler = self._cache_delta_rotation_eulers[i]

    @property
    def delta_rotation_quaternions(self): # Array of V4
        if self._cache_delta_rotation_quaternions is None:
            self._cache_delta_rotation_quaternions = np.empty(len(self)*4, np.float).reshape(len(self), 4)
            for i in range(len(self)):
                self._cache_delta_rotation_quaternions[i] = self[i].delta_rotation_quaternion
        return self._cache_delta_rotation_quaternions

    @delta_rotation_quaternions.setter
    def delta_rotation_quaternions(self, values): # Arrayf of V4
        self._cache_delta_rotation_quaternions = to_array(values, (len(self), 4), f'4-vector or array of {len(self)} 4-vectors')
        for i in range(len(self)):
            self[i].delta_rotation_quaternion = self._cache_delta_rotation_quaternions[i]

    @property
    def delta_scales(self): # Array of V3
        if self._cache_delta_scales is None:
            self._cache_delta_scales = np.empty(len(self)*3, np.float).reshape(len(self), 3)
            for i in range(len(self)):
                self._cache_delta_scales[i] = self[i].delta_scale
        return self._cache_delta_scales

    @delta_scales.setter
    def delta_scales(self, values): # Arrayf of V3
        self._cache_delta_scales = to_array(values, (len(self), 3), f'3-vector or array of {len(self)} 3-vectors')
        for i in range(len(self)):
            self[i].delta_scale = self._cache_delta_scales[i]

    @property
    def empty_display_sizes(self): # Array of float
        if self._cache_empty_display_sizes is None:
            self._cache_empty_display_sizes = np.empty(len(self), np.float)
            for i in range(len(self)):
                self._cache_empty_display_sizes[i] = self[i].empty_display_size
        return self._cache_empty_display_sizes

    @empty_display_sizes.setter
    def empty_display_sizes(self, values): # Arrayf of float
        self._cache_empty_display_sizes = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].empty_display_size = self._cache_empty_display_sizes[i]

    @property
    def empty_image_offsets(self): # Array of V2
        if self._cache_empty_image_offsets is None:
            self._cache_empty_image_offsets = np.empty(len(self)*2, np.float).reshape(len(self), 2)
            for i in range(len(self)):
                self._cache_empty_image_offsets[i] = self[i].empty_image_offset
        return self._cache_empty_image_offsets

    @empty_image_offsets.setter
    def empty_image_offsets(self, values): # Arrayf of V2
        self._cache_empty_image_offsets = to_array(values, (len(self), 2), f'2-vector or array of {len(self)} 2-vectors')
        for i in range(len(self)):
            self[i].empty_image_offset = self._cache_empty_image_offsets[i]

    @property
    def dimensions_s(self): # Array of V3
        if self._cache_dimensions_s is None:
            self._cache_dimensions_s = np.empty(len(self)*3, np.float).reshape(len(self), 3)
            for i in range(len(self)):
                self._cache_dimensions_s[i] = self[i].dimensions
        return self._cache_dimensions_s

    @dimensions_s.setter
    def dimensions_s(self, values): # Arrayf of V3
        self._cache_dimensions_s = to_array(values, (len(self), 3), f'3-vector or array of {len(self)} 3-vectors')
        for i in range(len(self)):
            self[i].dimensions = self._cache_dimensions_s[i]

    @property
    def hide_renders(self): # Array of bool
        if self._cache_hide_renders is None:
            self._cache_hide_renders = np.empty(len(self), np.bool)
            for i in range(len(self)):
                self._cache_hide_renders[i] = self[i].hide_render
        return self._cache_hide_renders

    @hide_renders.setter
    def hide_renders(self, values): # Arrayf of bool
        self._cache_hide_renders = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].hide_render = self._cache_hide_renders[i]

    @property
    def hide_selects(self): # Array of bool
        if self._cache_hide_selects is None:
            self._cache_hide_selects = np.empty(len(self), np.bool)
            for i in range(len(self)):
                self._cache_hide_selects[i] = self[i].hide_select
        return self._cache_hide_selects

    @hide_selects.setter
    def hide_selects(self, values): # Arrayf of bool
        self._cache_hide_selects = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].hide_select = self._cache_hide_selects[i]

    @property
    def hide_viewports(self): # Array of bool
        if self._cache_hide_viewports is None:
            self._cache_hide_viewports = np.empty(len(self), np.bool)
            for i in range(len(self)):
                self._cache_hide_viewports[i] = self[i].hide_viewport
        return self._cache_hide_viewports

    @hide_viewports.setter
    def hide_viewports(self, values): # Arrayf of bool
        self._cache_hide_viewports = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].hide_viewport = self._cache_hide_viewports[i]

    @property
    def instance_faces_scales(self): # Array of float
        if self._cache_instance_faces_scales is None:
            self._cache_instance_faces_scales = np.empty(len(self), np.float)
            for i in range(len(self)):
                self._cache_instance_faces_scales[i] = self[i].instance_faces_scale
        return self._cache_instance_faces_scales

    @instance_faces_scales.setter
    def instance_faces_scales(self, values): # Arrayf of float
        self._cache_instance_faces_scales = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].instance_faces_scale = self._cache_instance_faces_scales[i]

    @property
    def locations(self): # Array of V3
        if self._cache_locations is None:
            self._cache_locations = np.empty(len(self)*3, np.float).reshape(len(self), 3)
            for i in range(len(self)):
                self._cache_locations[i] = self[i].location
        return self._cache_locations

    @locations.setter
    def locations(self, values): # Arrayf of V3
        self._cache_locations = to_array(values, (len(self), 3), f'3-vector or array of {len(self)} 3-vectors')
        for i in range(len(self)):
            self[i].location = self._cache_locations[i]

    # xyzw access to locations

    @property
    def xs(self): 
        return self.locations[:, 0]

    @xs.setter
    def xs(self, values):
        self.locations[:, 0] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.locations = self._cache_locations

    @property
    def ys(self): 
        return self.locations[:, 1]

    @ys.setter
    def ys(self, values):
        self.locations[:, 1] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.locations = self._cache_locations

    @property
    def zs(self): 
        return self.locations[:, 2]

    @zs.setter
    def zs(self, values):
        self.locations[:, 2] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.locations = self._cache_locations

    @property
    def lock_scales(self): # Array of bool
        if self._cache_lock_scales is None:
            self._cache_lock_scales = np.empty(len(self), np.bool)
            for i in range(len(self)):
                self._cache_lock_scales[i] = self[i].lock_scale
        return self._cache_lock_scales

    @lock_scales.setter
    def lock_scales(self, values): # Arrayf of bool
        self._cache_lock_scales = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].lock_scale = self._cache_lock_scales[i]

    @property
    def pass_indices(self): # Array of int
        if self._cache_pass_indices is None:
            self._cache_pass_indices = np.empty(len(self), np.int)
            for i in range(len(self)):
                self._cache_pass_indices[i] = self[i].pass_index
        return self._cache_pass_indices

    @pass_indices.setter
    def pass_indices(self, values): # Arrayf of int
        self._cache_pass_indices = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].pass_index = self._cache_pass_indices[i]

    @property
    def rotation_eulers(self): # Array of V3
        if self._cache_rotation_eulers is None:
            self._cache_rotation_eulers = np.empty(len(self)*3, np.float).reshape(len(self), 3)
            for i in range(len(self)):
                self._cache_rotation_eulers[i] = self[i].rotation_euler
        return self._cache_rotation_eulers

    @rotation_eulers.setter
    def rotation_eulers(self, values): # Arrayf of V3
        self._cache_rotation_eulers = to_array(values, (len(self), 3), f'3-vector or array of {len(self)} 3-vectors')
        for i in range(len(self)):
            self[i].rotation_euler = self._cache_rotation_eulers[i]

    # xyzw access to rotation_eulers

    @property
    def rxs(self): 
        return self.rotation_eulers[:, 0]

    @rxs.setter
    def rxs(self, values):
        self.rotation_eulers[:, 0] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.rotation_eulers = self._cache_rotation_eulers

    @property
    def rys(self): 
        return self.rotation_eulers[:, 1]

    @rys.setter
    def rys(self, values):
        self.rotation_eulers[:, 1] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.rotation_eulers = self._cache_rotation_eulers

    @property
    def rzs(self): 
        return self.rotation_eulers[:, 2]

    @rzs.setter
    def rzs(self, values):
        self.rotation_eulers[:, 2] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.rotation_eulers = self._cache_rotation_eulers

    @property
    def rotation_modes(self): # Array of str
        if self._cache_rotation_modes is None:
            self._cache_rotation_modes = np.empty(len(self), np.object)
            for i in range(len(self)):
                self._cache_rotation_modes[i] = self[i].rotation_mode
        return self._cache_rotation_modes

    @rotation_modes.setter
    def rotation_modes(self, values): # Arrayf of str
        self._cache_rotation_modes = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].rotation_mode = self._cache_rotation_modes[i]

    @property
    def rotation_quaternions(self): # Array of V4
        if self._cache_rotation_quaternions is None:
            self._cache_rotation_quaternions = np.empty(len(self)*4, np.float).reshape(len(self), 4)
            for i in range(len(self)):
                self._cache_rotation_quaternions[i] = self[i].rotation_quaternion
        return self._cache_rotation_quaternions

    @rotation_quaternions.setter
    def rotation_quaternions(self, values): # Arrayf of V4
        self._cache_rotation_quaternions = to_array(values, (len(self), 4), f'4-vector or array of {len(self)} 4-vectors')
        for i in range(len(self)):
            self[i].rotation_quaternion = self._cache_rotation_quaternions[i]

    @property
    def scales(self): # Array of V3
        if self._cache_scales is None:
            self._cache_scales = np.empty(len(self)*3, np.float).reshape(len(self), 3)
            for i in range(len(self)):
                self._cache_scales[i] = self[i].scale
        return self._cache_scales

    @scales.setter
    def scales(self, values): # Arrayf of V3
        self._cache_scales = to_array(values, (len(self), 3), f'3-vector or array of {len(self)} 3-vectors')
        for i in range(len(self)):
            self[i].scale = self._cache_scales[i]

    # xyzw access to scales

    @property
    def scxs(self): 
        return self.scales[:, 0]

    @scxs.setter
    def scxs(self, values):
        self.scales[:, 0] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.scales = self._cache_scales

    @property
    def scys(self): 
        return self.scales[:, 1]

    @scys.setter
    def scys(self, values):
        self.scales[:, 1] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.scales = self._cache_scales

    @property
    def sczs(self): 
        return self.scales[:, 2]

    @sczs.setter
    def sczs(self, values):
        self.scales[:, 2] = to_array(values, (len(self), 1), f'value or array of {len(self)} values')
        self.scales = self._cache_scales

    @property
    def show_all_edges_s(self): # Array of bool
        if self._cache_show_all_edges_s is None:
            self._cache_show_all_edges_s = np.empty(len(self), np.bool)
            for i in range(len(self)):
                self._cache_show_all_edges_s[i] = self[i].show_all_edges
        return self._cache_show_all_edges_s

    @show_all_edges_s.setter
    def show_all_edges_s(self, values): # Arrayf of bool
        self._cache_show_all_edges_s = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].show_all_edges = self._cache_show_all_edges_s[i]

    @property
    def show_axis_s(self): # Array of bool
        if self._cache_show_axis_s is None:
            self._cache_show_axis_s = np.empty(len(self), np.bool)
            for i in range(len(self)):
                self._cache_show_axis_s[i] = self[i].show_axis
        return self._cache_show_axis_s

    @show_axis_s.setter
    def show_axis_s(self, values): # Arrayf of bool
        self._cache_show_axis_s = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].show_axis = self._cache_show_axis_s[i]

    @property
    def show_bounds_s(self): # Array of bool
        if self._cache_show_bounds_s is None:
            self._cache_show_bounds_s = np.empty(len(self), np.bool)
            for i in range(len(self)):
                self._cache_show_bounds_s[i] = self[i].show_bounds
        return self._cache_show_bounds_s

    @show_bounds_s.setter
    def show_bounds_s(self, values): # Arrayf of bool
        self._cache_show_bounds_s = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].show_bounds = self._cache_show_bounds_s[i]

    @property
    def show_empty_image_only_axis_aligneds(self): # Array of bool
        if self._cache_show_empty_image_only_axis_aligneds is None:
            self._cache_show_empty_image_only_axis_aligneds = np.empty(len(self), np.bool)
            for i in range(len(self)):
                self._cache_show_empty_image_only_axis_aligneds[i] = self[i].show_empty_image_only_axis_aligned
        return self._cache_show_empty_image_only_axis_aligneds

    @show_empty_image_only_axis_aligneds.setter
    def show_empty_image_only_axis_aligneds(self, values): # Arrayf of bool
        self._cache_show_empty_image_only_axis_aligneds = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].show_empty_image_only_axis_aligned = self._cache_show_empty_image_only_axis_aligneds[i]

    @property
    def show_empty_image_orthographics(self): # Array of bool
        if self._cache_show_empty_image_orthographics is None:
            self._cache_show_empty_image_orthographics = np.empty(len(self), np.bool)
            for i in range(len(self)):
                self._cache_show_empty_image_orthographics[i] = self[i].show_empty_image_orthographic
        return self._cache_show_empty_image_orthographics

    @show_empty_image_orthographics.setter
    def show_empty_image_orthographics(self, values): # Arrayf of bool
        self._cache_show_empty_image_orthographics = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].show_empty_image_orthographic = self._cache_show_empty_image_orthographics[i]

    @property
    def show_empty_image_perspectives(self): # Array of bool
        if self._cache_show_empty_image_perspectives is None:
            self._cache_show_empty_image_perspectives = np.empty(len(self), np.bool)
            for i in range(len(self)):
                self._cache_show_empty_image_perspectives[i] = self[i].show_empty_image_perspective
        return self._cache_show_empty_image_perspectives

    @show_empty_image_perspectives.setter
    def show_empty_image_perspectives(self, values): # Arrayf of bool
        self._cache_show_empty_image_perspectives = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].show_empty_image_perspective = self._cache_show_empty_image_perspectives[i]

    @property
    def show_in_fronts(self): # Array of bool
        if self._cache_show_in_fronts is None:
            self._cache_show_in_fronts = np.empty(len(self), np.bool)
            for i in range(len(self)):
                self._cache_show_in_fronts[i] = self[i].show_in_front
        return self._cache_show_in_fronts

    @show_in_fronts.setter
    def show_in_fronts(self, values): # Arrayf of bool
        self._cache_show_in_fronts = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].show_in_front = self._cache_show_in_fronts[i]

    @property
    def show_instancer_for_renders(self): # Array of bool
        if self._cache_show_instancer_for_renders is None:
            self._cache_show_instancer_for_renders = np.empty(len(self), np.bool)
            for i in range(len(self)):
                self._cache_show_instancer_for_renders[i] = self[i].show_instancer_for_render
        return self._cache_show_instancer_for_renders

    @show_instancer_for_renders.setter
    def show_instancer_for_renders(self, values): # Arrayf of bool
        self._cache_show_instancer_for_renders = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].show_instancer_for_render = self._cache_show_instancer_for_renders[i]

    @property
    def show_instancer_for_viewports(self): # Array of bool
        if self._cache_show_instancer_for_viewports is None:
            self._cache_show_instancer_for_viewports = np.empty(len(self), np.bool)
            for i in range(len(self)):
                self._cache_show_instancer_for_viewports[i] = self[i].show_instancer_for_viewport
        return self._cache_show_instancer_for_viewports

    @show_instancer_for_viewports.setter
    def show_instancer_for_viewports(self, values): # Arrayf of bool
        self._cache_show_instancer_for_viewports = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].show_instancer_for_viewport = self._cache_show_instancer_for_viewports[i]

    @property
    def show_names(self): # Array of bool
        if self._cache_show_names is None:
            self._cache_show_names = np.empty(len(self), np.bool)
            for i in range(len(self)):
                self._cache_show_names[i] = self[i].show_name
        return self._cache_show_names

    @show_names.setter
    def show_names(self, values): # Arrayf of bool
        self._cache_show_names = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].show_name = self._cache_show_names[i]

    @property
    def show_only_shape_keys(self): # Array of bool
        if self._cache_show_only_shape_keys is None:
            self._cache_show_only_shape_keys = np.empty(len(self), np.bool)
            for i in range(len(self)):
                self._cache_show_only_shape_keys[i] = self[i].show_only_shape_key
        return self._cache_show_only_shape_keys

    @show_only_shape_keys.setter
    def show_only_shape_keys(self, values): # Arrayf of bool
        self._cache_show_only_shape_keys = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].show_only_shape_key = self._cache_show_only_shape_keys[i]

    @property
    def show_texture_spaces(self): # Array of bool
        if self._cache_show_texture_spaces is None:
            self._cache_show_texture_spaces = np.empty(len(self), np.bool)
            for i in range(len(self)):
                self._cache_show_texture_spaces[i] = self[i].show_texture_space
        return self._cache_show_texture_spaces

    @show_texture_spaces.setter
    def show_texture_spaces(self, values): # Arrayf of bool
        self._cache_show_texture_spaces = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].show_texture_space = self._cache_show_texture_spaces[i]

    @property
    def show_transparents(self): # Array of bool
        if self._cache_show_transparents is None:
            self._cache_show_transparents = np.empty(len(self), np.bool)
            for i in range(len(self)):
                self._cache_show_transparents[i] = self[i].show_transparent
        return self._cache_show_transparents

    @show_transparents.setter
    def show_transparents(self, values): # Arrayf of bool
        self._cache_show_transparents = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].show_transparent = self._cache_show_transparents[i]

    @property
    def show_wires(self): # Array of bool
        if self._cache_show_wires is None:
            self._cache_show_wires = np.empty(len(self), np.bool)
            for i in range(len(self)):
                self._cache_show_wires[i] = self[i].show_wire
        return self._cache_show_wires

    @show_wires.setter
    def show_wires(self, values): # Arrayf of bool
        self._cache_show_wires = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].show_wire = self._cache_show_wires[i]

    @property
    def track_axis_s(self): # Array of str
        if self._cache_track_axis_s is None:
            self._cache_track_axis_s = np.empty(len(self), np.object)
            for i in range(len(self)):
                self._cache_track_axis_s[i] = self[i].track_axis
        return self._cache_track_axis_s

    @track_axis_s.setter
    def track_axis_s(self, values): # Arrayf of str
        self._cache_track_axis_s = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].track_axis = self._cache_track_axis_s[i]

    @property
    def types(self): # Array of str
        if self._cache_types is None:
            self._cache_types = np.empty(len(self), np.object)
            for i in range(len(self)):
                self._cache_types[i] = self[i].type
        return self._cache_types

    @property
    def up_axis_s(self): # Array of str
        if self._cache_up_axis_s is None:
            self._cache_up_axis_s = np.empty(len(self), np.object)
            for i in range(len(self)):
                self._cache_up_axis_s[i] = self[i].up_axis
        return self._cache_up_axis_s

    @up_axis_s.setter
    def up_axis_s(self, values): # Arrayf of str
        self._cache_up_axis_s = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].up_axis = self._cache_up_axis_s[i]

    @property
    def use_empty_image_alphas(self): # Array of bool
        if self._cache_use_empty_image_alphas is None:
            self._cache_use_empty_image_alphas = np.empty(len(self), np.bool)
            for i in range(len(self)):
                self._cache_use_empty_image_alphas[i] = self[i].use_empty_image_alpha
        return self._cache_use_empty_image_alphas

    @use_empty_image_alphas.setter
    def use_empty_image_alphas(self, values): # Arrayf of bool
        self._cache_use_empty_image_alphas = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].use_empty_image_alpha = self._cache_use_empty_image_alphas[i]

    @property
    def use_instance_faces_scales(self): # Array of bool
        if self._cache_use_instance_faces_scales is None:
            self._cache_use_instance_faces_scales = np.empty(len(self), np.bool)
            for i in range(len(self)):
                self._cache_use_instance_faces_scales[i] = self[i].use_instance_faces_scale
        return self._cache_use_instance_faces_scales

    @use_instance_faces_scales.setter
    def use_instance_faces_scales(self, values): # Arrayf of bool
        self._cache_use_instance_faces_scales = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].use_instance_faces_scale = self._cache_use_instance_faces_scales[i]

    @property
    def use_instance_vertices_rotations(self): # Array of bool
        if self._cache_use_instance_vertices_rotations is None:
            self._cache_use_instance_vertices_rotations = np.empty(len(self), np.bool)
            for i in range(len(self)):
                self._cache_use_instance_vertices_rotations[i] = self[i].use_instance_vertices_rotation
        return self._cache_use_instance_vertices_rotations

    @use_instance_vertices_rotations.setter
    def use_instance_vertices_rotations(self, values): # Arrayf of bool
        self._cache_use_instance_vertices_rotations = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].use_instance_vertices_rotation = self._cache_use_instance_vertices_rotations[i]

    @property
    def use_shape_key_edit_modes(self): # Array of bool
        if self._cache_use_shape_key_edit_modes is None:
            self._cache_use_shape_key_edit_modes = np.empty(len(self), np.bool)
            for i in range(len(self)):
                self._cache_use_shape_key_edit_modes[i] = self[i].use_shape_key_edit_mode
        return self._cache_use_shape_key_edit_modes

    @use_shape_key_edit_modes.setter
    def use_shape_key_edit_modes(self, values): # Arrayf of bool
        self._cache_use_shape_key_edit_modes = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].use_shape_key_edit_mode = self._cache_use_shape_key_edit_modes[i]

    @property
    def quaternions(self): # Array of V4
        if self._cache_quaternions is None:
            self._cache_quaternions = np.empty(len(self)*4, np.float).reshape(len(self), 4)
            for i in range(len(self)):
                self._cache_quaternions[i] = self[i].quaternion
        return self._cache_quaternions

    @quaternions.setter
    def quaternions(self, values): # Arrayf of V4
        self._cache_quaternions = to_array(values, (len(self), 4), f'4-vector or array of {len(self)} 4-vectors')
        for i in range(len(self)):
            self[i].quaternion = self._cache_quaternions[i]

    @property
    def hides(self): # Array of bool
        if self._cache_hides is None:
            self._cache_hides = np.empty(len(self), np.bool)
            for i in range(len(self)):
                self._cache_hides[i] = self[i].hide
        return self._cache_hides

    @hides.setter
    def hides(self, values): # Arrayf of bool
        self._cache_hides = to_array(values, (len(self), 1), f'1-vector or array of {len(self)} 1-vectors')
        for i in range(len(self)):
            self[i].hide = self._cache_hides[i]

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

