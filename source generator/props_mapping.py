#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 16:48:27 2020

@author: alain
"""

# *****************************************************************************************************************************
# Object properties to wrap

object_props = {
    'active_material_index'              : 'int',
    'active_shape_key_index'             : 'int',
    'bound_box'                          : 'bbox',
    'color'                              : 'V4',
    'delta_location'                     : 'V3',
    'delta_rotation_euler'               : 'V3',
    'delta_rotation_quaternion'          : 'V4',
    'delta_scale'                        : 'V3',
    'empty_display_size'                 : 'float',
    'empty_image_offset'                 : 'V2',
    'dimensions'                         : 'V3',
    'hide_render'                        : 'bool',
    'hide_select'                        : 'bool',
    'hide_viewport'                      : 'bool',
    'instance_faces_scale'               : 'float',
    'location'                           : 'V3',
    'lock_scale'                         : 'bool',
    'matrix_basis'                       : 'M4',
    'matrix_local'                       : 'M4',
    'matrix_parent_inverse'              : 'M4',
    'matrix_world'                       : 'M4',
    'pass_index'                         : 'int',
    'rotation_euler'                     : 'V3',
    'rotation_mode'                      : 'str',
    'rotation_quaternion'                : 'V4',
    'scale'                              : 'V3',
    'show_all_edges'                     : 'bool',
    'show_axis'                          : 'bool',
    'show_bounds'                        : 'bool',
    'show_empty_image_only_axis_aligned' : 'bool',
    'show_empty_image_orthographic'      : 'bool',
    'show_empty_image_perspective'       : 'bool',
    'show_in_front'                      : 'bool',
    'show_instancer_for_render'          : 'bool',
    'show_instancer_for_viewport'        : 'bool',
    'show_name'                          : 'bool',
    'show_only_shape_key'                : 'bool',
    'show_texture_space'                 : 'bool',
    'show_transparent'                   : 'bool',
    'show_wire'                          : 'bool',
    'track_axis'                         : 'str',
    'type'                               : 'str',
    'up_axis'                            : 'str',
    'use_empty_image_alpha'              : 'bool',
    'use_instance_faces_scale'           : 'bool',
    'use_instance_vertices_rotation'     : 'bool',
    'use_shape_key_edit_mode'            : 'bool'
    }

# *****************************************************************************************************************************
# Object methods to vectorize

WObject_meths = {
    'track_to'  : ({'location': 'V3'}, None),
    'orient'    : ({'axis':     'V3'}, None),
    'distance'  : ({'location': 'V3'}, 'float'),
}

# *****************************************************************************************************************************
# Mesh properties to wrap

mesh_props = {
    'auto_smooth_angle' : 'float',
    'auto_texspace'     : 'bool',
    'edges'             : 'AWEdge',
    'loops'             : 'AWLoop',
    'polygons'          : 'AWPolygon',
    'use_auto_smooth'   : 'bool',
    'use_auto_texspace' : 'bool',
    'vertices'          : 'AWMeshVertex'
    }

edge_props = {
    'bevel_weight'      : 'float',
    'crease'            : 'float',
    'hide'              : 'bool',
    'index'             : 'int',
    'is_loose'          : 'bool',
    'select'            : 'bool',
    'use_edge_sharp'    : 'bool',
    'use_seam'          : 'bool',
    'vertices'          : 'array'
    }

loop_props = {
    'bitangent_sign'    : 'float',
    'bitangent'         : 'V3',
    'edge_index'        : 'int',
    'index'             : 'int',
    'normal'            : 'V3',
    'tangent'           : 'V3',
    'vertex_index'      : 'int',
    }

polygon_props = {
    'area'              : 'float',
    'center'            : 'V3',
    'hide'              : 'bool',
    'index'             : 'int',
    'loop_start'        : 'int',
    'loop_total'        : 'int',
    'material_index'    : 'int',
    'normal'            : 'V3',
    'use_smooth'        : 'bool',
    'vertices'          : 'array'
    }

mesh_vertex_props = {
    'bevel_weight'      : 'float',
    'co'                : 'V3',
    'hide'              : 'bool',
    'index'             : 'int',
    'normal'            : 'V3',
    'undeformed_co'     : 'V3',
    }

# *****************************************************************************************************************************
# Curve properties to wrap

curve_props = {
    'bevel_depth'       : 'float',
    'bevel_factor_end'  : 'float',
    'bevel_factor_start': 'float',
    'bevel_object'      : 'WCurve',
    'bevel_resolution'  : 'int',
    'dimensions'        : 'str',
    'eval_time'         : 'float',
    'extrude'           : 'float',
    'fill_mode'         : 'str',
    'offset'            : 'float',
    'path_duration'     : 'int',
    'render_resolution_u' : 'int',
    'render_resolution_v' : 'int',
    'resolution_u'      : 'int',
    'resolution_v'      : 'int',
    'splines'           : 'AWSpline',
    'taper_object'      : 'WCurve',
    'twist_smooth'      : 'float',
    'use_auto_texspace' : 'bool',
    'use_deform_bounds' : 'bool',
    'use_fill_caps'     : 'bool',
    'use_fill_deform'   : 'bool',
    'use_map_taper'     : 'bool',
    'use_path_follow'   : 'bool',
    'use_path'          : 'bool',
    'use_radius'        : 'bool',
    'use_stretch'       : 'bool',
    }


spline_props = {
    'bezier_points'     : 'AWBezierSplinePoint',
    'character_index'   : 'int',
    'material_index'    : 'int',
    'order_u'           : 'int',
    'order_v'           : 'int',
    'point_count_u'     : 'int',
    'point_count_v'     : 'int',
    'points'            : 'AWSplinePoint',
    'radius_interpolation' : 'str',
    'resolution_u'      : 'int',
    'resolution_v'      : 'int',
    'type'              : 'str',
    'use_bezier_u'      : 'bool',
    'use_bezier_v'      : 'bool',
    'use_cyclic_u'      : 'bool',
    'use_cyclic_v'      : 'bool',
    'use_endpoint_u'    : 'bool',
    'use_endpoint_v'    : 'bool',
    'use_smooth'        : 'bool',
    }

bezier_spline_point_props = {
    'co'                : 'V3',
    'handle_left_type'  : 'str',
    'handle_left'       : 'V3',
    'handle_right_type' : 'str',
    'handle_right'      : 'V3',
    'hide'              : 'bool',
    'radius'            : 'float',
    'tilt'              : 'int',
    'weight_softbody'   : 'float',
    }

spline_point_props = {
    'co'                : 'V3',
    'radius'            : 'float',
    'tilt'              : 'float',
    'weight_softbody'   : 'float',
    'weight'            : 'float',
    }
