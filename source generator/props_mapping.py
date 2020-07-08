#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 16:48:27 2020

@author: alain
"""

cython = False
blanks = 4
    
tab =  " " * blanks
tab2 =  " " * (blanks*2)
tab_3 =  " " * (blanks*3)
tab__4 =  " " * (blanks*4)

com = tab + "# "

# *****************************************************************************************************************************

def get_plural(name):
    if len(name) == 1:
        return name + 's'
    
    if name[-2:] == 'ex':
        return name[:-2] + 'ices'
    elif name[-1] == 's':
        return name + '_s'
    else:
        return name + 's'
    
def np_type(vtype):
    if vtype == 'int':
        return 'np.int'
    elif vtype == 'bool':
        return 'np.bool'
    elif vtype in ['float', 'V2', 'V3', 'V4']:
        return 'np.float'
    else:
        return 'np.object'
    
def type_size(vtype):
    if vtype == 'V2':
        return 2
    elif vtype == 'V3':
        return 3
    elif vtype == 'V4':
        return 4
    else:
        return 1
    
    
    


# *****************************************************************************************************************************
# Property wrapper
# Generate source code for the wrapper object and the vectorization in the ArrayOf or Coll wrapper

class PropWrapper():
    vectorizable = ['int', 'bool', 'float', 'str', 'V2', 'V3', 'V4']
    
    def __init__(self, name, vtype, prop=None, readonly=False, shortcut=None, foreach=False, array_only=False):
        self.name     = name
        self.vtype    = vtype
        self.prop     = name if prop is None else prop
        self.readonly = readonly
        self.shortcut = shortcut
        self.foreach  = foreach and self.vtype in ['int', 'bool', 'float', 'V2', 'V3', 'V4'] and (not array_only)
        self.array_only = array_only
        
    def is_vectorizable(self):
        return self.vtype in self.vectorizable
    
    @property
    def type_size(self):
        return type_size(self.vtype)
    
    @property
    def np_type(self):
        return np_type(self.vtype)
        
    @property
    def vector_mult(self):
        if self.type_size == 1:
            return ''
        else:
            return f"*{self.type_size}"
        
    @property
    def reshape(self):
        if self.type_size == 1:
            return ''
        else:
            return f".reshape(len(self), {self.type_size})"

    @property
    def reshape_1dim(self):
        if self.type_size == 1:
            return ''
        else:
            return f".reshape(len(self) * {self.type_size})"
    
    @property
    def get_plural_name(self):
        return get_plural(self.name)
        
    @property
    def get_cache_name(self):
        return '_cache_' + self.get_plural_name
        
    @staticmethod
    def get_aof_name(name):
        return 'w' + name
    
    # ---------------------------------------------------------------------------
    # Wrap the property for the wrapping object
    
    def wrapping_code(self):
        
        if self.array_only:
            return

        yield tab + "@property"
        yield tab + f"def {self.name}(self): # {self.vtype}"
        yield tab2 + f"return self.obj.{self.prop}"
        yield ""
        if (self.shortcut is not None) and (self.type_size > 1):
            for i in range(self.type_size):
                name = self.shortcut + 'xyzw'[i]
                yield tab + "@property"
                yield tab + f"def {name}(self):"
                yield tab2 + f"return self.obj.{self.prop}[{i}]"
                yield ""
                
            
        
        if not self.readonly:
            yield tab + f"@{self.name}.setter"
            yield tab + f"def {self.name}(self, value): # {self.vtype}"
            yield tab2 + f"self.obj.{self.prop} = value"
            yield ""
            if (self.shortcut is not None) and (self.type_size > 1):
                for i in range(self.type_size):
                    name = self.shortcut + 'xyzw'[i]
                    yield tab + f"@{name}.setter"
                    yield tab + f"def {name}(self, value):"
                    yield tab2 + f"self.obj.{self.prop}[{i}] = value"
                    yield ""
        
        return
    
    # ---------------------------------------------------------------------------
    # Vectorized access
    
    def arrayof_init_code(self):
        if self.is_vectorizable():
            yield tab2 + f"self.{self.get_cache_name:30} = None"
            
    def arrayof_erase_cache_code(self):
        if self.is_vectorizable():
            yield tab2 + f"self.{self.get_cache_name:30} = None"
    
    def arrayof_code(self):
        if not self.is_vectorizable():
            return
        
        yield tab + "@property"
        yield tab + f"def {self.get_plural_name}(self): # Array of {self.vtype}"
        yield tab2 + f"if self.{self.get_cache_name} is None:"
        
        if self.foreach:
            
            yield tab_3 + f"self.{self.get_cache_name} = np.empty(len(self){self.vector_mult}, {self.np_type})"
            yield tab_3 + f"self.coll.foreach_get('{self.prop}', self.{self.get_cache_name})"
            if self.type_size > 1:
                yield tab_3 + f"self.{self.get_cache_name} = self.{self.get_cache_name}{self.reshape}"
                
        else:
            yield tab_3 + f"self.{self.get_cache_name} = np.empty(len(self){self.vector_mult}, {self.np_type}){self.reshape}"
            yield tab_3 + "for i in range(len(self)):"
            yield tab__4 + f"self.{self.get_cache_name}[i] = self[i].{self.name}"
            
        yield tab2 + f"return self.{self.get_cache_name}"
        yield ""
        
        if self.readonly:
            return
        
        yield tab + f"@{self.get_plural_name}.setter"
        yield tab + f"def {self.get_plural_name}(self, values): # Arrayf of {self.vtype}"
        error = f"f'{self.type_size}-vector or array of " + "{len(self)}" + f" {self.type_size}-vectors'"
        yield tab2 + f"self.{self.get_cache_name} = to_array(values, (len(self), {self.type_size}), {error})"
        
        if self.foreach:
            yield tab2 + f"self.coll.foreach_set('{self.prop}', self.{self.get_cache_name}{self.reshape_1dim})"
        else:
            yield tab2 + "for i in range(len(self)):"
            yield tab_3 + f"self[i].{self.name} = self.{self.get_cache_name}[i]"
            
        yield ""
            
        # ---------------------------------------------------------------------------
        # Shortcuts to partial access to vector values
    
        if (not self.is_vectorizable()) or (self.shortcut is None) or (self.type_size < 2):
            return
        
        yield com + f"xyzw access to {self.get_plural_name}"
        yield ""
        
        def code(index):
            name = self.shortcut + "xyzw"[index] + 's'
            yield tab + "@property"
            yield tab + f"def {name}(self): "
            yield tab2 + f"return self.{self.get_plural_name}[:, {index}]"
            yield ""
            if not self.readonly:
                yield tab + f"@{name}.setter"
                yield tab + f"def {name}(self, values):"
                error = "f'value or array of " + "{len(self)}" + " values'"
                yield tab2 + f"self.{self.get_plural_name}[:, {index}] = to_array(values, (len(self), 1), {error})"
                yield tab2 + f"self.{self.get_plural_name} = self.{self.get_cache_name}"
                yield ""
                
        for index in range(self.type_size):
            for line in code(index):
                yield line
                
# *****************************************************************************************************************************
# Methods wrapper
                
class MethodWrapper():
    def __init__(self, name, fargs, vtype, meth=None):
        self.name  = name
        self.fargs = fargs
        self.vtype = vtype
        self.meth  = name if meth is None else meth
        
    def arrayof_code(self):
        
        # Build the args use
        def_args = ""    # method definition: def method(self, ....):
        
        # loop zip =  for _i_arg0, _i_arg1 ... in zip(_arg0, _arg1 ...)
        sitem  = "_i_obj, "            # _i_arg0, _i_arg1 ...
        szip   = "self, "              # zip(_arg0, _arg1 ...) 
        scall  = f"_i_obj.{self.meth}("  # obj.method(_i_arg0, _i_arg1 ...)
        sep    = ""
        
        for name, vt in self.fargs.items():
            def_args += f", {name}"
            
            sitem    += f"_i_{name}, "
            szip     += f"_{name}, "
            
            scall    += sep + f"_i_{name}"
            sep = ", "
        scall += ")"
            
        # Method definition
        yield tab + f"def {self.name}(self{def_args}):"
        
        # Arrayization
        for name, vt in self.fargs.items():
            error = f"{np_type(vt)} or array of " + "{len(self)}" + f" {np_type(vt)}"
            yield tab2 + f"_{name:20} = to_array({name}, (len(self), {type_size(vt)}), f'{error}')"
            
        # Do we expect a result
        if self.vtype is not None:
            size = type_size(self.vtype)
            shape = "len(self)" if size == 1 else f"(len(self), {size})"
            yield tab2 + f"{'_res':21} = np.empty({shape}, {np_type(self.vtype)})"
            
            scall = f"_res[i] = " + scall
            sitem = "i, " + sitem
            szip  = "range(len(self)), " + szip
            
        # Source code
        yield tab2 + f"for ({sitem}) in zip({szip}):"
        yield tab_3 + f"{scall}"
        
        # Result
        if self.vtype is not None:
            yield tab2 + "return _res"
            
        yield ""
    

# *****************************************************************************************************************************
# Object wrapper generator
                
class WrapperGenerator():
    def __init__(self, class_name, gen_coll = True, wrapper_root_class="Wrapper"):
        self.class_name = class_name
        self.gen_coll   = gen_coll
        self.wrapper_root_class = wrapper_root_class
        
    # Iterator on PropWrapper to be overriden in sub classes
    def props(self):
        return
    
    # Iterator on MethodWrapper to be overriden in sub classes
    def methods(self):
        return
    
    # Source code lines to wrap collections
    # eg: self.wvertices = WMeshVertices(self.obj.vertices, self)
    
    def collprops_code(self):
        return 
    
    # Source code for the Wrapper Class
    
    def wrapper_code(self):
        yield "#" + "="*80
        yield f"# {self.class_name} class wrapper"
        yield ""
        yield f"class W{self.class_name}({self.wrapper_root_class}):"
        yield ""
        
        # ---------------------------------------------------------------------------
        # Collections
        
        go = True
        try:
            for line in self.collprops_code():
                if go:
                    yield tab + "def __init__(self, obj, wowner):"
                    yield tab2 + "super().__init__(self, obj, wowner)"
                    go = False
                yield line
            yield ""
        except:
            pass
            
        # ---------------------------------------------------------------------------
        # Properties
        
        for prop in self.props():
            for line in prop.wrapping_code():
                yield line
                
        return
    
    # Source code for the Array of class
    
    def arrayof_code(self):
        cname = get_plural(self.class_name)
        super_name = "CollWrapper" if self.gen_coll else "ArrayOf"
        yield "#" + "="*80
        yield f"# Array of W{cname}"
        yield ""
        
        yield f"class W{cname}({super_name}):"
        
        # === Init
        if self.gen_coll:
            yield tab + "def __init__(self, coll, wowner):"
            yield tab2 + f"super().__init__(coll, wowner, W{self.class_name})"
        else:
            yield tab + "def __init__(self, wowner):"
            yield tab2 + f"super().__init__(wowner, W{self.class_name})"
            
        for prop in self.props():
            for line in prop.arrayof_init_code():
                yield line
        yield ""
        
        # === Erase cache
        go = True
        for prop in self.props():
            for line in prop.arrayof_init_code():
                if go:
                    yield tab + "def erase_cache(self):"
                    yield tab2 + "super().erase_cache()"
                    go = False
                yield line
        yield ""
        
        # === Vectorization wrapping
        for prop in self.props():
            for line in prop.arrayof_code():
                yield line
                
        # === Methods
        try:
            for meth in self.methods():
                for line in meth.arrayof_code():
                    yield line
        except:
            pass
                
        
# *****************************************************************************************************************************
# Meshes
        
# =============================================================================================================================
# MeshVertex
    
class MeshVertexGenerator(WrapperGenerator):
    def __init__(self):
        super().__init__("MeshVertex")
        
    def props(self):
        yield PropWrapper(name='bevel_weight',  vtype='float', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='co',            vtype='V3',    prop=None, readonly=False, shortcut='',   foreach=True)
        yield PropWrapper(name='hide',          vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='index',         vtype='int',   prop=None, readonly=True,  shortcut=None, foreach=True)
        yield PropWrapper(name='normal',        vtype='V3',    prop=None, readonly=True,  shortcut='',   foreach=True)
        yield PropWrapper(name='undeformed_co', vtype='V3',    prop=None, readonly=True,  shortcut=None, foreach=True)

# =============================================================================================================================
# Edge
    
class EdgeGenerator(WrapperGenerator):
    def __init__(self):
        super().__init__("Edge")
        
    def props(self):
        yield PropWrapper(name='bevel_weight',   vtype='float', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='crease',         vtype='float', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='hide',           vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='index',          vtype='int',   prop=None, readonly=True,  shortcut=None, foreach=True)
        yield PropWrapper(name='is_loose',       vtype='bool',  prop=None, readonly=True,  shortcut=None, foreach=True)
        yield PropWrapper(name='select',         vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_edge_sharp', vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_seam',       vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='vertices',       vtype='array', prop=None, readonly=True,  shortcut=None, foreach=False)
        
# =============================================================================================================================
# Loop
    
class LoopGenerator(WrapperGenerator):
    def __init__(self):
        super().__init__("Loop")
        
    def props(self):
        yield PropWrapper(name='bitangent_sign', vtype='float', prop=None, readonly=True,  shortcut=None, foreach=True)
        yield PropWrapper(name='bitangent',      vtype='V3',    prop=None, readonly=True,  shortcut=None, foreach=True)
        yield PropWrapper(name='edge_index',     vtype='int',   prop=None, readonly=True,  shortcut=None, foreach=True)
        yield PropWrapper(name='index',          vtype='int',   prop=None, readonly=True,  shortcut=None, foreach=True)
        yield PropWrapper(name='normal',         vtype='V3',    prop=None, readonly=True,  shortcut=None, foreach=True)
        yield PropWrapper(name='tangent',        vtype='V3',    prop=None, readonly=True,  shortcut=None, foreach=True)
        yield PropWrapper(name='vertex_index',   vtype='int',   prop=None, readonly=True,  shortcut=None, foreach=True)        

# =============================================================================================================================
# Polygon
    
class PolygonGenerator(WrapperGenerator):
    def __init__(self):
        super().__init__("Polygon")
        
    def props(self):
        yield PropWrapper(name='area',           vtype='float', prop=None, readonly=True,  shortcut=None, foreach=True)
        yield PropWrapper(name='center',         vtype='V3',    prop=None, readonly=True,  shortcut=None, foreach=True)
        yield PropWrapper(name='hide',           vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='index',          vtype='int',   prop=None, readonly=True,  shortcut=None, foreach=True)
        yield PropWrapper(name='loop_start',     vtype='int',   prop=None, readonly=True,  shortcut=None, foreach=True)
        yield PropWrapper(name='loop_total',     vtype='int',   prop=None, readonly=True,  shortcut=None, foreach=True)
        yield PropWrapper(name='material_index', vtype='int',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='normal',         vtype='V3',    prop=None, readonly=True,  shortcut=None, foreach=True)
        yield PropWrapper(name='use_smooth',     vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='vertices',       vtype='array', prop=None, readonly=True,  shortcut=None, foreach=False)        

# =============================================================================================================================
# Mesh
        
class MeshGenerator(WrapperGenerator):
    def __init__(self):
        super().__init__("Mesh")
        
    def collprops_code(self):
        yield tab2 + "self.wvertices = WMeshVertices(self.obj.vertices, self)"
        yield tab2 + "self.wedges    = WEdges(self.obj.edges, self)"
        yield tab2 + "self.wloops    = WLoops(self.obj.loops, self)"
        yield tab2 + "self.wpolygons = WPolygons(self.obj.polygons, self)"
        
    def props(self):
        yield PropWrapper(name='auto_smooth_angle', vtype='float', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='auto_texspace',     vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_auto_smooth',   vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_auto_texspace', vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)


# *****************************************************************************************************************************
# Curve

# =============================================================================================================================
# BezierSplinePoint

class SplineBezierPointGenerator(WrapperGenerator):
    def __init__(self):
        super().__init__("SplineBezierPoint")
        
    def props(self):
        yield PropWrapper(name='co',                vtype='V3',    prop=None, readonly=False, shortcut='',   foreach=True)
        yield PropWrapper(name='handle_left_type',  vtype='str',   prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='handle_left',       vtype='V3',    prop=None, readonly=False, shortcut='l',  foreach=True)
        yield PropWrapper(name='handle_right_type', vtype='str',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='handle_right',      vtype='V3',    prop=None, readonly=False, shortcut='r',  foreach=True)
        yield PropWrapper(name='hide',              vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='radius',            vtype='float', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='tilt',              vtype='int',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='weight_softbody',   vtype='float', prop=None, readonly=False, shortcut=None, foreach=True)


# =============================================================================================================================
# SplinePoint

class SplinePointGenerator(WrapperGenerator):
    def __init__(self):
        super().__init__("SplinePoint")
        
    def props(self):
        yield PropWrapper(name='co',              vtype='V3',    prop=None, readonly=False, shortcut='',   foreach=True)
        yield PropWrapper(name='radius',          vtype='float', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='tilt',            vtype='float', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='weight_softbody', vtype='float', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='weight',          vtype='float', prop=None, readonly=False, shortcut=None, foreach=True)


# =============================================================================================================================
# Spline
        
class SplineGenerator(WrapperGenerator):
    def __init__(self):
        super().__init__("Spline")
        
    def collprops_code(self):
        yield tab2 + "self.wbezier_points = WBezierSplinePoints(self.obj.bezier_points, self)"
        yield tab2 + "self.wpoints        = WBezierSplinePoints(self.obj.points, self)"
        
    def props(self):
        yield PropWrapper(name='character_index',      vtype='int',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='material_index',       vtype='int',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='order_u',              vtype='int',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='order_v',              vtype='int',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='point_count_u',        vtype='int',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='point_count_v',        vtype='int',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='radius_interpolation', vtype='str',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='resolution_u',         vtype='int',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='resolution_v',         vtype='int',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='type',                 vtype='str',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='use_bezier_u',         vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_bezier_v',         vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_cyclic_u',         vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_cyclic_v',         vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_endpoint_u',       vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_endpoint_v',       vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_smooth',           vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)        

# =============================================================================================================================
# Curve
        
class CurveGenerator(WrapperGenerator):
    def __init__(self):
        super().__init__("Curve")
        
    def collprops_code(self):
        yield tab2 + "self.wsplines = WSplines(self.obj.splines, self)"
        
    def props(self):
        yield PropWrapper(name='bevel_depth',           vtype='float',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='bevel_factor_end',      vtype='float',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='bevel_factor_start',    vtype='float',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='bevel_object',          vtype='object', prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='bevel_resolution',      vtype='int',    prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='dimensions',            vtype='str',    prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='eval_time',             vtype='float',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='extrude',               vtype='float',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='fill_mode',             vtype='str',    prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='offset',                vtype='float',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='path_duration',         vtype='int',    prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='render_resolution_u',   vtype='int',    prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='render_resolution_v',   vtype='int',    prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='resolution_u',          vtype='int',    prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='resolution_v',          vtype='int',    prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='taper_object',          vtype='object', prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='twist_smooth',          vtype='float',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_auto_texspace',     vtype='bool',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_deform_bounds',     vtype='bool',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_fill_caps',         vtype='bool',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_fill_deform',       vtype='bool',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_map_taper',         vtype='bool',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_path_follow',       vtype='bool',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_path',              vtype='bool',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_radius',            vtype='bool',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_stretch',           vtype='bool',   prop=None, readonly=False, shortcut=None, foreach=True)

        

def temp(props):
    for key, val in props.items():
        print(f"yield PropWrapper(name='{key}', vtype='{val}', prop=None, readonly=False, shortcut=None, foreach=True)")
        

# *****************************************************************************************************************************
# Object

class ObjectGenerator(WrapperGenerator):
    def __init__(self):
        super().__init__("Object", gen_coll=False, wrapper_root_class="WObjectRoot")
        
    def props(self):
        yield PropWrapper(name='active_material_index',              vtype='int',   prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='active_shape_key_index',             vtype='int',   prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='bound_box',                          vtype='bbox',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='color',                              vtype='V4',    prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='delta_location',                     vtype='V3',    prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='delta_rotation_euler',               vtype='V3',    prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='delta_rotation_quaternion',          vtype='V4',    prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='delta_scale',                        vtype='V3',    prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='empty_display_size',                 vtype='float', prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='empty_image_offset',                 vtype='V2',    prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='dimensions',                         vtype='V3',    prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='hide_render',                        vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='hide_select',                        vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='hide_viewport',                      vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='instance_faces_scale',               vtype='float', prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='location',                           vtype='V3',    prop=None, readonly=False, shortcut='',   foreach=False)
        yield PropWrapper(name='lock_scale',                         vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='matrix_basis',                       vtype='M4',    prop=None, readonly=True,  shortcut=None, foreach=False)
        yield PropWrapper(name='matrix_local',                       vtype='M4',    prop=None, readonly=True,  shortcut=None, foreach=False)
        yield PropWrapper(name='matrix_parent_inverse',              vtype='M4',    prop=None, readonly=True,  shortcut=None, foreach=False)
        yield PropWrapper(name='matrix_world',                       vtype='M4',    prop=None, readonly=True,  shortcut=None, foreach=False)
        yield PropWrapper(name='pass_index',                         vtype='int',   prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='rotation_euler',                     vtype='V3',    prop=None, readonly=False, shortcut='r',  foreach=False)
        yield PropWrapper(name='rotation_mode',                      vtype='str',   prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='rotation_quaternion',                vtype='V4',    prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='scale',                              vtype='V3',    prop=None, readonly=False, shortcut='sc', foreach=False)
        yield PropWrapper(name='show_all_edges',                     vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='show_axis',                          vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='show_bounds',                        vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='show_empty_image_only_axis_aligned', vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='show_empty_image_orthographic',      vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='show_empty_image_perspective',       vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='show_in_front',                      vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='show_instancer_for_render',          vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='show_instancer_for_viewport',        vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='show_name',                          vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='show_only_shape_key',                vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='show_texture_space',                 vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='show_transparent',                   vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='show_wire',                          vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='track_axis',                         vtype='str',   prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='type',                               vtype='str',   prop=None, readonly=True,  shortcut=None, foreach=False)
        yield PropWrapper(name='up_axis',                            vtype='str',   prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='use_empty_image_alpha',              vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='use_instance_faces_scale',           vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='use_instance_vertices_rotation',     vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='use_shape_key_edit_mode',            vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=False)

        # Array only

        yield PropWrapper(name='quaternion', vtype='V4',   readonly=False, foreach=False, array_only=True)
        yield PropWrapper(name='hide',       vtype='bool', readonly=False, foreach=False, array_only=True)
        
    def methods(self):
        yield MethodWrapper("orient",    {"axis":     "V3"}, None,    None)
        yield MethodWrapper("track_to",  {"location": "V3"}, None,    None)
        yield MethodWrapper("distances", {"location": "V3"}, "float", "distance")
        


# *****************************************************************************************************************************
# Object methods to vectorize

WObject_meths = {
    'track_to'  : ({'location': 'V3'}, None),
    'orient'    : ({'axis':     'V3'}, None),
    'distance'  : ({'location': 'V3'}, 'float'),
}

#temp(object_props)