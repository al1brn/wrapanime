#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 16:48:27 2020

@author: alain
"""

import inspect

cython = False
blanks = 4
    
tab =  " " * blanks
tab2 =  " " * (blanks*2)
tab_3 =  " " * (blanks*3)
tab__4 =  " " * (blanks*4)

com = tab + "# "


# Array of properties are cached in collections

USE_CACHE = False

# *****************************************************************************************************************************
# Source code for bezier arrays
# Source code is used to generate the final code
# Can be used for 2D or 3D bezier curve
# __size is replaced by 2 or 3 at generation time

def get_bpoints(self):
    points = np.empty((len(self), __size*3), np.float).reshape(len(self), 3, __size)
    points[:, 0] = self.handle_lefts
    points[:, 1] = self.cos
    points[:, 2] = self.handle_rights
    return points.reshape(len(self), __size*3)

def set_bpoints(self, points):
    a = np.array(bpoints)
    if a.size != len(self)*__size*3:
        raise WrapException(
                "Set Bezier points error: the length of the points array is incorrect",
                f"Need: {len(self)} triplets of __size-vectors: {len(self)*__size*3}",
                f"Received: array {a.shape} of size {a.size}"
                )
        
    np.reshape(a, (len(self), 3, __size))
    self.handle_lefts  = a[:, 0]
    self.cos           = a[:, 1]
    self.handle_rights = a[:, 2]
    

# *****************************************************************************************************************************
# Generate source code from an existing source code
    
def method_from_function(f, header, hooks={}):
    
    source = inspect.getsource(f)
    for name, value in hooks.items():
        source = source.replace(name, value)
        
    lines = source.split('\n')

    # function header
    for line in header:
        yield tab + line
        
    # function body
    for i in range(1, len(lines)):
        yield tab + lines[i]

def test_method():       
    for line in method_from_function(get_bpoints, ["@property", "def bpoints3(self):"], {"__size": "3"}):
        print(line)
    for line in method_from_function(set_bpoints, ["@property", "def bpoints3(self, bpoints):"], {"__size": "3"}):
        print(line)
    


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
    def is_vector(self):
        return self.vtype in ['V2', 'V3', 'V4']
    
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
    
    def wrapping_code(self, bname="obj"):
        
        if self.array_only:
            return

        yield tab + "@property"
        yield tab + f"def {self.name}(self): # {self.vtype}"
        yield tab2 + f"return self.{bname}.{self.prop}"
        yield ""
        if (self.shortcut is not None) and (self.type_size > 1):
            for i in range(self.type_size):
                name = self.shortcut + 'xyzw'[i]
                yield tab + "@property"
                yield tab + f"def {name}(self):"
                yield tab2 + f"return self.{bname}.{self.prop}[{i}]"
                yield ""
        
        if not self.readonly:
            yield tab + f"@{self.name}.setter"
            yield tab + f"def {self.name}(self, value): # {self.vtype}"
            if self.is_vector:
                msg = f'"{self.name}"'
                yield tab2 + f"self.{bname}.{self.prop} = to_array(value, ({self.type_size},), {msg})"
            else:
                yield tab2 + f"self.{bname}.{self.prop} = value"
            yield ""
            
            if (self.shortcut is not None) and (self.type_size > 1):
                for i in range(self.type_size):
                    name = self.shortcut + 'xyzw'[i]
                    yield tab + f"@{name}.setter"
                    yield tab + f"def {name}(self, value):"
                    yield tab2 + f"self.{bname}.{self.prop}[{i}] = value"
                    yield ""
        
        return
    
    # ---------------------------------------------------------------------------
    # Vectorized access
    
    def arrayof_init_code(self):
        if USE_CACHE and self.is_vectorizable():
            yield tab2 + f"self.{self.get_cache_name:30} = None"
            
    def arrayof_erase_cache_code(self):
        if USE_CACHE and self.is_vectorizable():
            yield tab2 + f"self.{self.get_cache_name:30} = None"
    
    def arrayof_code(self, coll_name="self.coll"):
        
        if not self.is_vectorizable():
            return
        
        # Read the plural property, eg cos for vectorized access to co
        
        yield tab + "@property"
        yield tab + f"def {self.get_plural_name}(self): # Array of {self.vtype}"
            
        def read_array(array_name):
            create_array  = f"{array_name} = np.empty(len(self){self.vector_mult}, {self.np_type})"
            if self.foreach:
                yield create_array
                yield f"{coll_name}.foreach_get('{self.prop}', {array_name})"
                if USE_CACHE:
                    if self.type_size > 1:
                        yield f"{array_name} = {array_name}{self.reshape}"
                else:
                    yield f"return {array_name}{self.reshape}"
            else:
                yield f"{create_array}{self.reshape}"
                yield f"coll = {coll_name}"
                yield "for i in range(len(self)):"
                yield tab + f"{array_name}[i] = coll[i].{self.name}"
                if not USE_CACHE:
                    yield f"return {array_name}"
                    
        
        array_name = f"self.{self.get_cache_name}" if USE_CACHE else "array"
        
        if USE_CACHE:
            yield tab2 + f"if {array_name} is None:"
            for line in read_array(array_name):
                yield tab_3 + line
            yield tab2 + f"return {array_name}"
        else:
            for line in read_array(array_name):
                yield tab2 + line
        yield ""
        
        # Write the plural property if not readonly
        
        if not self.readonly:
        
            yield tab + f"@{self.get_plural_name}.setter"
            yield tab + f"def {self.get_plural_name}(self, values): # Arrayf of {self.vtype}"
            error = f"f'{self.type_size}-vector or array of " + "{len(self)}" + f" {self.type_size}-vectors'"
            array_creation = f"to_array(values, (len(self), {self.type_size}), {error})"
            
            if (not USE_CACHE) and self.foreach:
                yield tab2 + f"{coll_name}.foreach_set('{self.prop}', {array_creation}{self.reshape_1dim})"
            else:
                yield tab2 + f"{array_name} = {array_creation}"
            
                if self.foreach:
                    yield tab2 + f"{coll_name}.foreach_set('{self.prop}', {array_name}{self.reshape_1dim})"
                else:
                    yield tab2 + f"coll = {coll_name}"
                    yield tab2 + "for i in range(len(self)):"
                    yield tab_3 + f"coll[i].{self.name} = {array_name}[i]"
                
            yield ""
            
        # ---------------------------------------------------------------------------
        # Shortcuts to partial access to vector values
    
        if self.is_vectorizable() and (self.shortcut is not None) and (self.type_size >= 2):
        
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
                    if USE_CACHE:
                        yield tab2 + f"self.{self.get_plural_name}[:, {index}] = to_array(values, (len(self), 1), {error})"
                        yield tab2 + f"self.{self.get_plural_name} = self.{self.get_cache_name}"
                    else:
                        yield tab2 + f"{self.get_plural_name} = self.{self.get_plural_name}"
                        yield tab2 + f"{self.get_plural_name}[:, {index}] = to_array(values, (len(self), 1), {error})"
                        yield tab2 + f"self.{self.get_plural_name} = {self.get_plural_name}"
                        
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

    def __init__(self):
        
        self.gen_wrapper        = True
        self.init_args          = []
        self.indexed            = False
        self.class_name         = None
        self.root_class         = "Wrapper"
        self.obj                = None       # Path to the wrapped property
        self.bname              = None       # Blender name to use to name the property
        
        self.gen_coll           = True
        self.coll_class_name    = None
        self.coll_root_class    = "Wrapper"
        self.coll_obj           = None
        self.coll_bname         = None       # Blender name to use to name the property
        
        self.bpoints            = None  # Do not generate bpoints properties
        
        self.init()
        
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
    
    # ---------------------------------------------------------------------------
    # __init__ default generation if this enumerator is empty
    
    def init_method(self):
        return
        
    # --- __init__ arguments

    def init_params(self, item=True):
        s = ""
        for arg in self.init_args:
            s += ", " + arg
        if item and self.indexed:
            s += ", windex"
        return s
    
    # ---------------------------------------------------------------------------
    # __init__ for coll default generation if this enumerator is empty

    def coll_init_method(self):
        return
    
    # ---------------------------------------------------------------------------
    # __getitem__ default generation if this enumerator is empty
    
    def getitem_method(self):
        return

    # --- Collection only: how to instance in item WItem(???)
    
    @property
    def coll_item_init(self):
        s   = ""
        sep = ""
        for arg in self.init_args:
            s += f"{sep}self.{arg}"
            sep = ", "
        return s + sep + "index"
        
    # class init source code
    
    def init_code(self):
        pass
    
    def coll_init_code(self):
        pass
        
    # Does a enumerator yield lines
    
    @classmethod
    def is_not_empty(self, enum):
        try:
            for line in enum:
                return True
        except:
            return False
        
    # ====================================================================================================
    # Source code for the Wrapper Class
    
    def wrapper_code(self):
        
        yield "#" + "="*80
        yield f"# {self.class_name} class wrapper"
        yield ""
        yield f"class W{self.class_name}({self.root_class}):"
        yield ""
        
        # ---------------------------------------------------------------------------
        # Init
        
        if self.is_not_empty(self.init_method()):
            for line in self.init_method():
                yield line
        else:
            yield tab + f"def __init__(self{self.init_params(True)}):"
            for arg in self.init_args:
                yield tab2 + f"self.{arg:13} = {arg}"
            if self.indexed:
                yield tab2 + f"self.{'windex':13} = windex"
            if self.is_not_empty(self.init_code()):
                for line in self.init_code():
                    yield line
        yield ""

        # ---------------------------------------------------------------------------
        # Blender object access
        
        yield tab + "@property"
        yield tab + f"def {self.bname}(self): # The wrapped Blender {self.bname}"
        yield tab2 + f"return {self.obj}"
        yield ""
        
        # With bstruct generic name
        
        yield tab + "@property"
        yield tab + f"def bstruct(self): # The wrapped Blender {self.bname}"
        yield tab2 + f"return {self.obj}"
        yield ""
            
        # ---------------------------------------------------------------------------
        # Properties
        
        for prop in self.props():
            for line in prop.wrapping_code(self.bname):
                yield line
                
        return
    
    # ====================================================================================================
    # Source code for the Array of class
    
    def arrayof_code(self):
        
        if not self.gen_coll:
            return
        
        cname = get_plural(self.class_name)
        super_name = self.coll_root_class
        if super_name is None:
            super_name = "Wrapper"
            
        yield "#" + "="*80
        yield f"# Array of W{cname}"
        yield ""
        
        yield f"class W{cname}({super_name}):"
        yield ""
        
        # === Init
        
        if self.is_not_empty(self.coll_init_method()):
            for line in self.coll_init_method():
                yield line
        else:
            yield tab + f"def __init__(self{self.init_params(False)}):"
            for arg in self.init_args:
                yield tab2 + f"self.{arg:13} = {arg}"
            if self.is_not_empty(self.coll_init_code()):
                for line in self.coll_init_code():
                    yield line
        yield tab2 + f"self.item_class = W{self.class_name}"
        yield ""
        
        # === Collection
        
        yield tab + "@property"
        yield tab + f"def {self.coll_bname}(self): # The wrapped Blender {self.coll_bname}"
        yield tab2 + f"return {self.coll_obj}"
        yield ""
        
        yield tab + "@property"
        yield tab + f"def bcoll(self): # The wrapped Blender {self.coll_bname}"
        yield tab2 + f"return {self.coll_obj}"
        yield ""
        
        # === Array interface
        
        yield tab + "def __len__(self):"
        yield tab2 + f"return len({self.coll_obj})"
        yield ""
        
        if self.is_not_empty(self.getitem_method()):
            for line in self.getitem_method():
                yield line
        else:
            yield tab + "def __getitem__(self, index):"
            yield tab2 + f"return W{self.class_name}({self.coll_item_init})"
        yield ""
        
        # === Erase cache
        
        if USE_CACHE:
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
            for line in prop.arrayof_code(self.coll_obj):
                yield line
                
        # === Methods
        
        if self.is_not_empty(self.methods()):
            for meth in self.methods():
                for line in meth.arrayof_code():
                    yield line
        
        # === Bezier points setter and getter
        
        if self.bpoints is not None:
            
            # method name is used in the formatting string only
            meth_name = f"bpoints"
            
            for line in method_from_function(
                    get_bpoints,
                    ["@property", f"def {meth_name}(self):"],
                    {"__size": f"{self.bpoints}"}):
                yield line
                
            for line in method_from_function(
                    set_bpoints,
                    [f"@{meth_name}.setter", f"def {meth_name}(self, bpoints):"],
                    {"__size": f"{self.bpoints}"}):
                yield line
                    
        
                
        
# *****************************************************************************************************************************
# Meshes
        
# =============================================================================================================================
# MeshVertex
    
class MeshVertexGenerator(WrapperGenerator):
        
    def init(self):
        self.class_name         = "MeshVertex"
        
        self.init_args          = ["obj_name"]
        self.indexed            = True
        self.obj                = "bpy.data.objects[self.obj_name].data.vertices[self.windex]"
        self.bname              = "vertex"
        
        
        self.gen_coll           = True
        self.coll_obj           = "bpy.data.objects[self.obj_name].data.vertices"
        self.coll_bname         = "vertices"
        
        
        
    def props(self):
        yield PropWrapper(name='bevel_weight',  vtype='float', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='co',            vtype='V3',    prop=None, readonly=False, shortcut='',   foreach=True)
        yield PropWrapper(name='hide',          vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='index',         vtype='int',   prop=None, readonly=True,  shortcut=None, foreach=True)
        yield PropWrapper(name='normal',        vtype='V3',    prop=None, readonly=True,  shortcut='n',  foreach=True)
        yield PropWrapper(name='undeformed_co', vtype='V3',    prop=None, readonly=True,  shortcut=None, foreach=True)


# =============================================================================================================================
# Edge
    
class EdgeGenerator(WrapperGenerator):
    
    def init(self):
        self.class_name         = "Edge"
        
        self.init_args          = ["obj_name"]
        self.indexed            = True
        self.obj                = "bpy.data.objects[self.obj_name].data.edges[self.windex]"
        self.bname              = "edge"
        
        
        self.gen_coll           = True
        self.coll_obj           = "bpy.data.objects[self.obj_name].data.edges"
        self.coll_bname         = "edges"
    
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
    
    def init(self):
        self.class_name         = "Loop"
        
        self.init_args          = ["obj_name"]
        self.indexed            = True
        self.obj                = "bpy.data.objects[self.obj_name].data.loops[self.windex]"
        self.bname              = "loop"
        
        
        self.gen_coll           = True
        self.coll_obj           = "bpy.data.objects[self.obj_name].data.loops"
        self.coll_bname         = "loops"
        
    
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
    def init(self):
        self.class_name         = "Polygon"
        
        self.init_args          = ["obj_name"]
        self.indexed            = True
        self.obj                = "bpy.data.objects[self.obj_name].data.polygons[self.windex]"
        self.bname              = "polygons"
        
        
        self.gen_coll           = True
        self.coll_obj           = "bpy.data.objects[self.obj_name].data.polygons"
        self.coll_bname         = "polygons"
        
        
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

    def init(self):
        
        self.init_args          = ["obj_name"]
        self.class_name         = "Mesh"
        self.root_class         = "WMeshRoot"
        self.obj                = "bpy.data.objects[self.obj_name].data"
        self.bname              = "mesh"
        
        self.gen_coll           = False
        
    def init_code(self):
        yield tab2 + "self.wvertices = WMeshVertices(self.obj_name)"
        yield tab2 + "self.wedges    = WEdges(self.obj_name)"
        yield tab2 + "self.wloops    = WLoops(self.obj_name)"
        yield tab2 + "self.wpolygons = WPolygons(self.obj_name)"
        
    def props(self):
        yield PropWrapper(name='auto_smooth_angle', vtype='float', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='auto_texspace',     vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_auto_smooth',   vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_auto_texspace', vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)


# *****************************************************************************************************************************
# Curve

# =============================================================================================================================
# BezierSplinePoint

class BezierSplinePointGenerator(WrapperGenerator):

    def init(self):
        
        self.init_args          = ["obj_name", "owner_index"]
        self.indexed            = True
        self.class_name         = "BezierSplinePoint"
        self.obj                = "bpy.data.objects[self.obj_name].data.splines[self.owner_index].bezier_points[self.windex]"
        self.bname              = "bezier_point"
        
        self.gen_coll           = True
        self.coll_obj           = "bpy.data.objects[self.obj_name].data.splines[self.wowner_index].bezier_points"
        self.coll_bname         = "bezier_points"
        
        self.bpoints            = 3
        
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
    
    def init(self):
        
        self.init_args          = ["obj_name", "owner_index"]
        self.indexed            = True
        self.class_name         = "SplinePoint"
        self.obj                = "bpy.data.objects[self.obj_name].data.splines[self.owner_index].points[self.windex]"
        self.bname              = "point"
        
        self.gen_coll           = True
        self.coll_obj           = "bpy.data.objects[self.obj_name].data.splines[self.owner_index].points"
        self.coll_bname         = "points"
        
    def props(self):
        yield PropWrapper(name='co',              vtype='V4',    prop=None, readonly=False, shortcut='',   foreach=True)
        yield PropWrapper(name='radius',          vtype='float', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='tilt',            vtype='float', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='weight_softbody', vtype='float', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='weight',          vtype='float', prop=None, readonly=False, shortcut=None, foreach=True)


# =============================================================================================================================
# Spline
        
class SplineGenerator(WrapperGenerator):
        
    def init(self):
        
        self.init_args          = ["obj_name"]
        self.indexed            = True
        self.class_name         = "Spline"
        self.root_class         = "WSplineRoot"
        self.obj                = "bpy.data.objects[self.obj_name].data.splines[self.windex]"
        self.bname              = "spline"
        
        
        self.gen_coll           = True
        self.coll_root_class    = "WSplinesRoot"
        self.coll_obj           = "bpy.data.objects[self.obj_name].data.splines"
        self.coll_bname         = "splines"
        
        
    def init_code(self):
        yield tab2 + "self.wbezier_points = WBezierSplinePoints(self.obj_name, self.windex)"
        yield tab2 + "self.wpoints        = WSplinePoints(self.obj_name, self.windex)"
        
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
        
    def init(self):
        
        self.init_args          = ["obj_name"]
        self.class_name         = "Curve"
        self.obj                = "bpy.data.objects[self.obj_name].data"
        self.bname              = "curve"
        
        self.gen_coll           = False
        
    def init_code(self):
        yield tab2 + "self.wsplines = WSplines(self.obj_name)"
        
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

        # Homonyms
        yield PropWrapper(name="t0", vtype='float',  prop='bevel_factor_end',      readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name="t1", vtype='float',  prop='bevel_factor_start',    readonly=False, shortcut=None, foreach=True)


# *****************************************************************************************************************************
# Object

class ObjectGenerator(WrapperGenerator):
        
    def init(self):
        
        self.init_args          = ["name"]
        self.class_name         = "Object"
        self.root_class         = "WObjectRoot"
        self.obj                = "bpy.data.objects[self.name]"
        self.bname              = "object"
        
        self.gen_coll           = True
        self.coll_init_args     = ["coll_name"]
        self.coll_obj           = "bpy.data.collections[self.coll_name].objects"
        self.coll_bname         = "objects"
        
        
    def init_method(self):
        yield tab + "def __init__(name):"
        yield tab2 + "self.name = name"
        
    def coll_init_method(self):
        yield tab + "def __init__(self, coll_name):"
        yield tab2 + "self.coll_name = coll_name"
        
    def getitem_method(self):
        yield tab + "def __getitem__(self, index):"
        yield tab2 + f"return self.item_class({self.coll_obj}[index].name)"
        
    def props(self):
        yield PropWrapper(name='active_material_index',              vtype='int',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='active_shape_key_index',             vtype='int',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='bound_box',                          vtype='bbox',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='color',                              vtype='V4',    prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='delta_location',                     vtype='V3',    prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='delta_rotation_euler',               vtype='V3',    prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='delta_rotation_quaternion',          vtype='V4',    prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='delta_scale',                        vtype='V3',    prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='empty_display_size',                 vtype='float', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='empty_image_offset',                 vtype='V2',    prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='dimensions',                         vtype='V3',    prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='hide_render',                        vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='hide_select',                        vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='hide_viewport',                      vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='instance_faces_scale',               vtype='float', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='location',                           vtype='V3',    prop=None, readonly=False, shortcut='',   foreach=True)
        yield PropWrapper(name='lock_scale',                         vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='matrix_basis',                       vtype='M4',    prop=None, readonly=True,  shortcut=None, foreach=False)
        yield PropWrapper(name='matrix_local',                       vtype='M4',    prop=None, readonly=True,  shortcut=None, foreach=False)
        yield PropWrapper(name='matrix_parent_inverse',              vtype='M4',    prop=None, readonly=True,  shortcut=None, foreach=False)
        yield PropWrapper(name='matrix_world',                       vtype='M4',    prop=None, readonly=True,  shortcut=None, foreach=False)
        yield PropWrapper(name='pass_index',                         vtype='int',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='rotation_euler',                     vtype='V3',    prop=None, readonly=False, shortcut='r',  foreach=True)
        yield PropWrapper(name='rotation_mode',                      vtype='str',   prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='rotation_quaternion',                vtype='V4',    prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='scale',                              vtype='V3',    prop=None, readonly=False, shortcut='s',  foreach=True)
        yield PropWrapper(name='show_all_edges',                     vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='show_axis',                          vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='show_bounds',                        vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='show_empty_image_only_axis_aligned', vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='show_empty_image_orthographic',      vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='show_empty_image_perspective',       vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='show_in_front',                      vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='show_instancer_for_render',          vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='show_instancer_for_viewport',        vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='show_name',                          vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='show_only_shape_key',                vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='show_texture_space',                 vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='show_transparent',                   vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='show_wire',                          vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='track_axis',                         vtype='str',   prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='type',                               vtype='str',   prop=None, readonly=True,  shortcut=None, foreach=False)
        yield PropWrapper(name='up_axis',                            vtype='str',   prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='use_empty_image_alpha',              vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_instance_faces_scale',           vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_instance_vertices_rotation',     vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_shape_key_edit_mode',            vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)

        # Array only

        #yield PropWrapper(name='quaternion', vtype='V4',   readonly=False, foreach=False, array_only=True)
        #yield PropWrapper(name='hide',       vtype='bool', readonly=False, foreach=False, array_only=True)
        
    def methods(self):
        yield MethodWrapper("orient",    {"axis":     "V3"}, None,    None)
        yield MethodWrapper("track_to",  {"location": "V3"}, None,    None)
        yield MethodWrapper("distances", {"location": "V3"}, "float", "distance")
        
        
# *****************************************************************************************************************************
# Particle

class ParticleGenerator(WrapperGenerator):
        
    def init(self):
        
        self.init_args          = ["obj_name", "owner_index"]
        self.indexed            = True
        self.class_name         = "Particle"
        self.obj                = "bpy.data.objects[self.obj_name].particle_systems[self.wowner_index].particles[self.windex]"
        self.bname              = "particle"
        
        self.gen_coll           = True
        self.coll_obj           = "bpy.data.objects[self.obj_name].particle_systems[self.wowner_index].particles"
        self.coll_bname         = "particles"
        
        
    def props(self):
        yield PropWrapper(name='alive_state',     vtype='str',    prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='angular_velocity',vtype='V3',     prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='birth_time',      vtype='float',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='die_time',        vtype='float',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='is_exist',        vtype='bool',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='is_visible',      vtype='bool',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='lifetime',        vtype='float',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='location',        vtype='V3',     prop=None, readonly=False, shortcut="",   foreach=True)
        yield PropWrapper(name='prev_angular_velocity',vtype='V3',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='prev_location',   vtype='V3',     prop=None, readonly=False, shortcut="prev", foreach=True)
        yield PropWrapper(name='prev_rotation',   vtype='V4',     prop=None, readonly=False, shortcut="prevq", foreach=True)
        yield PropWrapper(name='prev_velocity',   vtype='V3',     prop=None, readonly=False, shortcut="prevv", foreach=True)
        yield PropWrapper(name='rotation',        vtype='V4',     prop=None, readonly=False, shortcut="q",  foreach=True)
        yield PropWrapper(name='size',            vtype='float',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='velocity',        vtype='V3',     prop=None, readonly=False, shortcut="v",  foreach=True)


# =============================================================================================================================
# Particle settings
        
class ParticleSettingsGenerator(WrapperGenerator):
        
    def init(self):
        
        self.init_args          = ["obj_name", "owner_index"]
        self.class_name         = "ParticleSettings"
        self.obj                = "bpy.data.objects[self.obj_name].particle_systems[self.wowner_index].settings"
        self.bname              = "settings"
        
        self.gen_coll           = False
        
    def props(self):
        yield PropWrapper(name='adaptive_angle',vtype='int',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='adaptive_pixel',vtype='int',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='angular_velocity_factor',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='angular_velocity_mode',vtype='str',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='apply_effector_to_children',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='apply_guide_to_children',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='bending_random',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='branch_threshold',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='brownian_factor',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='child_length',  vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='child_length_threshold',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='child_nbr',     vtype='int',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='child_parting_factor',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='child_parting_max',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='child_parting_min',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='child_radius',  vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='child_roundness',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='child_size',    vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='child_size_random',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='child_type',    vtype='str',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='clump_factor',  vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='clump_noise_size',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='clump_shape',   vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='color_maximum', vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='count',         vtype='int',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='courant_target',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='create_long_hair_children',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='damping',       vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='display_color', vtype='str',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='display_method',vtype='str',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='display_percentage',vtype='int',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='display_size',  vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='display_step',  vtype='int',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='distribution',  vtype='str',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='drag_factor',   vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='effect_hair',   vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='effector_amount',vtype='int',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='emit_from',     vtype='str',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='factor_random', vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='frame_end',     vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='frame_start',   vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='grid_random',   vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='grid_resolution',vtype='int',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='hair_length',   vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='hair_step',     vtype='int',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='hexagonal_grid',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='integrator',    vtype='str',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='invert_grid',   vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='is_embedded_data',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='is_fluid',      vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='is_library_indirect',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='jitter_factor', vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='keyed_loops',   vtype='int',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='keys_step',     vtype='int',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='kink',          vtype='str',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='kink_amplitude',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='kink_amplitude_clump',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='kink_amplitude_random',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='kink_axis',     vtype='str',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='kink_axis_random',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='kink_extra_steps',vtype='int',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='kink_flat',     vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='kink_frequency',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='kink_shape',    vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='length_random', vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='library',       vtype='???',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='lifetime',      vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='lifetime_random',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='line_length_head',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='line_length_tail',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='lock_boids_to_surface',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='mass',          vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='material',      vtype='int',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='material_slot', vtype='str',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='name',          vtype='str',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='name_full',     vtype='str',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='normal_factor', vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='object_align_factor',vtype='V3',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='object_factor', vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='particle_factor',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='particle_size', vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='path_end',      vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='path_start',    vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='phase_factor',  vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='phase_factor_random',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='physics_type',  vtype='str',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='radius_scale',  vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='react_event',   vtype='str',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='reactor_factor',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='render_step',   vtype='int',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='render_type',   vtype='str',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='rendered_child_count',vtype='int',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='root_radius',   vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='rotation_factor_random',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='rotation_mode', vtype='str',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='roughness_1',   vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='roughness_1_size',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='roughness_2',   vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='roughness_2_size',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='roughness_2_threshold',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='roughness_end_shape',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='roughness_endpoint',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='shape',         vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='show_guide_hairs',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='show_hair_grid',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='show_health',   vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='show_number',   vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='show_size',     vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='show_unborn',   vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='show_velocity', vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='size_random',   vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='subframes',     vtype='int',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='tag',           vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='tangent_factor',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='tangent_phase', vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='time_tweak',    vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='timestep',      vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='tip_radius',    vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='trail_count',   vtype='int',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='twist',         vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='type',          vtype='str',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_absolute_path_time',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_adaptive_subframes',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_advanced_hair',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_close_tip', vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_clump_curve',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_clump_noise',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_collection_count',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_collection_pick_random',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_dead',      vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_die_on_collision',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_dynamic_rotation',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_emit_random',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_even_distribution',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_fake_user', vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_global_instance',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_hair_bspline',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_modifier_stack',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_multiply_size_mass',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_parent_particles',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_react_multiple',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_react_start_end',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_regrow_hair',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_render_adaptive',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_rotation_instance',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_rotations', vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_roughness_curve',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_scale_instance',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_self_effect',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_size_deflect',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_strand_primitive',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_twist_curve',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_velocity_length',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_whole_collection',vtype='bool', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='userjit',       vtype='int',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='users',         vtype='int',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='virtual_parents',vtype='float',prop=None, readonly=False, shortcut=None, foreach=True)


# =============================================================================================================================
# Particle system

class ParticleSystemGenerator(WrapperGenerator):
        
    def init(self):
        
        self.init_args          = ["obj_name"]
        self.indexed            = True
        self.class_name         = "ParticleSystem"
        self.obj                = "bpy.data.objects[self.obj_name].particle_systems[self.windex]"
        self.bname              = "particle_system"
        
        self.gen_coll           = False
        
    def init_code(self):
        yield tab2 + "self.wsettings  = WParticleSettings(self.obj_name, self.windex)"
        yield tab2 + "self.wparticles = WParticles(self.obj_name, self.windex)"
        
    def props(self):
        yield PropWrapper(name='child_seed',                    vtype='int',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='dt_frac',                       vtype='float', prop=None, readonly=True,  shortcut=None, foreach=True)
        yield PropWrapper(name='invert_vertex_group_clump',     vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='invert_vertex_group_density',   vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='invert_vertex_group_field',     vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='invert_vertex_group_kink',      vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='invert_vertex_group_length',    vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='invert_vertex_group_rotation',  vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='invert_vertex_group_roughness_1',vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='invert_vertex_group_roughness_2',vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='invert_vertex_group_roughness_end',vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='invert_vertex_group_size',      vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='invert_vertex_group_tangent',   vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='invert_vertex_group_twist',     vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='invert_vertex_group_velocity',  vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='is_editable',                   vtype='bool',  prop=None, readonly=True,  shortcut=None, foreach=True)
        yield PropWrapper(name='is_edited',                     vtype='bool',  prop=None, readonly=True,  shortcut=None, foreach=True)
        yield PropWrapper(name='is_global_hair',                vtype='bool',  prop=None, readonly=True,  shortcut=None, foreach=True)
        yield PropWrapper(name='ps_name',                       vtype='str',   prop="name", readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='seed',                          vtype='int',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_hair_dynamics',             vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='use_keyed_timing',              vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='vertex_group_clump',            vtype='str',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='vertex_group_density',          vtype='str',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='vertex_group_field',            vtype='str',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='vertex_group_kink',             vtype='str',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='vertex_group_length',           vtype='str',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='vertex_group_rotation',         vtype='str',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='vertex_group_roughness_1',      vtype='str',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='vertex_group_roughness_2',      vtype='str',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='vertex_group_roughness_end',    vtype='str',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='vertex_group_size',             vtype='str',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='vertex_group_tangent',          vtype='str',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='vertex_group_twist',            vtype='str',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='vertex_group_velocity',         vtype='str',   prop=None, readonly=False, shortcut=None, foreach=True)


# *****************************************************************************************************************************
# Texture
    
class TextureGenerator(WrapperGenerator):
        
    def init(self):
        
        self.init_args          = ["name"]
        self.class_name         = "Texture"
        self.obj                = "bpy.data.texture[self.name]"
        self.bname              = "texture"
        
        self.gen_coll           = False
        
    def props(self):
        yield PropWrapper(name='cloud_type',          vtype='str',   prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='contrast',            vtype='float', prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='factor_blue',         vtype='float', prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='factor_green',        vtype='float', prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='factor_red',          vtype='float', prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='intensity',           vtype='float', prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='is_embedded_data',    vtype='bool',  prop=None, readonly=True,  shortcut=None, foreach=False)
        yield PropWrapper(name='is_evaluated',        vtype='bool',  prop=None, readonly=True,  shortcut=None, foreach=False)
        yield PropWrapper(name='is_library_indirect', vtype='bool',  prop=None, readonly=True,  shortcut=None, foreach=False)
        yield PropWrapper(name='nabla',               vtype='float', prop=None, readonly=False, shortcut=None, foreach=False)
        #yield PropWrapper(name='name',                vtype='str',   prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='name_full',           vtype='str',   prop=None, readonly=True,  shortcut=None, foreach=False)
        yield PropWrapper(name='noise_basis',         vtype='str',   prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='noise_depth',         vtype='int',   prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='noise_scale',         vtype='float', prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='noise_type',          vtype='str',   prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='saturation',          vtype='float', prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='tag',                 vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='type',                vtype='str',   prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='use_clamp',           vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='use_color_ramp',      vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='use_fake_user',       vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='use_nodes',           vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=False)
        yield PropWrapper(name='use_preview_alpha',   vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=False)



# *****************************************************************************************************************************
# Keyframe
        
class KeyFrameGenerator(WrapperGenerator):
    """
    def __init__(self):
        super().__init__("KeyFrame")
        self.bpoints = 2
        
    TO BE UPDATED !!!!!!
    """
    
    def init(self):
        
        self.init_args          = []
        self.class_name         = "KeyFrame"
        self.obj                = "bpy.data.textures[self.name]"
        self.bname              = "key_frame"
        
        
        self.gen_coll           = True
        self.coll_obj           = "bpy.data.textures[self.name]"
        self.coll_bname         = "key_frames"
        
        
        self.bpoints            = 2
    
        
    def props(self):
        yield PropWrapper(name='amplitude',           vtype='float', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='back',                vtype='float', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='co',                  vtype='V2',    prop=None, readonly=False, shortcut='',   foreach=True)
        yield PropWrapper(name='easing',              vtype='str',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='handle_left',         vtype='V2',    prop=None, readonly=False, shortcut='l',  foreach=True)
        yield PropWrapper(name='handle_left_type',    vtype='str',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='handle_right' ,       vtype='V2',    prop=None, readonly=False, shortcut='r',  foreach=True)
        yield PropWrapper(name='handle_right_type',   vtype='str',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='interpolation',       vtype='str',   prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='period',              vtype='float', prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='select_control_point',vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='select_left_handle',  vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        yield PropWrapper(name='select_right_handle', vtype='bool',  prop=None, readonly=False, shortcut=None, foreach=True)
        








