#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 16:47:30 2020

@author: alain
"""

print("Hello")

import datetime

import props_mapping as wmaps

def get_cache_name(name):
    return '_cache_' + name

def get_plural(name):
    if name[-2:] == 'ex':
        return name[:-2] + 'ices'
    elif name[-1] == 's':
        return name + '_s'
    else:
        return name + 's'

def get_aof_name(name):
    return 'w' + name

# *****************************************************************************************************************************
# properties are gven a a dictionnary 'property name' -> 'type'
# Where type is internal notation
#
# 3 kinds of properties:
# - vectorizable : for which a numpy array can bu used
# - arrayable    : the property is an array of wrappable objects (vertices of a mesh for instanc)
# - any          : for simple 'one for one' wrapping

def vectorizable_props(props):

    vectorizable = ['int', 'bool', 'float', 'str', 'V2', 'V3', 'V4']

    for name, vtype in props.items():
        if vtype in vectorizable:
            nptype = 'np.'+vtype
            if vtype in ['int', 'bool', 'float']:
                size   = 1
            elif vtype == 'str':
                nptype = 'np.object'
                size   = 1
            elif vtype in ['V2', 'V3', 'V4']:
                nptype = 'np.float'
                size   = int(vtype[-1])

            yield name, nptype, size

    return


def arrayable_props(props):

    for name, vtype in props.items():
        if vtype[:2] == 'AW':
            class_name = vtype[1:]
            yield name, class_name

    return

def has_arrayable_props(props):
    for name, class_name in arrayable_props(props):
        return True
    return False

# *****************************************************************************************************************************
# Used for formatting the source code
    
cython = False
blanks = 4
    
tab =  " " * blanks
tab2 =  " " * (blanks*2)
tab_3 =  " " * (blanks*3)
tab__4 =  " " * (blanks*4)

com = tab + "# "

# *****************************************************************************************************************************
# Code for simple property wrapping

def gen_wrapped_properties(props):

    yield com + "="*80
    yield com + "Direct properties wrappers"
    yield ""

    for name, vtype in props.items():
        yield tab + "@property"

        if vtype in ['int', 'str', 'float', 'bool']:
            if cython:
                yield tab + f"def {vtype} {name}(self):"
            else:
                yield tab + f"def {name}(self) -> {vtype}:"
        else:
            yield tab + f"def {name}(self): # {vtype}"

        yield tab2 + f"return self.obj.{name}"

        yield ""

        yield tab + f"@{name}.setter"
        yield tab + f"def {name}(self, value): # {vtype}"
        yield tab2 + f"self.obj.{name} = value"

        yield ""

    yield ""

    return

# *****************************************************************************************************************************
# Wrap a collection of wrapped types
# Generate the code for class __init__

def gen_arrayable_props(props):

    # ---------------------------------------------------------------------------
    # No arrayable props

    if not has_arrayable_props(props):
        return

    # ---------------------------------------------------------------------------
    # Initialisation

    yield com + "="*80
    yield com + "Initialization"
    yield com + "Create cache properties"
    yield ""
    yield tab + "def __init__(self, obj, wowner):"
    yield tab2 + "super().__init__(obj, wowner)"
    yield ""

    for name, class_name in arrayable_props(props):
        prop_name  = get_aof_name(name)
        cache_name = get_cache_name(prop_name)
        yield tab2 + f"self.{cache_name:20} = None"
    yield ""

    # ---------------------------------------------------------------------------
    # Erase cache

    yield com + "="*80
    yield com + "Wrap a collection of wrapped objects"
    yield ""
    yield tab + "def erase_cache(self):"
    yield tab2 + "super().erase_cache()"
    yield ""

    for name, class_name in arrayable_props(props):
        prop_name  = get_aof_name(name)
        cache_name = get_cache_name(prop_name)
        yield tab2 + f"if self.{cache_name} is not None:"
        yield tab_3 + f"self.{cache_name}.erase_cache()"
        yield tab_3 + f"self.{cache_name} = None"
        yield ""

    # ---------------------------------------------------------------------------
    # Implementation

    for name, class_name in arrayable_props(props):
        prop_name  = get_aof_name(name)
        cache_name = get_cache_name(prop_name)

        yield tab + "@property"
        yield tab + f"def {prop_name}(self): # Array of {class_name}"

        yield tab2 + f"arrayof = self.{cache_name}"
        yield tab2 + "if arrayof is None:"
        yield tab_3 + f"arrayof = ArrayOf{class_name}(self)"
        yield tab_3 + f"arrayof.array = self.obj.{name}"
        yield tab2 + f"self.{cache_name} = arrayof"

        yield tab2 + "return arrayof"

        yield ""

    yield ""

    return

# *****************************************************************************************************************************
# Vectorized access to properties of wrapped classes

def gen_vectorized_access(props):

    yield com + "="*80
    yield com + "Vectorized access to properties in collection of wrapped classes"
    yield ""

    for name, nptype, size in vectorizable_props(props):
        names      = get_plural(name)
        cache_name = get_cache_name(names)

        yield tab + "@property"
        yield tab + "def {}(self): # array of {}".format(names, props[name])

        yield tab2 + "values = self.{}".format(cache_name)
        yield tab2 + "if values is None:"

        if size == 1:
            npcreate = "np.empty(len(self), {})".format(nptype)
        else:
            npcreate = "np.empty((len(self), {}), {})".format(size, nptype)
        yield tab_3 + "values = {}".format(npcreate)
        yield tab_3 + "for i, obj in enumerate(self._array):"
        yield tab__4 + "values[i] = obj.{}".format(name)

        yield tab_3 + f"self.{cache_name} = values"
        yield tab2 + "return values"
        yield ""

        yield tab + f"@{names}.setter"
        yield tab + f"def {names}(self, values): # array of {props[name]}"
        yield tab2 + f"self.{cache_name} = np.array(to_array(values, (len(self), {size}), '{size}-vector or array of {size}-vectors'))"
        yield tab2 + f"for obj, v in zip(self._array, self.{cache_name}):"
        yield tab_3 + f"obj.{name} = v"
        yield ""

    yield ""

    return

# *****************************************************************************************************************************
# Write an ArrayOf class

def gen_arrayof(props, class_name):

    yield "# " + "*"*(80 + blanks)
    yield "# " + "Class ArrayOf {} automatically generated by wa_sourcegen".format(class_name)
    yield ""

    yield "class ArrayOf{}(ArrayOf):".format(class_name)
    yield ""

    yield com + '='*80
    yield com + "Initialization"
    yield com + "Create the cache properties"
    yield ""
    yield tab + "def __init__(self, wowner):"
    yield tab2 + "super().__init__({}, wowner)".format(class_name)
    yield ""

    for name, nptype, size in vectorizable_props(props):
        names      = get_plural(name)
        cache_name = get_cache_name(names)
        yield tab2 + "self.{:35} = None".format(cache_name)
    yield ""

    yield com + '='*80
    yield com + "Erase cache"
    yield tab + "def erase_cache(self):"
    yield tab2 + "super().erase_cache()"
    yield ""
    for name, nptype, size in vectorizable_props(props):
        names      = get_plural(name)
        cache_name = get_cache_name(names)
        yield tab2 + "self.{:35} = None".format(cache_name)

    yield ""

    for line in gen_vectorized_access(props):
        yield line


# *****************************************************************************************************************************
# Write a Wrapper class

def gen_wrapper(props, class_name, root_class = "Wrapper", with_arrayOf = True):

    yield "# " + "*"*(80 + blanks)
    yield "# " + "Class {} automatically generated by wa_sourcegen".format(class_name)
    yield ""

    yield "class {}({}):".format(class_name, root_class)
    yield ""

    # Collection of wrapped class

    for line in gen_arrayable_props(props):
        yield line

    # Otherwize wrap the properties

    for line in gen_wrapped_properties(props):
        yield line

    yield ""

    if with_arrayOf:
        for line in gen_arrayof(props, class_name):
            yield line

    return


# *****************************************************************************************************************************
# Generate Blender Wrapping

def gen_blender_wrapping():

    yield "# " + '*'*100
    yield "# Generated {}".format(datetime.date.today())
    yield ""
    yield "import numpy as np"
    #yield "from wa_root import ArrayOf, Wrapper"
    yield "from wrapanime.root.root import ArrayOf, Wrapper"
    yield ""


    # ----- Mesh base classes

    for line in gen_wrapper(wmaps.edge_props, "WEdge"):
        yield line

    for line in gen_wrapper(wmaps.loop_props, "WLoop"):
        yield line

    for line in gen_wrapper(wmaps.polygon_props, "WPolygon"):
        yield line

    for line in gen_wrapper(wmaps.mesh_vertex_props, "WMeshVertex"):
        yield line

    # Mesh

    for line in gen_wrapper(wmaps.mesh_props, "WMesh"):
        yield line

    # ----- Curve base classes

    for line in gen_wrapper(wmaps.bezier_spline_point_props, "WBezierSplinePoint"):
        yield line

    for line in gen_wrapper(wmaps.spline_point_props, "WSplinePoint"):
        yield line

    for line in gen_wrapper(wmaps.spline_props, "WSpline"):
        yield line

    # Curve

    for line in gen_wrapper(wmaps.curve_props, "WCurve"):
        yield line

    # ----- Object

    for line in gen_wrapper(wmaps.object_props, "WObject"):
        yield line


# *****************************************************************************************************************************
# Write code in a file

def write_code(path_name):
    with open(path_name, 'w') as f:
        for line in gen_blender_wrapping():
            print(line)
            f.write(line + "\n")


file_path = '/users/alain/desktop/python.py'
file_path = '/Users/alain/OneDrive/CloudStation/Blender/scripts/modules/wa_blendergen.py'
file_path = '/Users/alain/OneDrive/CloudStation/Blender/dev/scripts/modules/wrapanime/root/generated_wrappers.py'

write_code(file_path)
