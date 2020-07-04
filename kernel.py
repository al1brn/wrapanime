#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 15:08:42 2020

@author: alain
"""

import time
import inspect

import bpy

# ******************************************************************************************************************************************************
# ******************************************************************************************************************************************************
# function call
# ******************************************************************************************************************************************************
# ******************************************************************************************************************************************************

# DEPRECATED

def call_function(f, **kwargs):

    # Get the full argument specifications
    f_args = inspect.getfullargspec(f)

    index = kwargs.get("index")

    # Prepare the dictionnary to call the function f
    args = {}

    # The argument names are in f_args.args
    # The first ones have not necessarily a default value
    # Check that the **kwargs contain the necessary values

    total      = len(f_args.args)
    def_count  = len(f_args.defaults)

    for i in range(total-def_count):
        arg = f_args.args[i]
        try:
            args[arg] = kwargs[arg]
        except:
            error_header("Missing argument ERROR", kwargs)
            for kw, arg in kwargs.items():
                print("   {:9}: {}".format(kw, arg))
            raise NameError("Argument '{}' is mandatory in the fonction '{}' but kwargs list doesn't include it.".format(arg, f.__name__))

    # The same for arguments with default value but no exception
    for i in range(def_count):
        arg = f_args.args[total-def_count+i]
        try:
            args[arg] = kwargs[arg]
        except:
            args[arg] = f_args.defaults[i]

    # Now, need to see if args in kwargs are indexed liste

    for kws, arg in kwargs.items():
        if kws[-1] == 's' and len(kws) > 1 and issubclass(type(arg), List):
            kw = kws[:-1]
            if kw in args.keys():
                args[kw] = arg[index]

    # Let's call the function and return the result

    return f(**args)

# ******************************************************************************************************************************************************
# ******************************************************************************************************************************************************
# Kernel handler
# ******************************************************************************************************************************************************
# ******************************************************************************************************************************************************

update_functions = []

def update_clear():
    global update_functions
    update_functions = []

def update_register(f):
    update_functions.append(f)

def kernel_handler(scene):
    t0 = time.perf_counter()

    # Loops
    for f in update_functions:
        f(scene)

    duration = time.perf_counter() - t0
    if False:
        if True:
            print(scene.frame_current, ";", duration*1000)
        else:
            if duration < 1.:
                s = "{:.2f} ms".format(duration*1000)
            else:
                s = "{:.3f} s".format(duration)

            print("Frame {:4d}: {}".format(scene.frame_current, s))

# Execute once

def execute():
    kernel_handler(bpy.context.scene)


# ******************************************************************************************************************************************************
# ******************************************************************************************************************************************************
# Registering
# ******************************************************************************************************************************************************
# ******************************************************************************************************************************************************

def menu_func(self, context):
    #self.layout.operator(AddMoebius.bl_idname, icon='MESH_ICOSPHERE')
    pass


def register():
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(kernel_handler)
    pass

    #bpy.utils.register_class(AddMoebius)
    #bpy.types.VIEW3D_MT_mesh_add.append(menu_func)


def unregister():
    bpy.app.handlers.frame_change_pre.remove(kernel_handler)

    #bpy.utils.unregister_class(AddArrow)

    #bpy.types.VIEW3D_MT_mesh_add.remove(menu_func)


if __name__ == "__main__":
    register()
