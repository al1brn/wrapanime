#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 16:47:30 2020

@author: alain
"""

import datetime

import props_mapping as wmaps

# *****************************************************************************************************************************
# Generate Blender Wrapping

def wrappers_file():

    yield "# " + '*'*100
    yield "# Generated {}".format(datetime.date.today())
    yield ""
    yield "import numpy as np"
    yield "from wrapanime.wrappers.root import Wrapper, to_array, WObjectRoot, WMeshRoot, WSplineRoot, WSplinesRoot"
    yield "from wrapanime.utils.errors import WrapException"
    yield "import bpy"
    yield ""
    
    def generate(gen):
        for line in gen.wrapper_code():
            yield line
            
        for line in gen.collection_code():
            yield line
    
    for line in generate(wmaps.MeshVertexGenerator()):
        yield line
        
    for line in generate(wmaps.EdgeGenerator()):
        yield line
    for line in generate(wmaps.LoopGenerator()):
        yield line
    for line in generate(wmaps.PolygonGenerator()):
        yield line
        
    for line in generate(wmaps.MeshGenerator()):
        yield line
    
    for line in generate(wmaps.BezierSplinePointGenerator()):
        yield line
    for line in generate(wmaps.SplinePointGenerator()):
        yield line
    for line in generate(wmaps.SplineGenerator()):
        yield line
        
    for line in generate(wmaps.CurveGenerator()):
        yield line
    
    for line in generate(wmaps.ObjectGenerator()):
        yield line
        
        
        
    for line in generate(wmaps.ParticleGenerator()):
        yield line
        
    for line in generate(wmaps.ParticleSystemGenerator()):
        yield line
   
        
    for line in generate(wmaps.TextureGenerator()):
        yield line
        
    for line in generate(wmaps.KeyFrameGenerator()):
        yield line
        
    
# *****************************************************************************************************************************
# Write code in a file

def write_code(path_name):
    with open(path_name, 'w') as f:
        for line in wrappers_file():
            print(line)
            f.write(line + "\n")


file_path = '/users/alain/desktop/python.py'
file_path = '/Users/alain/OneDrive/CloudStation/Blender/scripts/modules/wa_blendergen.py'
file_path = '/Users/alain/OneDrive/CloudStation/Blender/dev/scripts/modules/wrapanime/wrappers/generated_wrappers.py'



#file_path = "/Users/alain.bernard@loreal.com/wrapanime/wrappers/generated_wrappers.py"

write_code(file_path)
    
