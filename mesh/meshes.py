#!/usr/bin/env python

"""
This file is part of wrapanime (Animation helper add-on for Blender).
Copyright (C) 2020 Alain Bernard
wrapanime@ligloo.net
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from math import cos, sin

import bpy

from mathutils import Matrix, Vector, Quaternion

from wrapanime.mesh.meshbuilder import MeshBuilder, Topology
from wrapanime.mesh.surface import Surface


def grid(size=2., x_count=10, y_count=10):
    surf = Surface(x0=-size/2, x1=size/2, y0=-size/2, y1=size/2, x_count=x_count, y_count=y_count)
    return MeshBuilder.FromSurface(surf)

def cylinder(radius=1., height=2., x_count=2, y_count=32):
    surf = Surface.Cylinder(radius=radius, z0=-height/2, z1=height/2, x_count=x_count, y_count=y_count)
    return MeshBuilder.FromSurface(surf)

def disk(radius=1., x_count=2, y_count=32):
    surf = Surface.Disk(radius=radius, x_count=x_count, y_count=y_count)
    return MeshBuilder.FromSurface(surf)

def cone(radius=1., slope=1., height=2., x_count=8, y_count=32):
    surf = Surface.Disk(radius=radius, x_count=x_count, y_count=y_count)
    surf.func = lambda rho, theta, t=slope: rho*slope
    return MeshBuilder.FromSurface(surf)
    
def sphere(radius=1., x_count=15, y_count=32):
    surf = Surface.Sphere(radius=radius, x_count=x_count, y_count=y_count)
    return MeshBuilder.FromSurface(surf)

def torus(radius=1., minor=0.25, x_count=48, y_count=12):
    surf = Surface.Torus(major_radius=radius, minor_radius=minor, x_count=x_count, y_count=y_count)
    return MeshBuilder.FromSurface(surf)




    
