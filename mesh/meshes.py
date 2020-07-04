import bpy
import bmesh
from mathutils import Matrix, Vector, Quaternion
from math import cos, sin, radians, degrees, pi, atan2, sqrt, acos, tan
import numpy as np

from wrapanime.mesh.meshbuilder import MeshBuilder

def clip(v, vmin, vmax):
    return max(vmin, min(vmax, v))

def grid(size=(2., 2.), rings=10, segments=10):
    rings    = clip(rings, 2, 100)
    segments = clip(segments, 2, 100)
    
