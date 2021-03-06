# =============================================================================================================================
# Ensure the package is accessible

import sys
import pathlib

wapath = str(pathlib.Path(__file__).parent.absolute())
waparent = str(pathlib.Path(__file__).parent.parent.absolute())

def clean_path(fp):
    try:
        sys.path.remove(fp)
    except:
        pass
    
clean_path(".")
clean_path(wapath)
clean_path(waparent)

sys.path.append(waparent)

    
# =============================================================================================================================
# Declare the useful objects

from wrapanime.functions  import bezier      as bezier
from wrapanime.utils      import blender     as blender
#from wrapanime.wrappers  import root        as root
from wrapanime.wrappers   import wrappers    as wrappers
from wrapanime.wrappers   import duplicator  as duplicator
from wrapanime.wrappers   import particles   as particles
from wrapanime.mesh       import surface     as surface
from wrapanime.mesh       import meshbuilder as meshbuilder

from wrapanime.wrappers   import noise       as noise
from wrapanime.functions  import tween       as tween


Noise           = noise.Noise

wrap            = wrappers.wrap
update_viewport = blender.update_viewport

BezierCurve     = bezier.BezierCurve

Surface         = surface.Surface
MeshBuilder     = meshbuilder.MeshBuilder

WObject         = wrappers.WObject
WEmpty          = wrappers.WEmpty
WMeshObject     = wrappers.WMeshObject
WCurveObject    = wrappers.WCurveObject

Duplicator      = duplicator.Duplicator
Particles       = particles.Particles

wrap_collection   = blender.wrap_collection
hidden_collection = blender.hidden_collection
get_collection    = blender.get_collection


WFCurve           = tween.WFCurve
Interpolation     = tween.Interpolation

