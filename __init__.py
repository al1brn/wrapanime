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

from wrapanime.functions import bezier      as bezier
from wrapanime.utils     import blender     as blender
#from wrapanime.wrappers  import root        as root
from wrapanime.wrappers  import wrappers    as wrappers
from wrapanime.mesh      import surface     as surface
from wrapanime.mesh      import meshbuilder as meshbuilder

wrap            = wrappers.wrap
update_viewport = blender.update_viewport

BezierCurve     = bezier.BezierCurve

Surface         = surface.Surface
MeshBuilder     = meshbuilder.MeshBuilder

WObject         = wrappers.WObject
WMeshObject     = wrappers.WMeshObject
WCurveObject    = wrappers.WCurveObject

