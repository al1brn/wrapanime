import itertools

from math import sin, cos, tan, atan2, pi, sqrt

import numpy as np

import bpy

from mathutils import Vector, Euler, Matrix, Quaternion

from wrapanime.utils.errors import WrapException

# ******************************************************************************************************************************************************
# ******************************************************************************************************************************************************
# Orientation
# ******************************************************************************************************************************************************
# ******************************************************************************************************************************************************

# -----------------------------------------------------------------------------------------------------------------------------

def get_axis(axis):
    """Get a Vector from an axis specification

    The specication can be either a letter in ['X', 'Y', 'Z'] or a `Vector` or a tuple.


    Parameters
    ----------
    axis : char in ['X', 'Y', 'Z'] or Vector like
        The axis specification.

    Returns
    -------
    Vector
        The vector corresponding to the axis specification.

    """
    if type(axis) is str:
        L = axis[-1].upper()
        fact = -1 if axis[0] == '-' else 1
        if L == 'X':
            return fact*Vector((1., 0., 0.))
        elif L == 'Y':
            return fact*Vector((0., 1., 0.))
        elif L == 'Z':
            return fact*Vector((0., 0., 1.))
        else:
            raise WrapException(
                    f"get_axis ERROR: unknown axis specification: '{axis}' not in 'XYZ'."
                    )

    return Vector(axis)

# *****************************************************************************************************************************
# Get a quaternion which orient a given direction toward a target direction
# Another direction, the "up", is adjusted upwards, ie along z (the up direction)
#
# - axis           : the vector to orient toward target_axis
# - target_axis    : the new direction toward which axis will point
# - up             : The up direction wich will be in the plane (target_axis, z)
# - up_direction   : An alternative to z in the (target_axis, z)

def tracker_quaternion(axis, target_axis, up='Y', up_direction='Z'):

    ax  = get_axis(axis)
    tax = get_axis(target_axis)
    if tax.length < 0.0001:
        tax = get_axis('Z')

    # Rotation from ax to tax
    try:
        angle = tax.angle(ax)
    except:
        raise WrapException(
                "tracker_quaternion ERROR: Axis or Target Axis is null !",
                f"axis: {ax} ({axis}), target: {tax} ({target_axis})"
                )

    P = tax.cross(ax)
    q = Quaternion(P, -angle)

    # Need to rotate around target_axis to align rotated up in the plane (target_axis, up direction)

    # up  rotated and the projected on the plane perp to target_axis
    upv = get_axis(up)
    upv.rotate(q)
    upv = upv - upv.dot(tax)*tax

    # up direction projected on the plane perpendicular to target_axis
    # No rotation since the target up direction is "absolute"
    upd = get_axis(up_direction)
    upd = upd - upd.dot(tax)*tax

    # Angle between the projections, then rotation
    ag = upv.angle(upd, 0.)
    if upv.cross(upd).dot(tax) < 0:
        ag = - ag
    qr = Quaternion(tax, ag)

    # Combine and return
    q.rotate(qr)
    return q

# *****************************************************************************************************************************
# A matrix to orient a figure drawn in XY plane perpendicular to an arbitrary axis
        
def XY_rotation(axis='Z', up='X'):
    return tracker_quaternion('Z', target_axis=axis, up=up)


# *****************************************************************************************************************************
# Orienter (Build in progress)

class Orienter():
    def __init__(self, axis='Z', up='Y', up_direction='Z'):
        self.axis   = get_axis(axis)
        self.up     = get_axis(up)
        self.up_dir = get_axis(up_direction)

    def execute(self, target):
        return tracker_quaternion(self.axis, target, up=self.up, up_direction= self.up_dir)

    def track_to(self, item, location):
        q = self.execute(Vector(location) - item.location)
        item.quaternion=  q

    def orient(self, item, vector):
        q = self.execute(vector)
        item.quaternion = q



