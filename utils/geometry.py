from math import sin, cos, tan, atan2, pi, sqrt

import numpy as np

import bpy

from mathutils import Vector, Euler, Matrix, Quaternion

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
            error_header("get_axis ERROR", message="Unknown axis specification: '{}' not in 'XYZ'.".format(axis))

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
        error_header("tracker_quaternion ERROR: Axis or Target Axis is null !", message="axis: {} ({}), target: {} ({})".format(ax, axis, tax, target_axis))

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


# *****************************************************************************************************************************
# *****************************************************************************************************************************


def to_spherical(V):

    Vxy = Vector((V[0], V[1]))

    theta = atan2(Vxy.y, Vxy.x)
    phi   = atan2(V[2], Vxy.length)

    return (Vector(V).length, theta, phi)

def to_cylindric(V):
    Vxy = Vector((V[0], V[1]))

    theta = atan2(Vxy.y, Vxy.x)
    return (Vxy.length, theta, V[2])

# ******************************************************************************************************************************************************
# ******************************************************************************************************************************************************
# Geom
# ******************************************************************************************************************************************************
# ******************************************************************************************************************************************************

def linear(v, space=(0., 1.), output=(0., 1.)):
    res = output[0] + (v-space[0])/(space[1]-space[0])*(output[1]-output[0])
    #print(v, space, output, res)
    if output[0] < output[1]:
        res = min(output[1], max(output[0], res))
        #print("-->", res)
    else:
        res = min(output[0], max(output[1], res))
    return res

def mixer(v0, v1, factor, extrapolation='CONSTANT'):
    v = (1.-factor)*v0 + factor*v1
    if v0 < v1:
        v = min(v1, max(v0, v))
    else:
        v = min(v1, max(v0, v))
