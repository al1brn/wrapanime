import itertools

from math import sin, cos, tan, atan2, pi, sqrt, degrees, radians

import numpy as np

#import bpy

#from mathutils import Vector, Euler, Matrix, Quaternion

#from wrapanime.utils.errors import WrapException

WrapException = Exception

# Default ndarray float type
ftype = np.float

# Zero
zero = 0.0001

# -----------------------------------------------------------------------------------------------------------------------------
# Get an axis

def get_axis(straxis, default=(0, 0, 1)):
    """Axis can be defined aither by a letter or a vector.
    
    Parameters
    ----------
    staraxis: array
        array of vector specs, ie triplets or letters: [(1, 2, 3), 'Z', (1, 2, 3), '-X']
    
    Returns
    -------
    array of normalized vectors
    
    """
    
    axis = np.array(straxis)
    hasstr = str(axis.dtype)[0] in ['o', '<']
    
    if hasstr:
        single = len(axis.shape) == 0
    else:
        single = len(axis.shape) == 1
    
    if single:
        axis = np.array([axis])
        
    rem = np.arange(len(axis))
    As  = np.zeros((len(axis), 3), ftype)
    
    # Axis by str
    if hasstr:
        
        pxs = np.where(axis == 'X')
        pys = np.where(axis == 'Y')
        pzs = np.where(axis == 'Z')
        nxs = np.where(axis == '-X')
        nys = np.where(axis == '-Y')
        nzs = np.where(axis == '-Z')
        
        As[pxs] = [ 1,  0,  0]
        As[pys] = [ 0,  1,  0]
        As[pzs] = [ 0,  0,  1]
        
        As[nxs] = [-1,  0,  0]
        As[nys] = [ 0, -1,  0]
        As[nzs] = [ 0,  0, -1]
        
        chs = np.append(pxs, pys)
        chs = np.append(chs, pzs)
        chs = np.append(chs, nxs)
        chs = np.append(chs, nys)
        chs = np.append(chs, nzs)
        
        rem = np.delete(rem, chs)
        
        with_chars = True
        
    else:
        # The axis can be a single vector
        # In that case the length of axis is 3 which is not
        # the number of expected vectors
        
        if axis.size == 3:
            As  = np.zeros(3, ftype).reshape(1, 3)
            rem = np.array([0])
            
        with_chars = False

            
    # Axis by vectors
    if len(rem > 0):
        
        if with_chars:
            # Didn't find better to convert np.object to np.ndarray :-()
            # Mixing letters and vectors should be rare
            for i in rem:
                As[i] = axis[i]
        else:
            As[rem] = axis
        
        V = As[rem]

        # Norm
        n = len(rem)
        norm = np.linalg.norm(V, axis=1)
        norm = np.resize(norm, n*3).reshape(3, n).transpose()
        
        # Normalize the vectors
        
        As[rem] = V / norm
        
    # nan replaced by default
    inans = np.where(np.isnan(As))[0]
    if len(inans) > 0:
        As[inans] = np.array(default, ftype)
        
    # Returns a single value or an array
    if single:
        return As[0]
    else:
        return As
    
# -----------------------------------------------------------------------------------------------------------------------------
# Dump the content of an array
        
def str_a(array):
    
    if array is None:
        return "Empty array"
    
    array = np.array(array)
    if array.shape == ():
        return f"Scale: {array:7.2f}"
    
    def strarray(array, prof=1):
        shape = array.shape
        if len(shape) > 1:
            sep = " "*prof + "["
            s = ""
            for a in array:
                s += sep + strarray(a, prof+1)
                sep = "\n" + " "*(prof+2)
            s += "]"
                
        elif hasattr(array, '__len__'):
            sep = " "*prof + "["
            s = ""
            for v in array:
                s += sep + f"{v:7.2f}"
                sep = " "
            s += "]"
        else:
            s = f"{array:7.2f}"
            
        return s

    lmax = 10
    if len(array) <= 2*lmax:
        lmax = len(array)
        
    lr = "\n" if len(array.shape) > 1 else ""
    
    s = f"< count: {len(array)}"
    for i in range(lmax):
        s += lr + " " + strarray(array[i])

    if lmax < len(array):
        s += lr + " ... "
        for i in reversed(range(lmax)):
            s += lr + " " + strarray(array[-1-i])
            
    s += "\n>"
    
    return s

# -----------------------------------------------------------------------------------------------------------------------------
# Check utilies in SAFE mode

def check(value, *args):
    if not value:
        raise WrapException("Vector geometry error", *args)
        
def check_vector(v, size, *args):
    check(len(v.shape) >= 1 and len(v.shape)<= 2, *args)
    check((v.shape[-1] > 1) and (size is None or v.shape[-1]==size), *args)
        
def check_matrix(m, *args):
    if len(m.shape) == 2 or len(m.shape) == 3:
        check(m.shape[-1] == m.shape[-2], *args)
    else:
        check(False, *args)
        
def check_quaternion(q, *args):
    check(len(q.shape) >= 1 and len(q.shape)<= 2, "Argument is not a valid quaternion", str_a(q), *args)
    check((q.shape[-1] == 4 ), "Quaternion must a vector of size 4", str_a(q), *args)
    
    
# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Vectors geometry


# -----------------------------------------------------------------------------------------------------------------------------
# Norm of vectors
        
def norm(v):
    """Norm of vectors.
    
    Parameters
    ----------
    v: vector or array of vectors
    
    Return
    ------
    float or array of float
        The vectors norms
    """
    
    vs = np.array(v, ftype)
    return np.linalg.norm(vs, axis=len(vs.shape)-1)

# -----------------------------------------------------------------------------------------------------------------------------
# Noramlizd vectors

def normalized(v):
    """Normalize vectors.
    
    Parameters
    ----------
    v: vector or array of vectors
    
    Returns
    -------
    vector or array of vectors
        The normalized vectors
    """
        
    vs = np.array(v, ftype)
    
    if vs.shape == ():
        return 1.
    elif len(vs.shape) == 1:
        return vs / norm(v)
    elif len(vs.shape) == 2:
        count = vs.shape[0]
        size  = vs.shape[1]
        n = np.resize(norm(v), (size, count)).transpose()
        return vs/n
    
    check(False, f"normalized: {vs.shape} is an invalid array shape", str_a(vs))

# -----------------------------------------------------------------------------------------------------------------------------
# Dot product between arrays of vectors
    
def dot(v, w):
    """Dot product between vectors.
    
    Parameters
    ----------
    v: vector or array of vectors
    w: vector or array of vectors
    
    Returns
    -------
    float or array of floats
    """
    
    vs = np.array(v, ftype)
    ws = np.array(w, ftype)
    
    # One is a scalar
    if vs.shape == () or ws.shape == ():
        return np.dot(vs, ws)
    
    # First is a single vector
    if len(vs.shape) == 1:
        return np.dot(ws, vs)
    
    # Second is a single vector
    if len(ws.shape) == 1:
        return np.dot(vs, ws)
    
    # Two arrays
    v_count = vs.shape[0]
    w_count = ws.shape[0]
    
    # v is array with only one vector
    if v_count == 1:
        return np.dot(ws, vs[0])
        
    # w is array with only one vector
    if w_count == 1:
        return np.dot(vs, ws[0])
    
    # Error
    if v_count != w_count:
        raise WrapException(
            f"Dot error: the two arrays of vectors don't have the same length: {v_count} ≠ {w_count}",
            str_a(vs), str_a(ws))
    
    return np.einsum('...i,...i', vs, ws)

# -----------------------------------------------------------------------------------------------------------------------------
# Cross product between arrays of vectors
    
def cross(v, w):
    """Cross product between vectors.
    
    Parameters
    ----------
    v: vector or array of vectors
    w: vector or array of vectors
    
    Returns
    -------
    vector or array of vectors
    """
    
    vs = np.array(v, ftype)
    ws = np.array(w, ftype)
    
    if (len(vs.shape) == 0) or (len(ws.shape) == 0) or \
            (len(ws.shape) > 2) or (len(ws.shape) > 2) or \
            (vs.shape[-1] != 3) or (ws.shape[-1] != 3):
        raise WrapException(
            f"Cross error: cross product need two vectors or arrays of vectors: {vs.shape} x {vs.shape}",
            str_a(vs), str_a(ws)
            )

    return np.cross(vs, ws)

def v_angle(v, w):
    """Angles between vectors.
    
    Parameters
    ----------
    v: vector or array of vectors
    w: vector or array of vectors
    
    Returns
    -------
    float or array of float
        The angle between the vectors
    """
    
    return np.arccos(dot(normalized(v), normalized(w)))
    

# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Matrix geometry
    
# -----------------------------------------------------------------------------------------------------------------------------
# Rotation matrix
    
def matrix(axis, angle):
    """Create matrices from direction and angle.
    
    Parmeters
    ---------
    axis: array of axis specifications. triplets or letters X, Y, Z or -X, -Y, -Z
        The axis around which to turn
        
    angle: float or array of floats
        The angle to rotate around the axis
        
    Return
    ------
    array (3x3) or array of array(3x3)
    """
    
    return q_to_matrix(quaternion(axis, angle))

# -----------------------------------------------------------------------------------------------------------------------------
# Multiplication between two arrays of matrices

def m_mul(ma, mb):
    """Matrices multiplication.
    
    This function is intented for square matrices only.
    To multiply matrices by vectors, use m_rotate instead
    
    Parameters
    ----------
    ma: array(n x n) or array of array(n x n)
        The first matrix to multiply
        
    mb: array(n x n) or array of array(n x n)
        The second matrix to multiply
        
    Returns
    -------
    array(n x n) or array of array(n x n)
        The multiplication ma.mb
    """
        
    mas = np.array(ma, ftype)
    mbs = np.array(mb, ftype)
    if not(
        ((len(mas.shape) > 1) and (mas.shape[-2] != mas.shape[-1])) and \
        ((len(mbs.shape) > 1) and (mbs.shape[-2] != mbs.shape[-1])) and \
        (len(mas.shape) <= 3) and (len(mbs.shape) <= 3) \
        ):
        raise WrapException(
            f"m_mul errors: arguments must be matrices or arrays of matrices, {mas.shape} x {mbs.shape} is not possible.",
            str_a(mas), str_a(mbs)
            )
        
    return np.matmul(mas, mbs)

# -----------------------------------------------------------------------------------------------------------------------------
# Dot product between amtrices and vectors
    
def m_rotate(m, v):
    """Vector rotation by a matrix.
    
    Parameters
    ----------
    m: array (n x n) or array of array(n x n)
        The rotation matrices
    v: array(n) or array of array (n)
    
    Returns
    -------
    array(n) or array of array(n)
    """
    
    ms = np.array(m, ftype)
    vs = np.array(v, ftype)
    
    if not(
        ((len(ms.shape) > 1) and (ms.shape[-2] != ms.shape[-1])) and \
        ((len(vs.shape) > 0) and (vs.shape[-1] != ms.shape[-1])) and \
        (len(ms.shape) <= 3) and (len(vs.shape) <= 2) \
        ):
        raise WrapException(
            f"m_rotate error: arguments must be matrices and vectors, {ms.shape} . {vs.shape} is not possible.",
            str_a(ms), str_a(vs)
            )
        
    # ---------------------------------------------------------------------------
    # A single vector 
    if len(vs.shape) == 1:
        return np.dot(ms, vs)

    if vs.shape[0] == 1:
        return np.dot(ms, vs[0])
    
    # ---------------------------------------------------------------------------
    # A single matrix
    if len(ms.shape) == 2:
        return np.dot(vs, ms.transpose())
    
    if ms.shape[0] == 1:
        return np.dot(vs, ms[0].transpose())
    
    # ---------------------------------------------------------------------------
    # Several matrices and severals vectors
    if len(ms) != len(vs):
        raise WrapException(
            f"m_rotate error: the length of arrays of matrices and vectors must be equal: {len(ms)} ≠ {len(vs)}",
            str_a(ms), str_a(vs)
            )
        
    return np.einsum('...ij,...j', ms, vs)

# -----------------------------------------------------------------------------------------------------------------------------
# Transpose matrices
    
def transpose(m):
    """Transpose a matrix.
    
    Parameters
    ----------
    m: array(n x n) or array of array(n x n)
        The matrices to transpose
    
    Returns
    -------
    array(n x n) or array of array(n x n)
    """
    
    ms = np.array(m, ftype)
    
    if not(
        ((len(ms.shape) > 1) and (ms.shape[-2] != ms.shape[-1])) and \
        (len(ms.shape) <= 3) \
        ):
        raise WrapException(
            f"transpose error: argument must be a matrix or an array of matrices. Impossible to transpose shape {ms.shape}.",
            str_a(ms)
            )
        
    # A single matrix
    if len(ms.shape) == 2:
        return np.transpose(ms)
    
    # Array of matrices
    return np.transpose(ms, (0, 2, 1))

# -----------------------------------------------------------------------------------------------------------------------------
# Invert matrices
    
def invert(m):
    """Invert a matrix.
    
    Parameters
    ----------
    m: array(n x n) or array of array(n x n)
        The matrices to invert
    
    Returns
    -------
    array(n x n) or array of array(n x n)
    """
    
    ms = np.array(m, ftype)
    
    if not(
        ((len(ms.shape) > 1) and (ms.shape[-2] != ms.shape[-1])) and \
        (len(ms.shape) <= 3) \
        ):
        raise WrapException(
            f"invert error: argument must be a matrix or an array of matrices. Impossible to invert shape {ms.shape}.",
            str_a(ms)
            )
        
    return np.linalg.inv(ms)

# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Quaternion geometry
    
def quaternion(axis, angle):
    """Initialize quaternions from axis and angles
    
    Parameters
    ----------
    axis: array of axis, 3-vectors or letters
        The axis compenent of the quaternions
        
    angle: float or array of floats
        The angle component of the quaternions
        
    Returns
    -------
    array(4) of array or array(4)
        The requested quaternions
    """
    
    axs = get_axis(axis)
    ags = np.array(angle, ftype)
    
    if not ( ( (len(axs.shape) == 1) and (len(axs)==3) ) or ( (len(axs.shape) == 2) and (axs.shape[-1]==3)) ):
        raise WrapException(
            f"quaternion error: argument must be vectors(3) and angles.",
            str_a(axs), str_a(ags)
            )
        
    # ---------------------------------------------------------------------------
    # Only one axis
    if len(axs.shape) == 1:
        
        # Only one angle: a single quaternion
        if len(ags.shape) == 0:
            axs *= np.sin(angle/2)
            return np.insert(axs, 0, np.cos(angle/2))
        
        # Several angles: create an array of axis
        axs = np.resize(axs, (len(ags), 3))
    
    # ---------------------------------------------------------------------------
    # Several axis but on single angle

    elif len(ags.shape) == 0:
        axs *= np.sin(angle/2)
        return np.insert(axs, 0, np.cos(angle/2), axis=1)
    
    # ---------------------------------------------------------------------------
    # Several axis and angles
    
    x_count = axs.shape[0]
    a_count = ags.shape[0]
    count = max(x_count, a_count)
    
    if not( (x_count in [1, count]) and (a_count in [1, count]) ):
        raise WrapException(
            f"quaternion error: The length of the arrays of axis and angles are not the same: {x_count} ≠ {a_count}",
            str_a(axs), str_a(ags)
            )
        
    # Adjust the lengths of the arrays
    if a_count < count:
        ags = np.resize(ags, (count, 1))
        
    if x_count < count:
        axs = np.resize(axs, (count, 3))
        
    # We can proceed
    
    ags /= 2
    axs *= np.resize(np.sin(ags), (3, count)).transpose()
    
    return np.insert(axs, 0, np.cos(ags), axis=1)


# -----------------------------------------------------------------------------------------------------------------------------
# Quaterions to axis and angles
    
def axis_angle(q):
    """Return the axis and angles components of a quaternion.
    
    Parameters
    ----------
    q: array(4) or array of array(4)
        The quaternions
        
    Returns
    -------
    array of array(3), array of float
        The axis and angles of the quaternions
    """
    
    qs = np.array(q, ftype)

    if not ( ( (len(qs.shape) == 1) and (len(qs)==4) ) or ( (len(qs.shape) == 2) and (qs.shape[-1]==4)) ):
        raise WrapException(
            f"axis_angle error: argument must be quaternions, a vector(4) or and array of vectors(4), not shape {qs.shape}",
            str_a(qs)
            )
        
    if len(qs.shape) == 1:
        sn  = norm(qs[1:4])
        if sn < zero:
            axs = np.array((0, 0, 1), ftype)
            ags = 0.
        else:
            axs = qs[1:4] / sn
            ags = 2*np.arccos(qs[0])
    else:
        sn  = norm(qs[:, 1:4])
        zs  = np.where(sn < 0)[0]
        nzs = np.delete(np.arange(len(sn)), zs)
        axs = np.empty((len(sn),3), ftype)
        ags = np.empty(len(sn), ftype)
        if len(zs) > 0:
            axs[zs] = np.array((0, 0, 1), ftype)
            ags[zs] = 0.
        if len(nzs) > 0:
            axs[nzs] = qs[nzs, 1:4] / np.resize(sn[nzs], (3, len(sn))).transpose()
            ags[nzs] = 2*np.arccos(qs[nzs, 0])
    
    return axs, ags

# -----------------------------------------------------------------------------------------------------------------------------
# Utility to dump the content of quaternions
    
def str_q(q):
    
    def sq(ax, ag):
        return f"<({ax[0]:7.2f} {ax[1]:7.2f} {ax[2]:7.2f}) {degrees(ag):6.1f}°>"
    
    axs, ags = axis_angle(q)
    
    if len(axs.shape) == 1:
        return sq(axs, ags)
    
    lmax = 10
    if 2*lmax >= len(axs):
        lmax = len(axs)
        
    s = f"quaternions: {len(axs)}"
    for i in range(lmax):
        s += "\n" + sq(axs[i], ags[i])
    if lmax < len(axs):
        s += "\n..."
        for i in reversed(range(lmax)):
            s += "\n" + sq(axs[-1-i], ags[-1-i])
            
    return s

# -----------------------------------------------------------------------------------------------------------------------------
# Quaternion conjugate

def conjugate(q):
    """Compute the conjugate of a quaternion.
    
    Parameters
    ----------
    q: array(4) or array of array(4)
        The quaternion to conjugate
        
    Returns
    -------
    array(4) or array of array(4)
        The quaternions conjugates
    """
    
    qs = np.array(q, ftype)
    
    if not ( ( (len(qs.shape) == 1) and (len(qs)==4) ) or ( (len(qs.shape) == 2) and (qs.shape[-1]==4)) ):
        raise WrapException(
            f"conjugate error: argument must be quaternions, a vector(4) or and array of vectors(4), not shape {qs.shape}",
            str_a(qs)
            )
        
    if len(qs.shape) == 1:
        qs[1:4] *= -1
    else:
        qs[:, 1:4] *= -1
        
    return qs

# -----------------------------------------------------------------------------------------------------------------------------
# Two quaternions multiplication
# Used to check the numpy algorithms

def _q_mul(qa, qb):
    """Utility: one, one quaternion multiplication."""
    
    a = qa[0]
    b = qa[1]
    c = qa[2]
    d = qa[3]

    e = qb[0]
    f = qb[1]
    g = qb[2]
    h = qb[3]
    
    coeff_1 = a*e - b*f - c*g - d*h
    coeff_i = a*f + b*e + c*h - d*g
    coeff_j = a*g - b*h + c*e + d*f
    coeff_k = a*h + b*g - c*f + d*e

    return np.array([coeff_1, coeff_i, coeff_j, coeff_k])


# -----------------------------------------------------------------------------------------------------------------------------
# Two quaternions multiplication
# numpy algorithm
    
def _np_q_mul(qa, qb):
    """Utility: one, one quaternion multiplication, numpy version."""

    s = qa[0]
    p = qa[1:4]
    t = qb[0]
    q = qb[1:4]
    
    a = s*t - sum(p*q)
    v = s*q + t*p + np.cross(p,q)
    
    return np.array((a, v[0], v[1], v[2]))

# -----------------------------------------------------------------------------------------------------------------------------
# Quaternions multiplications

def q_mul(qa, qb):
    """Quaternions multiplication.
    
    Parameters
    ----------
    qa: array(4) or array of array(4)
        The first quaternion to multiply
    
    qb: array(4) or array of array(4)
        The second quaternion to multiply
        
    Returns
    -------
    array(4) or array of array(4)
        The results of the multiplications: qa x qb
    """
    
    qas = np.array(qa, ftype)
    qbs = np.array(qb, ftype)
    
    if not (
        ( (len(qas.shape) == 1) and (len(qas)==4) ) or ( (len(qas.shape) == 2) and (qas.shape[-1]==4)) and \
        ( (len(qbs.shape) == 1) and (len(qbs)==4) ) or ( (len(qbs.shape) == 2) and (qbs.shape[-1]==4))
        ):
        raise WrapException(
            f"q_mul error: arguments must be quaternions or array of quaternions, impossible to compute shapes {qas.shape} x {qbs.shape}.",
            str_a(qas), str_a(qbs)
            )
    
    a_count = 1 if len(qas.shape) == 1 else qas.shape[0]
    b_count = 1 if len(qbs.shape) == 1 else qbs.shape[0]
    
    count = max(a_count, b_count)
    
    if not((a_count in [1, count]) and (b_count in [1, count])):
        raise WrapException(
            f"q_mul errors: the arrays of quaternions must have the same length: {a_count} ≠ {b_count}",
            str_q(qas), str_q(qbs)
            )
        
    # ---------------------------------------------------------------------------
    # Resize the arrays to the same size
    
    if len(qas.shape) == 1:
        qas = np.resize(qa, (count, 4))
    if len(qbs.shape) == 1:
        qbs = np.resize(qb, (count, 4))
        
    # ---------------------------------------------------------------------------
    # No array at all, let's return a single quaternion, not an array of quaternions
        
    if count == 1:
        q = _q_mul(qas[0], qbs[0])
        if (len(np.array(qa).shape) == 1) and (len(np.array(qb).shape) == 1):
            return q
        else:
            return np.array([q])
    
    # ---------------------------------------------------------------------------
    # a = s*t - sum(p*q)
    w = qas[:, 0] * qbs[:, 0] - np.sum(qas[:, 1:4] * qbs[:, 1:4], axis=1)
    
    # v = s*q + t*p + np.cross(p,q)
    v  = qbs[:, 1:4] * np.resize(qas[:, 0], (3, count)).transpose() + \
         qas[:, 1:4] * np.resize(qbs[:, 0], (3, count)).transpose() + \
         np.cross(qas[:, 1:4], qbs[:, 1:4])
         
    # Insert w before v
    return np.insert(v, 0, w, axis=1)

# -----------------------------------------------------------------------------------------------------------------------------
# Quaternion rotation

def q_rotate(q, v):
    """Rotate a vector with a quaternion.
    
    Parameters
    ----------
    q: array(4) or array of array(4)
        The rotation quaternions
    v: array(3) or array of array(3)
        The vectors to rotate
        
    Returns
    -------
    array(3) or array of array(3)
    """
    
    vs = np.array(v, ftype)
    if not ( ( (len(vs.shape) == 1) and (len(vs)==3) ) or ( (len(vs.shape) == 2) and (vs.shape[-1]==3)) ):
        raise WrapException(
            f"q_rotate error: second argument must be a vector(3) or and array of vectors(3), not shape {vs.shape}",
            str_a(vs)
            )
        
    # Vector --> quaternion by inserting a 0 at position 0
    if len(vs.shape) == 1:
        vs = np.insert(vs, 0, 0)
    else:
        vs = np.insert(vs, 0, 0, axis=1)
        
    # Rotation by quaternion multiplication
    w = q_mul(q, q_mul(vs, conjugate(q)))
    
    # Returns quaternion or array of quaternions
    if len(w.shape)== 1:
        return np.delete(w, 0)
    else:
        return np.delete(w, 0, axis=1)
    
# -----------------------------------------------------------------------------------------------------------------------------
# Quaternion to matrix
    
def q_to_matrix(q):
    """Transform quaternions to matrices.
    
    Parameters
    ----------
    q: array(4) or array of array(4)
        The quaternions to transform
        
    Returns
    -------
    array(3 x 3) or array of array(3 x 3)
    """
    
    qs = np.array(q, ftype)
    
    if not ( ( (len(qs.shape) == 1) and (len(qs)==4) ) or ( (len(qs.shape) == 2) and (qs.shape[-1]==4)) ):
        raise WrapException(
            f"q_to_matrix error: argument must be quaternions, a vector(4) or and array of vectors(4), not shape {qs.shape}",
            str_a(qs)
            )
    # m1
    # +w	 +z -y +x
    # -z +w +x +y
    # +y	 -x +w +z
    # -x -y -z +w
        
    # m2
    # +w	 +z -y -x
    # -z +w +x -y
    # +y	 -x +w -z
    # +x +y +z +w
    
    # ---------------------------------------------------------------------------
    # Only one quaternion
        
    if len(qs.shape)==1:
        m1 = np.stack((
            qs[[0, 3, 2, 1]]*(+1, +1, -1, +1),
            qs[[3, 0, 1, 2]]*(-1, +1, +1, +1),
            qs[[2, 1, 0, 3]]*(+1, -1, +1, +1),
            qs[[1, 2, 3, 0]]*(-1, -1, -1, +1)
            ))
        
        m2 = np.stack((
            qs[[0, 3, 2, 1]]*(+1, +1, -1, -1),
            qs[[3, 0, 1, 2]]*(-1, +1, +1, -1),
            qs[[2, 1, 0, 3]]*(+1, -1, +1, -1),
            qs[[1, 2, 3, 0]]*(+1, +1, +1, +1)
            ))
        
        m = np.matmul(m1, m2).transpose()
        return m[0:3, 0:3]
    
    # ---------------------------------------------------------------------------
    # The same with an array of quaternions
    
    m1 = np.stack((
        qs[:, [0, 3, 2, 1]]*(+1, +1, -1, +1),
        qs[:, [3, 0, 1, 2]]*(-1, +1, +1, +1),
        qs[:, [2, 1, 0, 3]]*(+1, -1, +1, +1),
        qs[:, [1, 2, 3, 0]]*(-1, -1, -1, +1)
        )).transpose((1, 0, 2))
    
    m2 = np.stack((
        qs[:, [0, 3, 2, 1]]*(+1, +1, -1, -1),
        qs[:, [3, 0, 1, 2]]*(-1, +1, +1, -1),
        qs[:, [2, 1, 0, 3]]*(+1, -1, +1, -1),
        qs[:, [1, 2, 3, 0]]*(+1, +1, +1, +1)
        )).transpose((1, 0, 2))
    
    m = np.matmul(m1, m2).transpose((0, 2, 1))
    return m[:, 0:3, 0:3]

# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Euler

euler_orders = ['XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX']
euler_i = {
        'XYZ': [0, 1, 2],
        'XZY': [0, 2, 1],
        'YXZ': [1, 0, 2],
        'YZX': [1, 2, 0],
        'ZXY': [2, 0, 1],
        'ZYX': [2, 1, 0],        
    }

# -----------------------------------------------------------------------------------------------------------------------------
# An euler triplets to string
    
def str_e(euler):
    return f"[{degrees(euler[0]):5.1f} {degrees(euler[1]):5.1f} {degrees(euler[2]):5.1f}]"
    
# -----------------------------------------------------------------------------------------------------------------------------
# Convert euler to a rotation matrix

def e_to_mat(e, order='XYZ'):
    """Transform euler triplets to matrices
    
    Parameters
    ----------
    e: array(3) or array or array(3)
        The eulers triplets
    order: str
        A valid order in euler_orders
    
    Returns
    -------
    array(3 x 3) or array of array(3 x 3)
    """
    
    es = np.array(e, ftype)
    
    if not ( ( (len(es.shape) == 1) and (len(es)==3) ) or ( (len(es.shape) == 2) and (es.shape[-1]==3)) ):
        raise WrapException(
            f"e_to_mat error: argument must be euler triplets, a vector(3) or and array of vectors(3), not shape {es.shape}",
            str_a(es)
            )
        
    if not order in euler_orders:
        raise WrapException(f"e_to_mat error: '{order}' is not a valid code for euler order, must be in {euler_orders}")
        
    if len(es.shape) == 1:
        ms = [matrix((1, 0, 0), es[0]),
              matrix((0, 1, 0), es[1]),
              matrix((0, 0, 1), es[2])]
    else:
        ms = [matrix((1, 0, 0), es[:, 0]),
              matrix((0, 1, 0), es[:, 1]),
              matrix((0, 0, 1), es[:, 2])]
        
    i, j, k = euler_i[order]
    return m_mul(ms[k], m_mul(ms[j], ms[i]))

# -----------------------------------------------------------------------------------------------------------------------------
# Convert euler to a quaternion
        
def e_to_quat(e, order='XYZ'):
    """Transform euler triplets to quaternions.
    
    Parameters
    ----------
    e: array(3) or array or array(3)
        The eulers triplets
    order: str
        A valid order in euler_orders
        
    Returns
    -------
    array(4) or array of array(4)
        The quaternions
    """
    
    es = np.array(e, ftype)
    
    if not ( ( (len(es.shape) == 1) and (len(es)==3) ) or ( (len(es.shape) == 2) and (es.shape[-1]==3)) ):
        raise WrapException(
            f"e_to_quat error: argument must be euler triplets, a vector(3) or and array of vectors(3), not shape {es.shape}",
            str_a(es)
            )
        
    if not order in euler_orders:
        raise WrapException(f"e_to_mat error: '{order}' is not a valid code for euler order, must be in {euler_orders}")
        
        
    if len(es.shape) == 1:
        qs = [quaternion((1, 0, 0), es[0]),
              quaternion((0, 1, 0), es[1]),
              quaternion((0, 0, 1), es[2])]
    else:
        qs = [quaternion((1, 0, 0), es[:, 0]),
              quaternion((0, 1, 0), es[:, 1]),
              quaternion((0, 0, 1), es[:, 2])]
        
    i, j, k = euler_i[order]
    return q_mul(qs[k], q_mul(qs[j], qs[i]))

# -----------------------------------------------------------------------------------------------------------------------------
# Convert a matrix to euler
# The conversion depends upon the order

def m_to_euler(m, order='XYZ'):
    """Transform matrices to euler triplets.
    
    Parameters
    ----------
    m: array(3 x 3) or array or array(3 x 3)
        The matrices
    order: str
        A valid order in euler_orders
        
    Returns
    -------
    array(3) or array of array(3)
        The euler triplets
    """
    
    zero = 0.0001
    ms = np.array(m, ftype)
    
    if not(
        ((len(ms.shape) > 1) and (ms.shape[-2] == 3) and (ms.shape[-2] == 3) ) and \
        (len(ms.shape) <= 3) \
        ):
        raise WrapException(
            f"m_to_euler error: argument must be a matrix(3x3) or an array of matrices. Impossible to convert shape {ms.shape}.",
            str_a(ms)
            )
        
    # ---------------------------------------------------------------------------
    # Indices in the array to compute the angles
    #
    # Base computation for XYZ is
    #
    #    if abs(ms[2, 0] + 1) < zero:
    #        phi    = 0.
    #        theta  = pi/2
    #        psy    = np.arctan2(ms[0, 1], ms[0, 2])
    #    elif abs(ms[2, 0] - 1) < zero:
    #        phi    = 0.
    #        theta  = -pi/2
    #        psy    = -np.arctan2(ms[0, 1], ms[0, 2])
    #    else:
    #        theta  = -np.arcsin(ms[2, 0])
    #        ctheta = np.cos(theta)
    #        psy    = np.arctan2(ms[2, 1]/ctheta, ms[2, 2]/ctheta)
    #        phi    = np.arctan2(ms[1, 0]/ctheta, ms[0, 0]/ctheta)
    
    transpose = False
        
    one = 1
    if order in ['XYZ', 'ZYX']:
        th_l = 2
        th_c = 0
        
        ps_l1 = 2
        ps_c1 = 1
        ps_l2 = 2
        ps_c2 = 2
        
        ph_l1 = 1
        ph_c1 = 0
        ph_l2 = 0
        ph_c2 = 0
        
        # Coef for sin(theta) = +- 1
        sp_l1 = 0 
        sp_l2 = 0 
        sp_c1 = 1
        sp_c2 = 2
    
        if order == 'ZYX':
            #ms = ms.transpose((0, 2, 1))
            transpose = True
            one = -1
            
        i_the = 1
        i_psy = 0
        i_phi = 2
        
    elif order in ['YZX', 'XZY']:
        th_l = 0
        th_c = 1
        
        ps_l1 = 0
        ps_c1 = 2
        ps_l2 = 0
        ps_c2 = 0
        
        ph_l1 = 2
        ph_c1 = 1
        ph_l2 = 1
        ph_c2 = 1

        # Coef for sin(theta) = +- 1
        sp_l1 = 1 
        sp_l2 = 2 
        sp_c1 = 2
        sp_c2 = 2    
        
        if order == 'XZY':
            #ms = ms.transpose((0, 2, 1))
            transpose = True
            one = -1

            sp_l1 = 2 
            sp_l2 = 1 
            sp_c1 = 0
            sp_c2 = 0    
            
        i_the = 2
        i_psy = 1
        i_phi = 0

    elif order in ['ZXY', 'YXZ']:
        th_l = 1
        th_c = 2
        
        ps_l1 = 0
        ps_c1 = 2
        ps_l2 = 2
        ps_c2 = 2
        
        ph_l1 = 1
        ph_c1 = 0
        ph_l2 = 1
        ph_c2 = 1
        
        # Coef for sin(theta) = +- 1
        sp_l1 = 0 
        sp_l2 = 0 
        sp_c1 = 1
        sp_c2 = 0
        
        if order == 'YXZ':
            #ms = ms.transpose((0, 2, 1))
            transpose = True
            one = -1
            
            sp_l1 = 2 
            sp_l2 = 2 
            sp_c1 = 0
            sp_c2 = 1
            
            
        i_the = 0
        i_psy = 1
        i_phi = 2
        
    else:
        raise WrapException(f"m_to_euler error: '{order}' is not a valid euler order")
        
    # ---------------------------------------------------------------------------
    # A single matrix
    if len(ms.shape) == 2:
        
        if transpose:
            ms = ms.transpose()
        
        # Default algorithm for order = XYZ
        # Just for reference !
        if False:
            if abs(ms[2, 0] + 1) < zero:
                phi    = 0.
                theta  = pi/2
                psy    = np.arctan2(ms[0, 1], ms[0, 2])
            elif abs(ms[2, 0] - 1) < zero:
                phi    = 0.
                theta  = -pi/2
                psy    = -np.arctan2(ms[0, 1], ms[0, 2])
            else:
                theta  = -np.arcsin(ms[2, 0])
                ctheta = np.cos(theta)
                psy    = np.arctan2(ms[2, 1]/ctheta, ms[2, 2]/ctheta)
                phi    = np.arctan2(ms[1, 0]/ctheta, ms[0, 0]/ctheta)
        
            return np.array((psy, theta, phi), ftype)
        else:
            euler = np.zeros(3, ftype)
            if abs(ms[th_l, th_c] + 1) < zero:
                euler[i_phi] = 0.
                euler[i_the] = pi/2 * one
                euler[i_psy] = np.arctan2(one * ms[sp_l1, sp_c1], one * ms[sp_l2, sp_c2])
            elif abs(ms[th_l, th_c] - 1) < zero:
                euler[i_phi] = 0.
                euler[i_the] = -pi/2 * one
                euler[i_psy] = -np.arctan2(one * ms[sp_l1, sp_c1], one * ms[sp_l2, sp_c2])
            else:
                euler[i_the] = -one * np.arcsin(ms[th_l, th_c])
                ctheta       = np.cos(euler[i_the])
                euler[i_psy] = np.arctan2(one * ms[ps_l1, ps_c1]/ctheta, ms[ps_l2, ps_c2]/ctheta)
                euler[i_phi] = np.arctan2(one * ms[ph_l1, ph_c2]/ctheta, ms[ph_l2, ph_c2]/ctheta)
        
            return euler
        
    # ---------------------------------------------------------------------------
    # Several matrices
        
    if transpose:
        ms = ms.transpose((0, 2, 1))
    
    eulers = np.zeros((len(ms), 3), ftype)
    rem    = np.arange(len(ms))
    neg_1  = np.where(np.abs(ms[:, th_l, th_c] + 1) < zero)
    pos_1  = np.where(np.abs(ms[:, th_l, th_c] - 1) < zero)
    
    if len(neg_1) > 0:
        rem = np.delete(rem, neg_1)
        eulers[neg_1, i_phi] = 0
        eulers[neg_1, i_the] = pi/2 * one
        eulers[neg_1, i_psy] = np.arctan2(one * ms[neg_1, sp_l1, sp_c1], one * ms[neg_1, sp_l2, sp_c2])
        
    if len(pos_1) > 0:
        rem = np.delete(rem, pos_1)
        eulers[pos_1, i_phi] = 0
        eulers[pos_1, i_the] = -pi/2 * one
        eulers[pos_1, i_psy] = -np.arctan2(one * ms[pos_1, sp_l1, sp_c1], one * ms[pos_1, sp_l2, sp_c2])
        
    if len(rem) > 0:
        theta  = -one * np.arcsin(ms[rem, th_l, th_c])
        ctheta = np.cos(theta)
        eulers[rem, i_the] = theta
        eulers[rem, i_psy] = np.arctan2(one * ms[rem, ps_l1, ps_c1]/ctheta, ms[rem, ps_l2, ps_c2]/ctheta)
        eulers[rem, i_phi] = np.arctan2(one * ms[rem, ph_l1, ph_c1]/ctheta, ms[rem, ph_l2, ph_c2]/ctheta)
            
        
    return eulers

# -----------------------------------------------------------------------------------------------------------------------------
# Get a quaternion which orient a given direction toward a target direction
# Another direction, the "up", is adjusted upwards, ie along z (the up direction)
#
# - axis           : the vector to orient toward target_axis
# - target_axis    : the new direction toward which axis will point
# - up             : The up direction wich will be in the plane (target_axis, z)
# - up_direction   : An alternative to z in the (target_axis, z)
    

def q_tracker(axis, target, up='Y', up_direction='Z'):
    """Work in progress"""
    
    axs = get_axis(axis)       # Vectors to rotate
    txs = get_axis(target)     # The target direction after rotation
    
    # ---------------------------------------------------------------------------
    # Let's align the array lengths
    # We work on (n, 3)
    
    single_axis   = len(axs.shape) == 1
    single_target = len(txs.shape) == 1
    a_count        = 1 if single_axis   else len(axs)
    t_count        = 1 if single_target else len(txs)
    count = max(a_count, t_count)
    
    if not ( (a_count in [1, count]) and (t_count in [1, count]) ):
        raise WrapException(
            f"q_tracker error: the arrays of axis and targets must have the same size: {a_count} ≠ {t_count}",
            str_a(axs), str_a(txs)
            )
        
    if a_count < count:
        axs = np.resize(axs, (count, 3))
    if t_count < count:
        txs = np.resize(txs, (count, 3))
        
    if len(axs.shape) == 1:
        axs = np.array([axs])
    if len(txs.shape) == 1:
        txs = np.array([txs])
    
    # ---------------------------------------------------------------------------
    # First rotation will be made around a vector perp to  (axs, txs)
    
    vrot = cross(axs, txs)  # Perp vector with norm == sine
    
    # if target is opposite of the axis, need to rotate 180°
    qrot = quaternion(vrot, np.arcsin(norm(vrot)))
    
    #invs = np.where(abs(dot(axs, txs)+1) < zero)[0]
    #if len(invs) > 0:
        #qrot[invs] = quaternion(vrot[invs], pi)
    
    
    # ---------------------------------------------------------------------------
    # This rotation places the up axis in a certain direction
    # An additional rotation around the target is required
    # to put the up axis in the plane (target, up_direction)
    
    upr = q_rotate(qrot, get_axis(up))
    
    # Projection in the plane perpendicular to the target
    upr_p = upr - dot(upr, txs)*txs
    
    # We need the normalized version of this vector
    upr_pn = norm(upr_p)
    
    # Norm can be null (when the up direction is // to the target)
    nzs = np.where(abs(upr_pn) > zero)[0]
    
    if len(nzs) > 0:
        
        # We normalize where it is possible
        upr_p[nzs] /= upr_pn[nzs]
        
        # A vector perpendicular to plane(target up_direction)
        # We compute them only on non null upr_p
        v = cross(get_axis(up_direction), txs)[nzs]
        
        # This vector can null if the target is // to up_direction
        v_n = norm(v)
        v_nzs = np.where(v_n > zero)[0]
        
        # We need to rotate the vectors upr when
        # - it is not // to the target (nzs)
        # - the up direction is not // to the target (v_nz)
        # Note that we take arccos and not arcsin since we need
        # the angle towards the plane, not towards the perp vector.
    
        if len(v_nzs) > 0:
            
            idx = np.arange(len(qrot))[nzs][v_nzs]
            
            v[v_nzs] /= v_n[v_nzs]
            ags = np.arccos(norm(cross(v[v_nzs], upr_p[idx])))
            
            # We can build the additional rotation along the target vector

            qrot[idx] = q_mul( quaternion(txs[idx], ags), qrot[idx] )
        
    # Let's return a single quaternion if singles were passed
    
    if single_axis and single_target:
        return qrot[0]
    else:
        return qrot
    
def test_tracker():
    axis = ['X', 'Y', 'Z', '-X', '-Y', '-Z', (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    for i in range(0):
        axis.append((np.random.random_sample(3)-0.5)*2)
        
    print('-'*100)
    print("Test q_tracker")
    print()
        
    kos   = 0
    total = 0
    for cnt, v, w in itertools.product(itertools.count(), axis, axis):
        total += 1
        
        vx = get_axis(v)
        wx = get_axis(w)
        q  = q_tracker(vx, wx)
        vr = q_rotate(q, vx)
        yr = q_rotate(q, (0, 1, 0))
        check1 = abs(1 - dot(vr, wx)) < zero  # rotated v == w
        check2 = dot(yr, (0, 0, 1))   > -zero # rotated Y positive along Z 
        check3 = abs(cross(yr, (0, 0, 1))[2]) < zero  # plane(Z, rotated Y) perp to plane (X, Y)
        #print(v, w, check1, check2, check3)
        if not(check1 and check2 and check3):
            kos += 1
            print(f"{cnt:3}", '-'*30)
            print("v --> w:", v, '-->', w)
            print("vr     :", vr)
            print(str_q(q))
            
    print()
    print(f"kos: {kos}/{total}")
    print('-'*100)
    print()
        
        
    q = q_tracker('X', (1, 0, 1), up='Y')
    print("Tracker:   ", str_q(q))
    print("X rotation:", q_rotate(q, (1, 0, 0)))
    print("Y rotation:", q_rotate(q, (0, 1, 0)))

print('toto')
test_tracker()

# -----------------------------------------------------------------------------------------------------------------------------
# Get a quaternion which orient a given direction toward a target direction
# Another direction, the "up", is adjusted upwards, ie along z (the up direction)
#
# - axis           : the vector to orient toward target_axis
# - target_axis    : the new direction toward which axis will point
# - up             : The up direction wich will be in the plane (target_axis, z)
# - up_direction   : An alternative to z in the (target_axis, z)

def tracker_quaternion_OLD(axis, target_axis, up='Y', up_direction='Z'):

    ax  = get_axis(axis)
    tax = get_axis(target_axis)
    if tax.length < zero:
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







# -----------------------------------------------------------------------------------------------------------------------------
# Test functions

def test_dot_cross():
    # Test dot and cross functions with different types of arguments
    
    def test(v, w):
        v = np.array(v, ftype)
        w = np.array(w, ftype)
        
        vr = [v] if len(v.shape) == 1 else v
        wr = [w] if len(w.shape) == 1 else w
        
        dtest = dot(v, w)
        ctest = cross(v, w)
        if len(vr) == 1:
            dref = [np.dot(vr[0], ww) for ww in wr]
            cref = [np.cross(vr[0], ww) for ww in wr]
        elif len(wr) == 1:
            dref = [np.dot(vv, wr[0]) for vv in vr]
            cref = [np.cross(vv, wr[0]) for vv in vr]
        else:
            dref = [np.dot(vv, ww) for vv, ww in zip(vr, wr)]
            cref = [np.cross(vv, ww) for vv, ww in zip(vr, wr)]
            
        dot_   = np.linalg.norm(np.array(dtest)  - np.array(dref))
        cross_ = np.linalg.norm((np.array(ctest) - np.array(cref)).reshape(max(len(vr), len(wr))*3))
        
        print(f"dot: {dot_:.5f}, cross: {cross_:.5f}")
    
    count = 30
    vs = ((np.random.random_sample(count*3)-0.5) * 4).reshape(count, 3)
    ws = ((np.random.random_sample(count*3)-0.5) * 4).reshape(count, 3)
    
    test( vs[0],   ws[0])
    test([vs[0]],  ws[0])
    test( vs[0],  [ws[0]])
    test([vs[0]], [ws[0]])

    test(  vs,     ws[0])
    test(  vs[0],  ws)
    test(  vs,    [ws[0]])
    test( [vs[0]], ws)
    
    test(vs, ws)
    
def test_quaternion():
    print("-"*100)
    print("Test quaternion")
    print(quaternion((1, 2, 3), radians(30)))
    print(quaternion((1, 2, 3), (radians(30), radians(40))))
    print(quaternion(((1, 2, 3), (3, 4, 5)), radians(30)))
    print(quaternion(((1, 2, 3), (3, 4, 5)), (radians(30), radians(40))))
    
#test_quaternion()
    








def dev():
    r = [None, None, None]
    r[0] = np.array([[1, 0, 0], [0, 2, -3], [0, 3, 2]])
    r[1] = np.array([[12, 0, 13], [0, 1, 0], [-13, 0, 12]])
    r[2] = np.array([[22, -23, 0], [23, 22, 0], [0, 0, 1]])
    print('-'*30, 'X')
    print(r[0])
    print('-'*30, 'Y')
    print(r[1])
    print('-'*30, 'Z')
    print(r[2])
    
    def pm(order):
        o = _o2i(order)
        m = m_mul(r[o[2]], m_mul(r[o[1]], r[o[0]]))
        print('-'*30, order, o)
        print(m)
        
    #pm('XYZ')
    #pm('YZX')
    #pm('ZXY')
    
    #pm('ZYX')
    #pm('YZX')
    #pm('YXZ')
    

def test_m_to_euler():

    orders = ['XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX']
    
    def comp_algo(eulers):
        tot = 0.
        for order in orders:
            mas = e_to_mat(eulers)
            mbs = [e_to_mat(euler) for euler in eulers]
            for ma, mb in zip(mas, mbs):
                va = ma.reshape(9)
                vb = np.reshape(mb, 9)
                diff = np.linalg.norm(va - vb)
                tot += diff
                
        print('-'*50)
        print(f"Algo comparison: {tot:.5f} on {len(eulers)} eulers\n")
            
    
    print('-'*50)
    print("Test conversion: matrix to euler")
    print()
    
    print("--- Base test")
    for order in orders:
        euler = np.array((radians(30), radians(40), radians(50)))
        Ms = [
            e_to_mat(euler, order)
            ]
        print(Ms)
        print(f"{order} Input> ", euler/pi*180)
        print(f"    Euler>", m_to_euler(Ms, order)/pi*180)
        print()
        
    # Eulers
    eulers = []
    for i in range(3):
        euler = np.zeros(3, ftype)
        euler[i] = pi/2
        eulers.append(euler.copy())

        euler[(i+1)%3] = radians(30)
        eulers.append(euler.copy())

        euler[(i+1)%3] = 0
        euler[(i+2)%3] = radians(40)
        eulers.append(euler.copy())

        euler[(i+1)%3] = radians(50)
        euler[(i+2)%3] = radians(60)
        eulers.append(euler.copy())
        
    comp_algo(eulers)
        
    print("--- Spec tests")
    
    kos   = 0
    total = 0
    for order in orders:
        if order in orders:
            ms     = e_to_mat(eulers, order)
            ebacks = m_to_euler(ms, order)
            for i in range(len(ms)):
                
                total += 1
                
                euler = eulers[i]
                eback = ebacks[i]
                
                M    = e_to_mat(euler, order)
                Mb   = e_to_mat(eback, order)
                d    = (M - Mb).reshape(9)
                diff = norm(d)
                
                if diff > zero:
                    kos += 1
                    
                    print(f"{order} Input> {str_e(euler)}")
                    print(f"out>       {str_e(eback)}")
                    print(f"Diff: {diff:.5f}")
                    print()
                    print("-"*10)
                    print(str_a(M))
                    print()
                    print(str_a(np.transpose(M)))
                    print()
                    print(str_a(Mb))
                    print("-"*10)
                    print()
                    
    print(f"\nkos counts = {kos}/{total}\n")

    
    
    print("--- Results must be zero")
    print()
    
    count = 20
    for order in orders:
    # Base permutations
        Ms = [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 0,-1], [0, 1, 0]],
            [[0, 1, 0], [1, 0, 0], [0, 0,-1]],
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        ]
        
        for i in range(count):
            euler = np.random.randint(0, 360, 3)/180*pi
            Ms.append(e_to_mat(euler, order))
            
        eulers = m_to_euler(Ms, order)

        for i, m, euler in zip(range(len(Ms)), Ms, eulers):
            ms = e_to_mat(euler, order)
    
            ma = np.array(m).reshape(9)
            mb = ms.reshape(9)
        
        
        print(f"{order}> {i:3} {np.linalg.norm(ma-mb):.6f}")
    
#test_m_to_euler()    
    
    
    











        

    
  
def test_q():
    
    qx = quaternion((1, 0, 0), radians(45))
    qy = quaternion((0, 1, 0), radians(45))
    qz = quaternion((0, 0, 1), radians(45))
    
    vi = np.array((1, 0, 0))
    vj = np.array((0, 1, 0))
    vk = np.array((0, 0, 1))
    
    qs = [qx, qy, qz]
    vs = [vi, vj, vk]
    for i, j in itertools.product(range(3), range(3)):
        print(f"q{'xyy'[i]} * v{'ijk'[j]} = {q_rotate(qs[i], vs[j])}")
        m = q_to_matrix(qs[i])
        v = m_rotate(m, vs[j])
        print(f"M --->    {v}")
        print()
        
    print('-'*100)
    print("random")
    print()
        
    for i in range(1):
        qa = quaternion((np.random.random_sample(3)-0.5)*2, np.random.random_sample(1)*2*pi)
        qb = quaternion((np.random.random_sample(3)-0.5)*2, np.random.random_sample(1)*2*pi)
        
        print(str_q(qa))
        print(str_q(qb))
        
        qa = quaternion((0, 0, 1), radians(45))
        qa = quaternion((0, 0, 1), radians(45))
        
        qas = np.resize(qa, (3, 4))
        qbs = np.resize(qb, (3, 4))
        
        v0 = (np.random.random_sample(3)-0.5)*3
        v0 = (1, 0, 0)
        
        vq1 = q_rotate(qa, v0)
        vq2 = q_rotate(qb, vq1)
        
        ma = q_to_matrix(qa)
        mb = q_to_matrix(qb)
        
        vm1 = m_rotate(ma, v0)
        vm2 = m_rotate(mb, vm1)
        
        print(f"q1xv1> q vs m: {norm(vq1-vm1):.5f}, {norm(vq2-vm2):.5f}")
        
        vq1 = q_rotate(qas, v0)
        vq2 = q_rotate(qbs, vq1)
        
        mas = q_to_matrix(qas)
        mbs = q_to_matrix(qbs)
        
        vm1 = m_rotate(mas, v0)
        vm2 = m_rotate(mbs, vm1)
        
        print(f"qsxv1> q vs m: {norm(vq1-vm1)}, {norm(vq2-vm2)}")
        
        v0s = np.resize(v0, (3, 3))
        
        vq1 = q_rotate(qa, v0s)
        vq2 = q_rotate(qb, vq1)
        
        ma = q_to_matrix(qa)
        mb = q_to_matrix(qb)
        
        vm1 = m_rotate(ma, v0s)
        vm2 = m_rotate(mb, vm1)
        
        print(f"q1xvs> q vs m: {norm(vq1-vm1)}, {norm(vq2-vm2)}")
        
        vq1 = q_rotate(qas, v0s)
        vq2 = q_rotate(qbs, vq1)
        
        mas = q_to_matrix(qas)
        mbs = q_to_matrix(qbs)
        
        vm1 = m_rotate(mas, v0s)
        vm2 = m_rotate(mbs, vm1)
        
        print(f"qsxvs> q vs m: {norm(vq1-vm1)}, {norm(vq2-vm2)}")
        
#test_q()
    

def test():
    print('-'*100)
    print("Some test")
    print()
    
    def comp(a, b, msg=""):
        ca = np.array(a)
        cb = np.array(b)
        ca = ca.reshape(ca.size)
        cb = cb.reshape(cb.size)
        diff = np.sum(np.abs(ca-cb))
        
        print(f"{msg:15} : {diff}")
        
        if diff > zero:
            print('-'*20)
            print(msg)
            print(str_a(a))
            print()
            print(str_a(b))
            print('---')
            print("DIFF:", diff)
            print()
    
    
    size  = 3
    count = 10
    v  = np.random.random_sample(size)*3.
    vs = (np.random.random_sample(count*size)*3).reshape(count, size)
    w  = np.random.random_sample(size)*3.
    ws = (np.random.random_sample(count*size)*3).reshape(count, size)
    
    ma  = (np.random.random_sample(size*size)*3).reshape(size, size)
    mb  = (np.random.random_sample(size*size)*3).reshape(size, size)
    mas = (np.random.random_sample(count*size*size)*3).reshape(count, size, size)
    mbs = (np.random.random_sample(count*size*size)*3).reshape(count, size, size)
    
    v_norm  = np.linalg.norm(v)
    vs_norm = np.array([np.linalg.norm(a) for a in vs]) 
    
    comp(norm(v),  v_norm,  "norm(v)")
    comp(norm(vs), vs_norm, "norm(vs)")
    
    comp(normalized(v),  v/v_norm, "|v|")
    comp(normalized(vs), [a/np.linalg.norm(a) for a in vs], "|vs|")
    
    dot_1_1 = np.dot(v, w)
    dot_1_s = [np.dot(v, a) for a in ws]
    dot_s_1 = [np.dot(a, w) for a in vs]
    dot_s_s = [np.dot(a, b) for a, b in zip(vs, ws)]
    
    comp(dot(v,  w ), dot_1_1, "v  dot w")
    comp(dot(v,  ws), dot_1_s, "v  dot ws")
    comp(dot(vs, w ), dot_s_1, "vs dot w")
    comp(dot(vs, ws), dot_s_s, "vs dot ws")
    
    cross_1_1 = np.cross(v, w)
    cross_1_s = [np.cross(v, a) for a in ws]
    cross_s_1 = [np.cross(a, w) for a in vs]
    cross_s_s = [np.cross(a, b) for a, b in zip(vs, ws)]
    
    comp(cross(v,  w ), cross_1_1, "v  cross w")
    comp(cross(v,  ws), cross_1_s, "v  cross ws")
    comp(cross(vs, w ), cross_s_1, "vs cross w")
    comp(cross(vs, ws), cross_s_s, "vs dcrossot ws")
    
    mmul_1_1 = np.matmul(ma, mb)
    mmul_1_s = [np.matmul(ma, m) for m in mbs]
    mmul_s_1 = [np.matmul(m, mb) for m in mas]
    mmul_s_s = [np.matmul(m, n) for m, n in zip(mas, mbs)]
    
    comp(matmul(ma,  mb),  mmul_1_1, "ma  x mb")
    comp(matmul(ma,  mbs), mmul_1_s, "ma  x mbs")
    comp(matmul(mas, mb),  mmul_s_1, "mas x mb")
    comp(matmul(mas, mbs), mmul_s_s, "mas x mbs")
    
    mdot_1_1 = np.dot(ma, v)
    mdot_1_s = [np.matmul(ma, a) for a in vs]
    mdot_s_1 = [np.matmul(a, v) for a in mas]
    mdot_s_s = [np.matmul(a, b) for a, b in zip(mas, vs)]
    
    comp(matdot(ma,  v),  mdot_1_1, "ma  . v")
    comp(matdot(ma,  vs), mdot_1_s, "ma  . vs")
    comp(matdot(mas, v),  mdot_s_1, "mas . v")
    comp(matdot(mas, vs), mdot_s_s, "mas . vs")
    
    comp(transpose(mas), [np.transpose(a) for a in mas], "transpose")
    
    qa  = np.random.random_sample(4)*3.
    qas = (np.random.random_sample(count*4)*3).reshape(count, 4)
    qb  = np.random.random_sample(4)*3.
    qbs = (np.random.random_sample(count*4)*3).reshape(count, 4)
    
    qmul_1_1 = _q_mul(qa, qb)
    qmul_1_s = [_q_mul(qa, a) for a in qbs]
    qmul_s_1 = [_q_mul(a, qb) for a in qas]
    qmul_s_s = [_q_mul(a, b) for a, b in zip(qas, qbs)]
    
    comp(q_mul(qa,  qb),  qmul_1_1, "qa  * qb")
    comp(q_mul(qa,  qbs), qmul_1_s, "qa  * qbs")
    comp(q_mul(qas, qb),  qmul_s_1, "qas * qb")
    comp(q_mul(qas, qbs), qmul_s_s, "qas * qbs")
    
    
#test()
    
    


