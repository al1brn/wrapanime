import itertools
from math import pi, degrees, radians
import numpy as np

#import wrapanime
#from wrapanime.utils.errors import WrapException

# Default ndarray float type
ftype = np.float

# Zero
zero = 1e-6


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
        
        pxs = np.where(axis == 'X')[0]
        pys = np.where(axis == 'Y')[0]
        pzs = np.where(axis == 'Z')[0]
        nxs = np.where(axis == '-X')[0]
        nys = np.where(axis == '-Y')[0]
        nzs = np.where(axis == '-Z')[0]
        
        As[pxs] = [ 1,  0,  0]
        As[pys] = [ 0,  1,  0]
        As[pzs] = [ 0,  0,  1]
        
        As[nxs] = [-1,  0,  0]
        As[nys] = [ 0, -1,  0]
        As[nzs] = [ 0,  0, -1]
        
        rem = np.delete(rem, np.concatenate(pxs, pys, pzs, nxs, nys, nzs))
        
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
        
        # zeros
        zs = np.where(norm < zero)[0]
        if len(zs) > 0:
            norm[zs] = 1.
            V[zs] = (0, 0, 1)
        
        # Normalize the vectors
        norm = np.resize(norm, n*3).reshape(3, n).transpose()
        
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
        
def _str(array, dim=1, vtype='scalar'):
    
    if array is None:
        return "[None array]"
    
    if dim is None:
        return f"{array}"
    
    array = np.array(array)
    
    if array.size == 0:
        return f"[Empty array of shape {np.array(array).shape}]"
    
    def scalar(val):
        if vtype.lower() in ['euler', 'degrees']:
            return f"{degrees(val):6.1f}°"
        else:
            return f"{val:7.2f}"
    
    def vector(vec):
        s = ""
        
        if vtype.lower() in ['quat', 'quaternion']:
            if len(vec) != 4:
                s = "!quat: "
            else:
                ax, ag = axis_angle(vec)
                return f"<{vector(ax)} {degrees(ag):6.1f}°>"
                
        lmax = 5
        if len(vec) <= 2*lmax:
            lmax = len(vec)
            
        for i in range(lmax):
            s += " " + scalar(vec[i])
            
        if len(vec) > lmax:
            s += " ..."
            for i in reversed(range(lmax)):
                s += " " + scalar(vec[-1-i])
            
        return "[" + s[1:] + "]"
    
    def matrix(mat, prof=""):
        s = ""
        sep = "["
        for vec in mat:
            s += sep + vector(vec)
            sep = "\n " + prof
        return s + "]"
            
            
    def arrayof():
        lmax = 10
        if len(array) <= 2*lmax:
            lmax = len(array)
            
        s = ""
        sep = "\n["
        for i in range(lmax):
            if dim == 1:
                sep = "\n[" if i == 0 else "\n "
                s += sep + vector(array[i])
            else:
                sep = "\n[" if i == 0 else "\n "
                s += sep + matrix(array[i], prof=" ")
                
        if len(array) > lmax:
            s += f"\n ... total={len(array)}"
            
            for i in reversed(range(lmax)):
                if dim == 1:
                    s += sep + vector(array[-1-i])
                else:
                    s += sep + matrix(array[-1-i], prof=" ")
        return s + "]"
            
            
    if dim == 0:
        if len(array.shape == 0):
            return scalar(array)
        elif len(array.shape) == 1:
            return vector(array)
        else:   
            return f"<Not an array of scalars>\n{array}"
        
    elif dim == 1:
        if len(array.shape) == 0:
            return f"[Scalar {scalar(array)}, not a vector]"
        elif len(array.shape) == 1:
            return vector(array)
        elif len(array.shape) == 2:
            return arrayof()
        else:
            return f"<Not an array of vectors>\n{array}"
            
    elif dim == 2:
        if len(array.shape) < 2:
            return f"<Not a matrix>\n{array}"
        elif len(array.shape) == 2:
            return matrix(array)
        elif len(array.shape) == 3:
            return arrayof()
        else:
            return f"<Not an array of matrices>\n{array}"
        
    else:
        return f"[array of shape {array.shape} for object of dim {dim}]\n{array}"
    
    
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
    
    raise WrapException(
            f"normalized error: invalid array shape {vs.shape} for vector or array of vectors.",
            _str(vs, 1)
        )

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
            _str(vs, 1), _str(ws, 1)
            )
    
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
            _str(vs, 1), _str(ws, 1)
            )

    return np.cross(vs, ws)

# -----------------------------------------------------------------------------------------------------------------------------
# Angles between vectors

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
    
    return np.arccos(np.maximum(-1, np.minimum(1, dot(normalized(v), normalized(w)))))
    

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
        ((len(mas.shape) > 1) and (mas.shape[-2] == mas.shape[-1])) and \
        ((len(mbs.shape) > 1) and (mbs.shape[-2] == mbs.shape[-1])) and \
        (len(mas.shape) <= 3) and (len(mbs.shape) <= 3) \
        ):
        raise WrapException(
            f"m_mul errors: arguments must be matrices or arrays of matrices, {mas.shape} x {mbs.shape} is not possible.",
            _str(mas, 2), _str(mbs, 2)
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
        ((len(ms.shape) > 1) and (ms.shape[-2] == ms.shape[-1])) and \
        ((len(vs.shape) > 0) and (vs.shape[-1] == ms.shape[-1])) and \
        (len(ms.shape) <= 3) and (len(vs.shape) <= 2) \
        ):
        raise WrapException(
            f"m_rotate error: arguments must be matrices and vectors, {ms.shape} . {vs.shape} is not possible.",
            _str(ms, 2), _str(vs, 1)
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
            _str(ms, 2), _str(vs, 1)
            )
        
    return np.einsum('...ij,...j', ms, vs)

# -----------------------------------------------------------------------------------------------------------------------------
# Transpose matrices
    
def m_transpose(m):
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
        ((len(ms.shape) > 1) and (ms.shape[-2] == ms.shape[-1])) and \
        (len(ms.shape) <= 3) \
        ):
        raise WrapException(
            f"transpose error: argument must be a matrix or an array of matrices. Impossible to transpose shape {ms.shape}.",
            _str(ms, 2)
            )
        
    # A single matrix
    if len(ms.shape) == 2:
        return np.transpose(ms)
    
    # Array of matrices
    return np.transpose(ms, (0, 2, 1))

# -----------------------------------------------------------------------------------------------------------------------------
# Invert matrices
    
def m_invert(m):
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
        ((len(ms.shape) > 1) and (ms.shape[-2] == ms.shape[-1])) and \
        (len(ms.shape) <= 3) \
        ):
        raise WrapException(
            f"invert error: argument must be a matrix or an array of matrices. Impossible to invert shape {ms.shape}.",
            _str(ms, 2)
            )
        
    return np.linalg.inv(ms)

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
    
    ms = np.array(m, ftype)
    
    if not(
        ((len(ms.shape) > 1) and (ms.shape[-2] == 3) and (ms.shape[-2] == 3) ) and \
        (len(ms.shape) <= 3) \
        ):
        raise WrapException(
            f"m_to_euler error: argument must be a matrix(3x3) or an array of matrices. Impossible to convert shape {ms.shape}.",
            _str(ms, 2)
            )
        
    single = len(ms.shape) == 2
    if single:
        ms = np.reshape(ms, (1, 3, 3))
        
    # ---------------------------------------------------------------------------
    # Indices in the array to compute the angles

    if order == 'XYZ':
        
        # cz.cy              | cz.sy.sx - sz.cx   | cz.sy.cx + sz.sx  
        # sz.cy              | sz.sy.sx + cz.cx   | sz.sy.cx - cz.sx  
        # -sy                | cy.sx              | cy.cx        
        
        xyz = [1, 0, 2]
        
        ls0, cs0, sgn = (2, 0, -1)
        ls1, cs1, lc1, cc1 = (2, 1, 2, 2)
        ls2, cs2, lc2, cc2 = (1, 0, 0, 0)
        
        ls3, cs3, lc3, cc3 = (0, 1, 1, 1)
        
    elif order == 'XZY':
        
        # cy.cz              | -cy.sz.cx + sy.sx  | cy.sz.sx + sy.cx  
        # sz                 | cz.cx              | -cz.sx            
        # -sy.cz             | sy.sz.cx + cy.sx   | -sy.sz.sx + cy.cx    
        
        xyz = [1, 2, 0]
        
        ls0, cs0, sgn = (1, 0, +1)
        ls1, cs1, lc1, cc1 = (1, 2, 1, 1)
        ls2, cs2, lc2, cc2 = (2, 0, 0, 0)
        
        ls3, cs3, lc3, cc3 = (0, 2, 2, 2)
        
    elif order == 'YXZ':
        
        # cz.cy - sz.sx.sy   | -sz.cx             | cz.sy + sz.sx.cy  
        # sz.cy + cz.sx.sy   | cz.cx              | sz.sy - cz.sx.cy  
        # -cx.sy             | sx                 | cx.cy
                     
        xyz = [0, 1, 2]
        
        ls0, cs0, sgn = (2, 1, +1)
        ls1, cs1, lc1, cc1 = (2, 0, 2, 2)
        ls2, cs2, lc2, cc2 = (0, 1, 1, 1)
        
        ls3, cs3, lc3, cc3 = (1, 0, 0, 0)
        
    elif order == 'YZX':
        
        # cz.cy              | -sz                | cz.sy             
        # cx.sz.cy + sx.sy   | cx.cz              | cx.sz.sy - sx.cy  
        # sx.sz.cy - cx.sy   | sx.cz              | sx.sz.sy + cx.cy    
                   
        xyz = [2, 1, 0]
        
        ls0, cs0, sgn = (0, 1, -1)
        ls1, cs1, lc1, cc1 = (0, 2, 0, 0)
        ls2, cs2, lc2, cc2 = (2, 1, 1, 1)
        
        ls3, cs3, lc3, cc3 = (1, 2, 2, 2)
        
    elif order == 'ZXY':
        
        # cy.cz + sy.sx.sz   | -cy.sz + sy.sx.cz  | sy.cx             
        # cx.sz              | cx.cz              | -sx               
        # -sy.cz + cy.sx.sz  | sy.sz + cy.sx.cz   | cy.cx  
                              
        xyz = [0, 2, 1]
        
        ls0, cs0, sgn = (1, 2, -1)
        ls1, cs1, lc1, cc1 = (1, 0, 1, 1)
        ls2, cs2, lc2, cc2 = (0, 2, 2, 2)
        
        ls3, cs3, lc3, cc3 = (2, 0, 0, 0)
        
    elif order == 'ZYX':
        
        # cy.cz              | -cy.sz             | sy                
        # cx.sz + sx.sy.cz   | cx.cz - sx.sy.sz   | -sx.cy            
        # sx.sz - cx.sy.cz   | sx.cz + cx.sy.sz   | cx.cy

        xyz = [2, 0, 1]
        
        ls0, cs0, sgn = (0, 2, +1)
        ls1, cs1, lc1, cc1 = (0, 1, 0, 0)
        ls2, cs2, lc2, cc2 = (1, 2, 2, 2)
        
        ls3, cs3, lc3, cc3 = (2, 1, 1, 1)
        
    else:
        raise WrapException(f"m_to_euler error: '{order}' is not a valid euler order")
        
    # ---------------------------------------------------------------------------
    # Compute the euler angles
    
    angles = np.zeros((len(ms), 3), ftype)   # Place holder for the angles in the order of their computation
    
    # Computation depends upoin sin(angle 0) == ±1
    
    #rem    = np.arange(len(ms))                           # sin(angle 0) ≠ ±1
    
    neg_1  = np.where(np.abs(ms[:, ls0, cs0] + 1) < zero)[0] # sin(angle 0) = -1
    pos_1  = np.where(np.abs(ms[:, ls0, cs0] - 1) < zero)[0] # sin(angle 0) = +1
    rem    = np.delete(np.arange(len(ms)), np.concatenate((neg_1, pos_1)))
    
    
    if len(neg_1) > 0:
        angles[neg_1, 0] = -pi/2 * sgn
        angles[neg_1, 1] = 0
        angles[neg_1, 2] = np.arctan2(sgn * ms[neg_1, ls3, cs3], ms[neg_1, lc3, cc3])
        
    if len(pos_1) > 0:
        angles[pos_1, 0] = pi/2 * sgn
        angles[pos_1, 1] = 0
        angles[pos_1, 2] = np.arctan2(sgn * ms[pos_1, ls3, cs3], ms[pos_1, lc3, cc3])
        
    if len(rem) > 0:
        angles[rem, 0] = sgn * np.arcsin(ms[rem, ls0, cs0])
        angles[rem, 1] = np.arctan2(-sgn * ms[rem, ls1, cs1], ms[rem, lc1, cc1])
        angles[rem, 2] = np.arctan2(-sgn * ms[rem, ls2, cs2], ms[rem, lc2, cc2])
        
    # ---------------------------------------------------------------------------
    # Returns the result
    
    if single:
        return angles[0, xyz]
    else:
        return angles[:, xyz]
    


# -----------------------------------------------------------------------------------------------------------------------------
# Conversion matrix to quaternion

def m_to_quat(m):
    order = 'XYZ'
    return e_to_quat(m_to_euler(m, order), order)

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
            _str(axs, 1), _str(ags, 0)
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
            _str(axs, 1), _str(ags, 0)
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
            _str(qs, 1, 'quat')
            )
        
    if len(qs.shape) == 1:
        sn  = norm(qs[1:4])
        if sn < zero:
            axs = np.array((0, 0, 1), ftype)
            ags = 0.
        else:
            axs = qs[1:4] / sn
            ags = 2*np.arccos(np.maximum(-1, np.minimum(1, qs[0])))
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
            ags[nzs] = 2*np.arccos(np.maximum(-1, np.minimum(1, qs[nzs, 0])))
    
    return axs, ags

# -----------------------------------------------------------------------------------------------------------------------------
# Quaternion conjugate

def q_conjugate(q):
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
            _str(qs, 1, 'quat')
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
            _str(qas, 1, 'quat'), _str(qbs, 1, 'quat')
            )
    
    a_count = 1 if len(qas.shape) == 1 else qas.shape[0]
    b_count = 1 if len(qbs.shape) == 1 else qbs.shape[0]
    
    count = max(a_count, b_count)
    
    if not((a_count in [1, count]) and (b_count in [1, count])):
        raise WrapException(
            f"q_mul errors: the arrays of quaternions must have the same length: {a_count} ≠ {b_count}",
            _str(qas, 1, 'quat'), _str(qbs, 1, 'quat')
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
            _str(vs, 1)
            )
        
    # Vector --> quaternion by inserting a 0 at position 0
    if len(vs.shape) == 1:
        vs = np.insert(vs, 0, 0)
    else:
        vs = np.insert(vs, 0, 0, axis=1)
        
    # Rotation by quaternion multiplication
    w = q_mul(q, q_mul(vs, q_conjugate(q)))
    
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
            _str(qs, 1, True)
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
# Conversion quaternion --> euler
    
def q_to_euler(q, order='XYZ'):
    return q_to_euler(q_to_matrix(q), order)

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
# Convert euler to a rotation matrix

def e_to_matrix(e, order='XYZ'):
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
            _str(es, 1, 'euler')
            )

    if not order in euler_orders:
        raise WrapException(f"e_to_mat error: '{order}' is not a valid code for euler order, must be in {euler_orders}")
        
    single = len(es.shape) == 1
    if single:
        es = np.reshape(es, (1, 3))
        
    m = np.zeros((len(es), 3, 3), ftype)
    
    cx = np.cos(es[:, 0])
    sx = np.sin(es[:, 0])
    cy = np.cos(es[:, 1])
    sy = np.sin(es[:, 1])
    cz = np.cos(es[:, 2])
    sz = np.sin(es[:, 2])

    if order == 'XYZ':
        m[:, 0, 0] = cz*cy
        m[:, 0, 1] = cz*sy*sx - sz*cx
        m[:, 0, 2] = cz*sy*cx + sz*sx
        m[:, 1, 0] = sz*cy
        m[:, 1, 1] = sz*sy*sx + cz*cx
        m[:, 1, 2] = sz*sy*cx - cz*sx
        m[:, 2, 0] = -sy
        m[:, 2, 1] = cy*sx
        m[:, 2, 2] = cy*cx

    elif order == 'XZY':
        m[:, 0, 0] = cy*cz
        m[:, 0, 1] = -cy*sz*cx + sy*sx
        m[:, 0, 2] = cy*sz*sx + sy*cx
        m[:, 1, 0] = sz
        m[:, 1, 1] = cz*cx
        m[:, 1, 2] = -cz*sx
        m[:, 2, 0] = -sy*cz
        m[:, 2, 1] = sy*sz*cx + cy*sx
        m[:, 2, 2] = -sy*sz*sx + cy*cx

    elif order == 'YXZ':
        m[:, 0, 0] = cz*cy - sz*sx*sy
        m[:, 0, 1] = -sz*cx
        m[:, 0, 2] = cz*sy + sz*sx*cy
        m[:, 1, 0] = sz*cy + cz*sx*sy
        m[:, 1, 1] = cz*cx
        m[:, 1, 2] = sz*sy - cz*sx*cy
        m[:, 2, 0] = -cx*sy
        m[:, 2, 1] = sx
        m[:, 2, 2] = cx*cy

    elif order == 'YZX':
        m[:, 0, 0] = cz*cy
        m[:, 0, 1] = -sz
        m[:, 0, 2] = cz*sy
        m[:, 1, 0] = cx*sz*cy + sx*sy
        m[:, 1, 1] = cx*cz
        m[:, 1, 2] = cx*sz*sy - sx*cy
        m[:, 2, 0] = sx*sz*cy - cx*sy
        m[:, 2, 1] = sx*cz
        m[:, 2, 2] = sx*sz*sy + cx*cy

    elif order == 'ZXY':
        m[:, 0, 0] = cy*cz + sy*sx*sz
        m[:, 0, 1] = -cy*sz + sy*sx*cz
        m[:, 0, 2] = sy*cx
        m[:, 1, 0] = cx*sz
        m[:, 1, 1] = cx*cz
        m[:, 1, 2] = -sx
        m[:, 2, 0] = -sy*cz + cy*sx*sz
        m[:, 2, 1] = sy*sz + cy*sx*cz
        m[:, 2, 2] = cy*cx

    elif order == 'ZYX':
        m[:, 0, 0] = cy*cz
        m[:, 0, 1] = -cy*sz
        m[:, 0, 2] = sy
        m[:, 1, 0] = cx*sz + sx*sy*cz
        m[:, 1, 1] = cx*cz - sx*sy*sz
        m[:, 1, 2] = -sx*cy
        m[:, 2, 0] = sx*sz - cx*sy*cz
        m[:, 2, 1] = sx*cz + cx*sy*sz
        m[:, 2, 2] = cx*cy    
    
    if single:
        return m[0]
    else:
        return m

# -----------------------------------------------------------------------------------------------------------------------------
# Rotate a vector with an euler

def e_rotate(e, v, order='XYZ'):
    return m_rotate(e_to_matrix(e, order), v)

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
            _str(es, 1, 'euler')
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
# Get a quaternion which orient a given axis toward a target direction
# Another contraint is to have the up axis oriented towards the sky
# The sky direction is the normally the Z
#
# - axis   : The axis to rotate toward the target axis
# - target : Thetarget direction for the axis
# - up     : The up direction wich must remain oriented towards the sky
# - sky    : The up direction must be rotated in the plane (target, sky)

def q_tracker(axis, target, up='Y', sky='Z', no_up = False):
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
            _str(axs, 3), _str(txs, 3)
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
    crot = dot(axs, txs)    # Dot products = cosine
    qrot = quaternion(vrot, np.arccos(np.maximum(-1, np.minimum(1, crot))))
    
    # Particular cases = axis and target are aligned
    sames = np.where(abs(crot - 1) < zero)[0]
    opps  = np.where(abs(crot + 1) < zero)[0]    
    
    # Where they are the same, null quaternion
    if len(sames) > 0:
        qrot[sames] = quaternion((0, 0, 1), 0)
        
    # Where they are opposite, we must rotate 180° around a perp vector
    if len(opps) > 0:
        # Let's try a rotation around the X axis
        vx = cross(axs[opps], (1, 0, 0))
        
        # Doesnt' work where the cross product is null
        xzs = np.where(norm(vx) < zero)[0]
        rem = np.arange(len(vx))
        
        # If cross product with X is null, it's where vrot == X
        # we can rotate 180° around Y
        if len(xzs) > 0:
            idx = np.arange(count)[opps][xzs]
            qrot[idx] = quaternion((0, 1, 0), pi)
            rem = np.delete(rem, xzs)
            
        # We can use this vector to rotate 180°
        if len(rem) > 0:
            idx = np.arange(count)[opps][rem]
            qrot[idx] = quaternion(vx, pi)
            
    # No up management
    if no_up:
        if single_axis and single_target:
            return qrot[0]
        else:
            return qrot
            
            
    # ---------------------------------------------------------------------------
    # This rotation places the up axis in a certain direction
    # An additional rotation around the target is required
    # to put the up axis in the plane (target, up_direction)
    
    upr = q_rotate(qrot, get_axis(up))
    
    # Projection in the plane perpendicular to the target
    J = upr - dot(upr, txs)*txs
    
    # We need the normalized version of this vector
    Jn = norm(J)
    
    # Norm can be null (when the up direction is // to the target)
    # In that case, nothing to do
    nzs = np.where(abs(Jn) > zero)[0]
    
    if len(nzs) > 0:
        
        # Normalized version of the vector to rotate
        J[nzs] /= Jn[nzs]
        
        # Target axis and J are two perpendicular normal vectors
        # They are considered to form the two first vector of a base
        # I = txs
        # J = normalized projection of up perpendicular to I
        # We want to rotate the J vector around I to align it along the sky axis
        
        # Let's compute K
        K = cross(txs[nzs], J[nzs])
        
        # We are interested by the components of the sky vector on J and K
        sks = get_axis(sky)
        q2  = quaternion(txs[nzs], np.arctan2(dot(sks, K), dot(sks, J[nzs])))
        
        qrot[nzs] = q_mul( q2, qrot[nzs])
        
        
    # Let's return a single quaternion if singles were passed
    
    if single_axis and single_target:
        return qrot[0]
    else:
        return qrot
    
    
    
def test_tracker():
    axis = ['X', 'Y', 'Z', '-X', '-Y', '-Z', (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    
    for i in range(10):
        axis.append((np.random.random_sample(3)-0.5)*2)
        
    print('-'*100)
    print("Test q_tracker")
    print()
        
    kos   = 0
    total = 0
    for v, w in itertools.product(axis, axis):
        if total == 0 or True:
            vx = get_axis(v)
            wx = get_axis(w)
            q  = q_tracker(vx, wx)
            vr = q_rotate(q, vx)
            yr = q_rotate(q, (0, 1, 0))
            check1 = abs(1 - dot(normalized(vr), normalized(wx))) < zero  # rotated v == w
            check2 = dot(yr, (0, 0, 1))   > -zero # rotated Y positive along Z 
            check3 = abs(cross(yr, (0, 0, 1))[2]) < zero  # plane(Z, rotated Y) perp to plane (X, Y)
            
            if not check2:
                qt = q_tracker(vx, wx, no_up=True)
                tests = 360
                dag = radians(360/tests)
                uzs = []
                for i in range(tests):
                    q2 = quaternion(wx, i*dag)
                    y2 = q_rotate(q_mul(q2, qt), (0, 1, 0))
                    uz = dot(y2, (0, 0, 1))
                    uzs.append(uz)
                    mn = min(np.array(uzs))
                    mx = max(np.array(uzs))
                    if mx <= 0.:
                        check2 = True
            
            if not(check1 and check2 and check3):
                kos += 1
                print(f"{total:3}", '-'*30)
                print("v --> w:", _str(v), '-->', w)
                print("vr     :", _str(vr), check1)
                print("yr     :", _str(yr), check2, check3)
                if not check2:
                    print("uzs    :", mn, mx)
                #print(str_q(q))
        
        total += 1

    print()
    print(f"kos: {kos}/{total}")
    print('-'*100)
    print()
        

#test_tracker()

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
            mas = e_to_matrix(eulers)
            mbs = [e_to_matrix(euler) for euler in eulers]
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
            e_to_matrix(euler, order)
            ]
        print(_str(Ms, 2))
        print(f"{order} Input>", _str(euler, 1, 'degrees'))
        print(f"    Euler>", _str(m_to_euler(Ms, order)[0], 1, 'degrees'))
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
    
    count = 30
    for i in range(count):
        eulers.append(np.random.randint(0, 360, size=3)/180*pi)
        
    print("\n--- Eulers for test")
    print(_str(eulers, 1, 'degrees'))
        
    print("\n--- Spec tests")
    
    kos   = 0
    total = 0
    for order in orders:
        if order in orders:
            ms     = e_to_matrix(eulers, order)
            ebacks = m_to_euler(ms, order)
            for i in range(len(ms)):
                
                total += 1
                
                euler = eulers[i]
                eback = ebacks[i]
                
                M    = e_to_matrix(euler, order)
                Mb   = e_to_matrix(eback, order)
                d    = (M - Mb).reshape(9)
                diff = norm(d)
                
                if diff > zero:
                    kos += 1
                    
                    print(f"{order} Input> {_str(euler, 1, 'degrees')}")
                    print(f"out>       {_str(eback, 1, 'degrees')}")
                    print(f"Diff: {diff:.5f}")
                    print()
                    print("-"*10)
                    print(_str(M, 2))
                    print()
                    print(_str(np.transpose(M), 2))
                    print()
                    print(_str(Mb, 2))
                    print("-"*10)
                    print()
                    
    print(f"\nkos counts = {kos}/{total}\n")

    
    
    print("--- Results must be zero")
    print()
    
    count = 100
    kos   = 0
    total = 0
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
            Ms.append(e_to_matrix(euler, order))
            
        Ms.append(
                [[-0.73, -0.58, -0.36],
                 [ 0.68, -0.63, -0.37],
                 [-0.02, -0.51,  0.86]]                
                )
            
        eulers = m_to_euler(Ms, order)

        for i, m, euler in zip(range(len(Ms)), Ms, eulers):
            ms = e_to_matrix(euler, order)
    
            ma = np.array(m).reshape(9)
            mb = ms.reshape(9)
            
            df = np.linalg.norm(ma-mb)
            if df > 0.0001:
                print('-'*30)
                print("m init", _str(m))
                print("m -> e", _str(euler, 1, 'degrees'), order, "diff=", df)
                print("e -> m", _str(ms))
                kos += 1
            total += 1
            
    print(f"\nkos = {kos}/{total}")
        
    
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
    
    
# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# TESTS    

track = False

def compare(rgeo, rref, dim=1, vtype='scalar', message="Compare"):
    ra = np.array(rgeo)
    rr = np.array(rref)
    if ra.shape != rr.shape:
        return f"{message}: Arrays are not of the same shape: {ra.shape} ≠ {rr.shape}\n"

    d = ra.reshape(ra.size) - rr.reshape(rr.size)
    diff = np.linalg.norm(d)
    if diff > 0.0001:
        return f"{message}: Arrays are different:\n{_str(ra, dim, vtype)}\n\n{_str(rr, dim, vtype)}\n"
    
    return ""

# -----------------------------------------------------------------------------------------------------------------------------
# Perform exhaustive test of a function with the various ways to call

def test_func1(prm, geo_f, ref_f, dim, vtype, message):
    
    if track:
        print(f"--- {message} {'-'*30}")
        
    def ptrack(step):
        if track:
            print(" >", step)
        return step
    
    mess = ""
    
    # single
    step  = ptrack("(1)")
    mess += compare(
            geo_f(prm[0]), 
            ref_f(prm[0]), 
            dim, vtype, message + step)
    
    # [single]
    step  = ptrack("([1])")
    mess += compare(
            geo_f([prm[0]]), 
            [ref_f(prm[0])],
            dim, vtype, message + step)
    
    # plural
    step  = ptrack("(n)")
    mess += compare(
             geo_f(prm), 
            [ref_f(p) for p in prm], 
            dim, vtype, message + step)
    
    return mess

# -----------------------------------------------------------------------------------------------------------------------------
# Perform exhaustive test of a function with the various ways to call

def test_func(prm1, prm2, geo_f, ref_f, dim, vtype, message):
    
    if track:
        print(f"--- {message} {'-'*30}")
        
    def ptrack(step):
        if track:
            print(" >", step)
        return step
    
    mess = ""
    
    # single - single
    step  = ptrack("(1, 1)")
    mess += compare(
            geo_f(prm1[0], prm2[0]), 
            ref_f(prm1[0], prm2[0]), 
            dim, vtype, message + step)
    
    # [single] - single
    step  = ptrack("([1], 1)")
    mess += compare(
            geo_f([prm1[0]], prm2[0]), 
            [ref_f( prm1[0], prm2[0])],
            dim, vtype, message + step)
    
    # single - [single]
    step  = ptrack("(1, [1])")
    mess += compare(
             geo_f(prm1[0],[prm2[0]]), 
            [ref_f(prm1[0], prm2[0])],
            dim, vtype, message + step)
    
    # [single] - [single]
    step  = ptrack("([1], [1])")
    mess += compare(
             geo_f([prm1[0]], [prm2[0]]), 
            [ref_f( prm1[0],   prm2[0])], 
            dim, vtype, message + step)
    
    # single - plural
    step  = ptrack("(1, n)")
    mess += compare(
             geo_f(prm1[0], prm2), 
            [ref_f(prm1[0], p2) for p2 in prm2], 
            dim, vtype, message + step)
    
    # plural - single
    step  = ptrack("(n, 1)")
    mess += compare(
             geo_f(prm1, prm2[0]), 
            [ref_f(p1, prm2[0]) for p1 in prm1], 
            dim, vtype, message + step)
    
    # plural - plural
    step  = ptrack( "(n, n)")
    mess += compare(
             geo_f(prm1, prm2), 
            [ref_f(p1, p2) for p1, p2 in zip(prm1, prm2)], 
            dim, vtype, message + step)
    
    return mess
    

# -----------------------------------------------------------------------------------------------------------------------------
# Vectors for tests
    
def test_vectors(count=10, size=3):
    if size == 3:
        v = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1),
             (1, 1, 0), (1, 0, 1), (0, 1, 1), (-1, -1, 0), (-1, 0, -1), (0, -1, 1),
             (-1, 1, 0), (-1, 0, 1), (0, -1, 1), (1, -1, 0), (1, 0, -1), (0, 1, 1),
             (1, 1, 1), (1, -1, 1), (1, 1, -1), (1, -1, -1),
             (-1, 1, 1), (-1, -1, 1), (-1, 1, -1), (-1, -1, -1)
             ]
    else:
        v = []
    for i in range(count):
        v.append((np.random.random_sample(size)-0.5)*3)
        
    v = np.array(v)
    idx = np.arange(count)
    np.random.shuffle(idx)
    return v[idx]

# -----------------------------------------------------------------------------------------------------------------------------
# Matrices for tests
    
def test_matrices(count=10, size=3):
    if size == 3:
        m = [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1] ],
            [[1, 0, 0], [0, 0,-1], [0, 1, 0] ],
            [[0, 1, 0], [1, 0, 0], [0, 0,-1] ],
            [[0, 1, 0], [0, 0, 1], [1, 0, 0] ],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0] ],
            [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
            ]
    else:
        m = []
        
    for i in range(count):
        e = np.random.random_sample(3)*2*pi
        rm = e_to_matrix(e)
        m.append(rm)
        
    m = np.array(m)
    idx = np.arange(count)
    np.random.shuffle(idx)
    return m[idx]

# -----------------------------------------------------------------------------------------------------------------------------
# Matrices for tests
    
def test_eulers(count=10):
    e = [(10, 0, 0), (0, 20, 0), (0, 0, 30),
         (10, 20, 0), (10, 0, 30), (0, 20, 30),
         (10, 20, 30),
         (90, 20, 30), (10, 90, 30), (10, 20, 90),
         (-90, 20, 30), (10, -90, 30), (10, 20, -90)
        ]
        
    for i in range(count):
        re = np.random.randint(360, size=3)
        e.append(re)
        
    e = np.array(e, ftype)/360*2*pi
    idx = np.arange(count)
    np.random.shuffle(idx)
    return e[idx]


# -----------------------------------------------------------------------------------------------------------------------------
# Perform exhaustive test of a function with the various ways to call
    
def dot_test():
    v = test_vectors(30)
    w = test_vectors(30)
    return test_func(v, w, dot, lambda a, b: np.dot(a, b), 0, 'scalar', 'dot')
    
def cross_test():
    v = test_vectors(30)
    w = test_vectors(30)
    return test_func(v, w, cross, lambda a, b: np.cross(a, b), 1, 'scalar', 'cross')

def v_angle_test():
    v = test_vectors(30)
    w = test_vectors(30)
    return test_func(v, w, v_angle, lambda a, b: v_angle(a, b), 0, 'scalar', 'v_angle')

def m_mul_test():
    ma = test_matrices(30)
    mb = test_matrices(30)
    return test_func(ma, mb, m_mul, lambda a, b: np.matmul(a, b), 2, 'scalar', 'm_mul')
    
def m_rotate_test():
    m = test_matrices(30)
    v = test_vectors(30)
    return test_func(m, v, m_rotate, lambda a, b: np.dot(a, b), 1, 'scalar', 'm_rotate')

def m_invert_test():
    ma = test_matrices(30)
    return test_func1(ma, lambda a: m_invert(a), lambda a: np.linalg.inv(a), 2, 'scalar', 'm_invert')

def m_to_euler_test(count=30):
    m = test_matrices(count)
    mess = test_func1(m, lambda a: m_to_euler(a), lambda a: m_to_euler(a), 1, 'scalar', 'm_to_euler')
    
    # test back to matrices
    # Matrices can be different
    
    v = test_vectors(len(m))
    for order in euler_orders:
        e = m_to_euler(m, order)
        v1 = m_rotate(m, v)
        v2 = e_rotate(e, v)
        mess += compare(v1, v2, 1, 'scalar', f"m_to_euler {order}")

        break
    
    if mess == "":
        mess = "m_to_euler: OK"

    return mess


def e_to_matrix_test(count=30):
    e = test_eulers(count)         # A set of euler triplets
    v = test_vectors(count)    # Vectors to test the eulers
    
    mess = ""
    
    
    for order in ['XYZ']: #euler_orders:
        eb = m_to_euler(e_to_matrix(e, order), order) # Back and forth
        
        v1 = e_rotate(e,  v)        # Rotate with initial ones
        v2 = e_rotate(eb, v)        # Rotate with transformed ones
        
        for i, a, b in zip(range(len(v1)), v1, v2):
            se = _str(e[i], 1, 'degrees')
            mess += compare(a, b, dim=1, vtype='scalar', message=f"\n{i:2}> e_to_matrix {order} - {se}")
            
    if mess == "":
        mess = "e_to_matrix: OK"
    

    return mess

def m_to_quat_test(count=30):
    m = test_matrices(count)
    mess = test_func1(m, lambda a: q_to_matrix(m_to_quat(a)), lambda a: a, 2, 'scalar', 'm_to_quat')
    if mess == "":
        mess = "m_to_quat: OK"
        
    return mess


print(m_to_quat_test(30))


