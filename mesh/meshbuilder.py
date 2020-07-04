#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Special meshes
Created on Mon Jun  1 15:52:31 2020

@author: Alain Berard
"""

import itertools

import bpy
import bmesh
from mathutils import Matrix, Vector, Quaternion
from math import cos, sin, radians, degrees, pi, atan2, sqrt, acos, tan
import numpy as np

from wrapanime.utils.errors import WrapException

def clip(v, vmin, vmax):
    return min(vmax, max(vmin, v))

def plane_vertex(x, y, size=1.):
    return [y, x, 0]

def cylinder_vertex(x, y, size=1.):
    return [size*cos(y), size*sin(y), x]

def torus_vertex(x, y, size=1.):
    M = Matrix.Rotation(x, 3, 'Z')
    V = size*0.1*Vector((cos(y), 0., sin(y)))
    W = M @ V
    return W + size*Vector((cos(x), sin(x), 0.))

def cone_vertex(x, y, size=1.):
    r = -x*0.3
    return [r*cos(y), r*sin(y), x]

def sphere_vertex(x, y, size=1.):
    r = size*cos(x)
    return [r*cos(y), r*sin(y), size*sin(x)]


class Topology():

    TOPOLOGIES = ['PLANE', 'CYLINDER', 'TORUS', 'CONE', 'SPHERE']

    def __init__(self, topology, rings, segments, size=1.):
        self.check_topology(topology)
        self.topology  = topology
        self.rings     = clip(rings,    2, 1000)
        self.segments  = clip(segments, 2, 1000)
        self.size      = 1.

        self.poles = 0
        if self.topology == 'SPHERE':
            self.poles = 2
        elif self.topology == 'CONE':
            self.poles = 1

        if self.topology == 'PLANE':
            self.vertex = plane_vertex
        elif self.topology == 'CYLINDER':
            self.vertex = cylinder_vertex
        elif self.topology == 'TORUS':
            self.vertex = torus_vertex
        elif self.topology == 'CONE':
            self.vertex = cone_vertex
        elif self.topology == 'SPHERE':
            self.vertex = sphere_vertex

    @classmethod
    def check_topology(Cls, topology):
        if not topology in Cls.TOPOLOGIES:
            raise WrapException("Topology error",
                "The topology '{}' is not valid. it must be in {}".format(topology, TOPOLOGIES))

    @property
    def verts_count(self):
        return self.rings*self.segments + self.poles

    def pole_index(self, pole):
        return self.rings*self.segments + pole

    def pole_vertex(self, pole):
        if self.topology == 'CONE':
            return [0., 0., 0.]
        elif self.topology == 'SPHERE':
            if pole == 0:
                return [0., 0., self.size]
            else:
                return [0., 0., -self.size]

    def ij(self, index):
        return divmod(index, self.segments)

    def vert_index(self, i, j):
        return (i % self.rings)*self.segments + (j % self.segments)

    def rings_iter(self):
        if self.topology == 'PLANE':
            x0 = self.size/2
            dx = -self.size/(self.rings-1)
            return map(lambda i: x0 + i*dx, range(self.rings))
        elif self.topology == 'CYLINDER':
            x0 = self.size/2
            dx = -self.size/(self.rings-1)
            return map(lambda i: x0 + i*dx, range(self.rings))
        elif self.topology == 'TORUS':
            dag = 2*pi/self.rings
            return map(lambda i: i*dag, range(self.rings))
        elif self.topology == 'CONE':
            dx = -self.size/(self.rings-1)
            return map(lambda i: i*dx, range(self.rings))
        elif self.topology == 'SPHERE':
            dag = pi/(self.rings+1)
            ag0 = pi/2-dag
            return map(lambda i: ag0 - i*dag, range(self.rings))

    def segments_iter(self):
        if self.topology == 'PLANE':
            y0 = -self.size/2
            dy = self.size/(self.segments-1)
            return map(lambda j: y0 + j*dy, range(self.segments))
        elif self.topology == 'CYLINDER':
            dag = 2*pi/self.segments
            return map(lambda j: j*dag, range(self.segments))
        elif self.topology == 'TORUS':
            dag = 2*pi/self.segments
            return map(lambda j: j*dag, range(self.segments))
        elif self.topology == 'CONE':
            dag = 2*pi/self.segments
            return map(lambda j: j*dag, range(self.segments))
        elif self.topology == 'SPHERE':
            dag = 2*pi/self.segments
            return map(lambda j: j*dag, range(self.segments))

    def verts(self):
        verts = np.zeros(self.verts_count*3, np.float).reshape(self.verts_count, 3)
        for i, (x, y) in zip(itertools.count(0), itertools.product(self.rings_iter(), self.segments_iter())):
            verts[i] = self.vertex(x, y, self.size)

        for i in range(self.poles):
            verts[self.pole_index(i)] = self.pole_vertex(i)

        return verts

    def faces(self):
        # Faces without the poles
        faces = []
        imax = self.rings if self.topology in ['TORUS'] else self.rings-1
        jmax = self.segments if self.topology in ['CYLINDER', 'TORUS', 'CONE', 'SPHERE'] else self.segments-1
        for i in range(imax):
            for j in range(jmax):
                faces.append([self.vert_index(i, j), self.vert_index(i+1, j), self.vert_index(i+1, j+1), self.vert_index(i, j+1)])

        # poles
        if self.poles > 0:
            pole = self.pole_index(0)
            for j in range(jmax):
                faces.append([pole, self.vert_index(0, j), self.vert_index(0, j+1)])

        if self.poles > 1:
            pole = self.pole_index(1)
            for j in range(jmax):
                faces.append([pole, self.vert_index(self.rings-1, j), self.vert_index(self.rings-1, j+1)])

        return faces

    def uvs(self):
        # Faces without the poles
        uvs = []
        imax = self.rings if self.topology in ['TORUS'] else self.rings-1
        jmax = self.segments if self.topology in ['CYLINDER', 'TORUS', 'CONE', 'SPHERE'] else self.segments-1
        du = 1./(imax + self.poles)
        dv = 1./jmax
        u0 = du if self.poles > 0 else 0.

        for i in range(imax):
            u = u0+i*du
            for j in range(jmax):
                v = j*dv
                #uvs.append([(u, v), (u+du, v), (u+du, v+dv), (u, v+dv)])
                uvs.append([(v, 1.-u), (v, 1.-u-du), (v+dv, 1.-u-du), (v+dv, 1.-u)])

        # poles
        if self.poles > 0:
            pole = self.pole_index(0)
            for j in range(jmax):
                v = j*dv
                uvs.append([(v+dv/2, 1.), (v, 1.-du), (v+dv, 1.-du)])

        if self.poles > 1:
            pole = self.pole_index(1)
            for j in range(jmax):
                v = j*dv
                uvs.append([(v+dv/2, 0.), (v, du), (v+dv, du)])

        return uvs


# ******************************************************************************************************************************************************
# ******************************************************************************************************************************************************
# Helpers to build meshes
# ******************************************************************************************************************************************************
# ******************************************************************************************************************************************************

# ======================================================================================================================================================
# Remove doubles
# ======================================================================================================================================================

def remove_doubles_object(obj, dist):

    bm = bmesh.new()   # create an empty BMesh
    bm.from_mesh(obj.data)   # fill it in from a Mesh

    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=dist)

    # Finish up, write the bmesh back to the mesh
    bm.to_mesh(obj.data)
    bm.free()  # free and prevent further access

    obj.data.validate()
    obj.data.update()

# ======================================================================================================================================================
# Helper to build meshes
# ======================================================================================================================================================

class MeshBuilder():
    """ Mesh builder helper.

    Store vertices in a array and the geometry faces by their indices.
    One the geometry is built, the mesh is created with create_object method.

    Parameters
    ----------
    secure : boolean
        True for debugging. Performs advanced checkes during the mesh construction.

    Attributes
    ----------

    secure : boolean
        Apply advances checkings when True
    verts : array of Vector
        Store the actual vertices as Vector
    faces : array of tuple of int
        A face is a tuple of indices into the the verts array
    uvs : array of couples of floats
        The uv coordinates for each vertex of the face. The face the uv maps is the one
        with the same index. Hence uvs and faces must be the same length. An entry in uvs
        can be None if no map is defined for this face
    o_faces : array of tuple of int
        In secure mode only, store a copy of the faces where the vertices indices are
        ordered. When a new face is added, it is verified if the face was already created.
    """

    def __init__(self, secure=False):

        self.secure = secure
        self.verts  = [] # The vertices
        self.faces  = [] # The faces : tuples of vertices
        self.uvs    = [] # uv mapping. Must be the size of faces

        if self.secure:
            self.o_faces = [] # Ordered faces

    # =============================================================================
    # Add a new vertex
    # =============================================================================

    def vert(self, v):
        """Add a new vertex

        Parameters
        ----------
        v : vertex tuple, array or Vector
            A new vertex in the mesh

        Returns
        -------
        int
            Index of the newly created vertex
        """

        if self.secure:
            assert len(v) == 3

        self.verts.append(Vector(v))
        return len(self.verts)-1

    # =============================================================================
    # Add a list of vertices
    # =============================================================================

    def add_verts(self, vs):
        """Add a list of new vertices.

        Parameters
        ----------
        vs : array of vector-like
            An array of vertex to add

        Returns
        -------
        array of int
            The indices of the newly created vertices
        """

        n = len(self.verts)
        self.verts += [Vector(v) for v in vs]
        return [n+i for i in range(len(vs))]

    # =============================================================================
    # Add a new face
    # =============================================================================

    def face(self, f, inverse=False, uv=None):
        """Add a new face.

        Parameter
        ---------
        f : tuple of int
            The face to add. In secure mode, the face indices are ordered and it is verified
            if the face was already created. If not, the face is successfully added and the
            'ordered' version is stored in o_faces.
        inverse : boolean
            The face is stored in reverse order if True
        uv : tuple of couple of float
            The uv map of the face. Can be Non if no uv map is defined for this face.

        Raises
        ------
        NameError
            Insertion is non consistent
        """

        # ---------------------------------------------------------------------------
        # Security checks

        if self.secure:
            if len(f) <= 2:
                print("face:", f)
                raise NameError("Builder.face ERROR> A face must be made of at least 2 vertices")

            for iv in f:
                if (iv < 0) or (iv >= len(self.verts)):
                    print("face:", f, " --> index {} not in [0, {}[".format(iv, len(self.verts)))
                    raise NameError("Builder.face ERROR> One index in the face is not valid")

            if uv is not None:
                if len(uv) != len(f):
                    print("Face:", f)
                    print("uv:  ", uv)
                    raise NameError("Builder.face ERROR> The length of the face is not equal to the length of uv")

            o_f = [iv for iv in f]
            o_f.sort()
            if o_f in self.o_faces:
                print("face:", f)
                raise NameError("Builder.face ERROR> The face is already registered")

            self.o_faces.append(o_f)

        # ---------------------------------------------------------------------------
        # Security checks passed

        # Inverse the face if required
        if inverse:
            f1 = tuple(a for a in reversed(f))
        else:
            f1 = tuple(f)

        # Append the face and the uv
        self.faces.append(f1)
        self.uvs.append(uv)

        # Return the index of the newly registered fae
        return len(self.faces)-1

    # =============================================================================
    # Add a list of faces
    # =============================================================================

    def add_faces(self, fs, uvs=None):
        """Add an array of faces.

        In secure mode, the faces are added one per one in a loop calling face method.

        Parameters
        ----------
        fs : array of tuple of int
            Array containing the faces to add
        uvs : array of tuple of couple of float
            Array containing the uv map of each face to add. The array must be of the same
            size as fs

        Raises
        ------
        NameError
            When faces and uvs insertion is not consistent

        Returns
        -------
        array of int
            The array containing the indices of the newly created faces
        """

        # ---------------------------------------------------------------------------
        # Security checks

        if self.secure:
            if uvs is not None:
                if len(fs) != len(uvs):
                    print("faces:", len(fs), "-->", fs)
                    print("uvs:  ", len(uvs),"-->", uvs)
                    raise NameError("Builder.add_faces ERROR> The number of faces must be equal to the number of uvs")

            # Insertion one per one
            faces = []
            for i in range(len(fs)):
                faces.append(self.face(fs[i], uv=uvs[i]))
            return faces

        # ---------------------------------------------------------------------------
        # No security checks

        n = len(self.faces)
        self.faces += fs
        if uvs is None:
            self.uvs += [None for i in range(len(fs))]
        else:
            self.uvs += uvs

        return [n + i for i in range(len(fs))]

    # =============================================================================
    # Merge
    # =============================================================================

    def merge(self, other):
        """Merge the content of another builder within the builder.

        Parameters
        ----------
        other: Builder
            The builder to merge
        """

        n = len(self.verts)
        self.add_verts(other.verts)
        faces = [[n + iv for iv in face] for face in other.faces]
        self.add_faces(faces, other.uvs)

    # =============================================================================
    # Clone the builder
    # =============================================================================

    def clone(self, clone_faces=True):
        """Clone the builder to another builder.

        Parameters
        ----------
        clone_faces: Boolean
            Clone only the vertices if False

        Returns
        -------
        Builder
            The cloned builder
        """

        builder = Builder(secure=self.secure)
        builder.verts = [Vector(v) for v in self.verts]
        if clone_faces:
            builder.faces = [tuple(face) for face in self.faces]
            builder.uvs   = [None if mp is None else [uv for uv in mp] for mp in self.uvs]

        return builder

    # =============================================================================
    # Create the object
    # =============================================================================

    def create_object(self, name="Special", remove_doubles=None):
        """Create the blender object.

        Parameters
        ----------
        name: str
            Name of the object to create
        rmeove_doubles : float
            Apply the remove_doubles method with the value passed as an arg to remove closed vertices in the create mesh.
            No doubles removing if None

        Returns
        -------
        Object
            The created Blender object
        """

        # Create the mesh
        mesh = bpy.data.meshes.new(name=name)

        # Add the vertices
        verts = [x for V in self.verts for x in V]
        mesh.vertices.add(len(self.verts))
        mesh.vertices.foreach_set("co", verts)

        # No edge in this implementation
        #mesh.edges.add(num_edges)
        #mesh.edges.foreach_set("vertices", edges)

        # Loops
        verts_indices = [index for face in self.faces for index in face]
        loop_totals   = [len(face) for face in self.faces]
        loop_starts   = [0 for i in range(len(self.faces))]
        for i in range(1, len(self.faces)):
            loop_starts[i] = loop_starts[i-1] + loop_totals[i-1]

        mesh.loops.add(len(verts_indices))
        mesh.loops.foreach_set("vertex_index", verts_indices)

        # Polygons
        mesh.polygons.add(len(self.faces))
        mesh.polygons.foreach_set("loop_start", loop_starts)
        mesh.polygons.foreach_set("loop_total", loop_totals)

        # Create UV coordinate layer and set values
        none_uvs = self.uvs.count(None)
        if none_uvs < len(self.uvs):

            # Create the uv layer
            uv_layer = mesh.uv_layers.new()

            # Rapid iv uv are specified for all faces
            if none_uvs == 0:
                uvs = [uv_co for uv in self.uvs for uv_co in uv]
                for i, uv in enumerate(uv_layer.data):
                    uv.uv = uvs[i]

            # Slower algorithm when uvs are missing
            for ip in range(len(self.faces)):
                uv = self.uvs[ip]
                if uv is not None:
                    loop_start = mesh.polygons[ip].loop_start
                    for i in range(len(uv)):
                        uv_layer.data[loop_start + i].uv = uv[i]

        """
        # Create vertex color layer and set values
        vcol_lay = mesh.vertex_colors.new()
        for i, col in enumerate(vcol_lay.data):
            col.color[0] = vertex_colors[3*i+0]
            col.color[1] = vertex_colors[3*i+1]
            col.color[2] = vertex_colors[3*i+2]
            col.color[3] = 1.0                     # Alpha?
        """

        # We're done setting up the mesh values, update mesh object and
        # let Blender do some checks on it
        mesh.update()
        mesh.validate()

        # Create Object whose Object Data is our new mesh
        obj = bpy.data.objects.new(name, mesh)

        # Add *Object* to the scene, not the mesh
        scene = bpy.context.scene
        scene.collection.objects.link(obj)

        # Select the new object and make it active
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        if remove_doubles is not None:
            remove_doubles_object(obj, remove_doubles)

        return obj

    # =============================================================================
    # Polygon
    # =============================================================================

    def polygon(self, radius=1, count=6, axis='Z'):
        dag = 2*pi/count
        if axis == 'X':
            verts = self.add_verts([(0., radius*cos(i*dag), radius*sin(i*dag)) for i in range(count)])
        elif axis == 'Y':
            verts = self.add_verts([(radius*sin(i*dag), 0., radius*cos(i*dag)) for i in range(count)])
        else:
            verts = self.add_verts([(radius*cos(i*dag), radius*sin(i*dag), 0.) for i in range(count)])

        return verts

    # =============================================================================
    # Compute the center of a face
    # =============================================================================

    def face_center(self, face):
        """Compute the center of a face.

        Parameters
        ----------
        face : tuple of int
            The face to compyte the center of

        Returns
        -------
        Vector
            Center  of the face
        """

        v = Vector((0., 0., 0.))
        for iv in face:
            v += self.verts[iv]
        return v/len(face)

    # =============================================================================
    # Compute the normal of a face
    # =============================================================================

    def face_normal(self, face):
        """Compute the normal to a face.

        NOT TESTED

        Parameters
        ----------
        face : tuple of int
            The face to compute the normal of. Must contain at least 3 vertices.

        Returns
        -------
        Vector or None
            The normal to the face or None if no normal can be computed
        """

        if len(face) < 3:
            return None

        # Vector from center to vertices
        c = self.face_center(face)
        vx = [self.verts[iv]-c for iv in face]

        # Max one
        lx = [v.length for v in vx]
        id_max = lx.index(max(lx))
        v0 = vx[id_max].normalized()

        # Most "perp" one (including length)
        dx = [abs(v0.dot(v)) for v in vx]
        id_p = dx.index(min(dx))
        v1 = vx[id_p].normalized()

        perp = v0.cross(v1)
        if v1.dot(v0) < 0:
            perp *= -1

        return perp.normalized()

    # =============================================================================
    # Map a face
    # =============================================================================

    def uv_from_face(self, face):
        """NOT TESTED"""

        if len(face) < 3:
            return None

        c = self.verts[0]
        vx = [self.verts[iv]-c for iv in face]

        p = self.face_normal(face)
        vp = [v - v.dot(p)*p for v in vx]

        vi = vp[1].normalized()
        vj = p.cross(vi)

        uvx = np.array([vi.dot(v) for v in vp])
        uvy = np.array([vj.dot(v) for v in vp])

        try:
            uvx /= max(abs(uvx))
        except:
            pass

        try:
            uvy /= max(abs(uvy))
        except:
            pass

        uv = [[uvx[i], uvy[i]] for i in range(len(face))]
        return uv

    # =============================================================================
    # Bounding box
    # =============================================================================

    def bounding_box(self):
        """Bounding box of the mesh.

        Returns
        -------
        tuple of Vectors
            The opposite corners of the bounding box
        """

        v0 = Vector(self.verts[0])
        v1 = Vector(self.verts[0])
        for v in self.verts:
            v0.x = min(v.x, v0.x)
            v0.y = min(v.y, v0.y)
            v0.z = min(v.z, v0.z)

            v1.x = max(v.x, v1.x)
            v1.y = max(v.y, v1.y)
            v1.z = max(v.z, v1.z)
        return v0, v1

    # =============================================================================
    # Translation
    # =============================================================================

    def translate(self, translation):
        """Translate the mesh.

        Parameters
        ----------
        translation : vector-like
            The translation vector
        """

        T = Vector(translation)
        for v in self.verts:
            v += T

    # =============================================================================
    # Rotation
    # =============================================================================

    def rotate(self, axis=(0., 0., 1.), angle=0.):
        """Rotate the mesh around an axis of a given angle.

        Parameters
        ----------
        axis : vector-like
            The axis to turn around
        angle : float
            The angle to turn
        """

        q = Quaternion(Vector(axis).normalized(), angle)
        for v in self.verts:
            v.rotate(q)

    # =============================================================================
    # Map a vertices loop along the x-axis
    # =============================================================================

    def verts_uloc(self, verts, u_bounds=[0., 1.]):
        """u of uv map coordinates for a sequence of vertices.

        The successive vertices are mapped along the u-axis within the intervalle u_bounds.
        The vertices are flattened : the distance (always positive) between two
        successive vertices is used.

        Example : for a regular polygon of 4 edges, the result is:
            u_bounds = [0.0, 1.0] --> [0.0, 0.25,  0.5,  0.75,  1.]
            u_bounds = [0.5, 0.6] --> [0.5, 0.525, 0.55, 0.575, 0.6]

        Parameters
        ----------
        verts : array of int
            Indices of the vertices to manage
        u_bounds : couple of float
            min and max x values for the mapping

        Returns
        -------
        array of float
            The x abscisses for the uv map of the vertices sequences

        """

        n = len(verts)
        if n <= 2:
            return tuple(u_bounds)

        # Vertices copy
        VX = [self.verts[iv] for iv in verts]

        # Length of each side of the face
        # Note that if n == 2, there is no loop
        u_len = np.array([(VX[(i+1)%n]-VX[i]).length for i in range(n)])

        # The total length must match the x amplitude
        L = sum(u_len)
        u_len *= (u_bounds[1]-u_bounds[0]) / L

        # X Location is mapped between uvx[0] and uvx[1]
        u_loc = [u_bounds[0] + sum(u_len[:i]) for i in range(n)]

        # Append the max limit to have n+1 locations
        u_loc.append(u_bounds[1])

        return u_loc

    # =============================================================================
    # Link two loops with faces
    # =============================================================================

    def link_with_faces(self, verts0, verts1, u_bounds=[0., 1.], v_bounds=[0., 1.]):
        """Create faces between two sequences.

        The two sequences must be of the same size.

        Parameters
        ----------
        verts0 : array of int
            The first sequence to link with faces
        verts1 : array of int
            The second sequence to link with faces
        u_bounds : couple of float
            The u bounds for the uv map
        v_bounds : couple of float
            The v bounds for the uv map

        Returns
        array of int
            The indices of the created faces
        """

        # ---------------------------------------------------------------------------
        # Security checks

        if self.secure:
            if len(verts0) != len(verts1):
                print("verts0", verts0)
                print("verts1", verts1)
                raise NameError("Builder.link_with_faces ERROR> The two sequences to link must be the same length")

        # ---------------------------------------------------------------------------
        # u locations

        u_loc0 = self.verts_uloc(verts0, u_bounds=u_bounds)
        u_loc1 = self.verts_uloc(verts1, u_bounds=u_bounds)

        # ---------------------------------------------------------------------------
        # Faces

        n = len(verts0)
        y0 = u_bounds[0]
        y1 = u_bounds[1]
        faces = [[verts0[i], verts0[(i+1)%n], verts1[(i+1)%n], verts1[i]] for i in range(n)]
        uvs   = [[(u_loc0[i], y0), (u_loc0[i+1], y0), (u_loc1[i+1], y1), (u_loc1[i], y1)] for i in range(n)]

        return self.add_faces(faces, uvs)


    # =============================================================================
    # Extrusion
    # =============================================================================

    def extrude(self, verts, amount, steps=1, u_bounds=[0., 1.], v_bounds=[0., 1.], close=False):
        """Extrude vertices of certain amount.

        The extruded faces are uv mmaped along u for the ring and along v
        for the extrusion. u_bounds and v_bounds allow to constraint the mapping
        to a sub area.

        Extrude uses an existing sequence of vertices. Cylinder create all the sequences,
        including the first one.

        Parameters
        ----------
        verts : array of int
            Indices of the vertex to extrude
        amount : vector-like or function(step_index, section_index, vertex) -> vertex
            The vector to use for extrusion.
            If amount is a function, it takes three parameters:
                Parameters
                ----------
                step_index : int
                    The extrusion step
                section_index : int
                    The index in the section (not the index of the vertex)
                vertex : Vector
                    The vertex value

                Returns
                -------
                Vector
                    The extruded vertex

        steps : int
            The extrusion can be made in several steps, ie by extruding several
            vertex on the extrusion path
        u_bounds : tuple of float
            The min and max for x uv mapping
        v_bounds : tuple of float
            Tthe min and max for y uv mapping
        close : boolean
            The end of the extrusion must be linked to the begining

        Returns
        -------
        array of array of int, array of int
            array of the created vertices arraged in edges and array of the created faces.
            In the created array of vertices, the index 0 if for the initial section
        """

        # ---------------------------------------------------------------------------
        # x uv mapping

        u_loc = self.verts_uloc(verts, u_bounds)

        # ---------------------------------------------------------------------------
        # y uv mapping is easier :-)

        dy = v_bounds[1]-v_bounds[0]
        if close:
            dy /= steps+1
        else:
            dy /= steps
        v0 = v_bounds[0]

        # ---------------------------------------------------------------------------
        # Extrude one line of vertices per vertex

        with_function = type(amount).__name__ == "function"

        if with_function:
            lines = [[iv] + self.add_verts([amount(j, sec_i, self.verts[iv]) for j in range(steps)]) for sec_i, iv in enumerate(verts)]
        else:
            V = Vector(amount)/steps
            lines = [[iv] + self.add_verts([self.verts[iv] + (j+1)*V for j in range(steps)]) for iv in verts]

        # ---------------------------------------------------------------------------
        # Create the faces when they ara at least 2 vertices in the list

        n = len(verts)
        if n == 2:
            uvs = [ [[u_bounds[0], v0+dy*i], [u_bounds[1], v0+dy*i], [u_bounds[1], v0+dy*(i+1)], [u_bounds[0], v0+dy*(i+1)]] for i in range(steps)]
            fs  = [ [lines[0][i], lines[1][i], lines[1][i+1], lines[0][i+1]] for i in range(steps)]
            faces = self.add_faces(fs, uvs)

        elif n > 2:
            uvs = [ [[u_loc[j], v0+dy*i], [u_loc[j+1], v0+dy*i], [u_loc[j+1], v0+dy*(i+1)], [u_loc[j], v0+dy*(i+1)]] for i in range(steps) for j in range(n)]
            fs  = [ [lines[j][i], lines[(j+1)%n][i], lines[(j+1)%n][i+1], lines[j][i+1]] for i in range(steps) for j in range(n)]
            faces = self.add_faces(fs, uvs)

        # ---------------------------------------------------------------------------
        # Close the extrusion

        if close:
            self.link_with_faces([lines[i][-1] for i in range(n)], verts, u_bounds=u_bounds, v_bounds=[v_bounds[1]-dy, v_bounds[1]])

        # ---------------------------------------------------------------------------
        # Return lines and faces

        return lines, faces

    # =============================================================================
    # Extrusion
    # =============================================================================

    def cylinder(self, section, path, steps=1, u_bounds=[0., 1.], v_bounds=[0., 1.], close=False):
        """Create a cylinder with a given section.

        The extruded faces are uv mmaped along u for the ring and along v
        for the extrusion. u_bounds and v_bounds allow to constraint the mapping
        to a sub area.

        Extrude uses an existing sequence of vertices. Cylinder create all the sequences,
        including the first one.

        Parameters
        ----------
        section : array of vector-like
            The vertices forming the section of the cylinder
        path : vector-like or function(step_index, section_index, vertex) -> vertex
            If vector-like, it represents to total amount of the extrusion to perform

            If pathis a function, it takes three parameters:
                Parameters
                ----------
                step_index : int
                    The extrusion step
                section_index : int
                    The index in the section (not the index of the vertex)
                vertex : Vector
                    The vertex value

                Returns
                -------
                Vector
                    The extruded vertex

        steps : int
            The extrusion can be made in several steps, ie by extruding several
            vertex on the extrusion path
        u_bounds : tuple of float
            The min and max for x uv mapping
        v_bounds : tuple of float
            Tthe min and max for y uv mapping
        close : boolean
            The end of the extrusion must be linked to the begining

        Returns
        -------
        array of array of int, array of int
            array of the created vertices arraged in edges and array of the created faces.
        """

        n = len(section)

        # ---------------------------------------------------------------------------
        # Extrude one line of vertices per vertex

        with_function = type(path).__name__ == "function"
        if with_function:
            lines = [self.add_verts([path(j, i, section[i]) for j in range(steps)]) for i in range(n)]
        else:
            V = Vector(path)/steps
            lines = [self.add_verts([section[i] + j*V for j in range(steps)]) for i in range(n)]

        # ---------------------------------------------------------------------------
        # x uv mapping

        u_loc = self.verts_uloc([lines[i][0] for i in range(n)], u_bounds)

        # ---------------------------------------------------------------------------
        # y uv mapping is easier :-)

        dy = v_bounds[1]-v_bounds[0]
        if close:
            dy /= steps
        else:
            dy /= (steps-1)
        v0 = v_bounds[0]

        # ---------------------------------------------------------------------------
        # Create the faces when they ara at least 2 vertices in the list

        if n == 2:
            uvs = [ [[u_bounds[0], v0+dy*i], [u_bounds[1], v0+dy*i], [u_bounds[1], v0+dy*(i+1)], [u_bounds[0], v0+dy*(i+1)]] for i in range(steps)]
            fs  = [ [lines[0][i], lines[1][i], lines[1][i+1], lines[0][i+1]] for i in range(steps)]
            faces = self.add_faces(fs, uvs)

        elif n > 2:
            uvs = [ [[u_loc[j], v0+dy*i], [u_loc[j+1], v0+dy*i], [u_loc[j+1], v0+dy*(i+1)], [u_loc[j], v0+dy*(i+1)]] for i in range(steps-1) for j in range(n)]
            fs  = [ [lines[j][i], lines[(j+1)%n][i], lines[(j+1)%n][i+1], lines[j][i+1]] for i in range(steps-1) for j in range(n)]
            faces = self.add_faces(fs, uvs)

        # ---------------------------------------------------------------------------
        # Close the extrusion

        if close:
            self.link_with_faces([lines[i][-1] for i in range(n)], [lines[i][0] for i in range(n)], u_bounds=u_bounds, v_bounds=[v_bounds[1]-dy, v_bounds[1]])

        # ---------------------------------------------------------------------------
        # Return lines and faces

        return lines, faces

    # =============================================================================
    # Twist
    # =============================================================================

    def twist(self, axis, angle):
        """Twist the shape along an axis and with a given angle.

        The twist is made around an axis and is of a certain angle.
        The middle of the mesh is unchanged, twisting is made half of the angle
        on each half of the mesh.

        Parameters
        ----------
        axis : vector-like
            The axis to twist the mesh around
        angle: float
            The angle to rotate
        """

        A = Vector(axis).normalized()

        mn = None
        mx = None
        for v in self.verts:
            x = A.dot(v)
            mn = x if mn is None else min(x, mn)
            mx = x if mx is None else max(x, mx)

        amp = mx-mn
        if amp < 0.0001:
            return

        for i, v in enumerate(self.verts):
            x = A.dot(v)
            ag = -angle/2. + angle*(x-mn)/amp
            q = Quaternion(A, ag)
            self.verts[i].rotate(q)

    # =============================================================================
    # Bend around z axis
    # =============================================================================

    def bendz(self, angle=0.):
        """Bend the shape around the Z axis and towards X axis.

        The length along x is bended such as forming an arc of the same length under the
        given angle. Given the length of the arc and the angle, the radius is computed with arc/angle.
        The center is located on the y-axis at radius distance of the y-middle of the mesh.

        A vertex is projected with the following algorithm:
            - Theta = x / radius
            - (x', y') = rotation of (x, y) around the center

        Parameters
        ----------
        angle : float
            The angle to bend
        """

        if abs(angle) < radians(0.1):
            return

        # ----- Length of the mesh
        b0, b1 = self.bounding_box()
        length = b1.x - b0.x
        if length < 0.001:
            return

        # ---- Bending location
        # The radius is so that the length of the arc under angle is the length of the mesh
        radius = length/angle
        cy = (b0.y + b1.y)/2. + radius

        # Bend loop
        for i, v in enumerate(self.verts):
            rho   = cy - v.y
            theta = v.x/radius
            self.verts[i] = Vector((rho*sin(theta), -rho*cos(theta)+cy, v.z))

    # =============================================================================
    # Surface with topology
    # =============================================================================

    def surface(self, topology, rings=10, segments=10):
        topo = Topology(topology, rings, segments)
        self.add_verts(topo.verts())
        self.add_faces(topo.faces(), topo.uvs())
