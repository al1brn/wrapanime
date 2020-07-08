#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Special meshes
Created on Mon Jun  1 15:52:31 2020

@author: Alain Berard

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

import itertools
import numpy as np

import bpy
import bmesh

from mathutils import Matrix, Vector, Quaternion
from math import cos, sin, radians, pi

from wrapanime.utils.errors import WrapException
import wrapanime.utils.geometry as wgeo
import wrapanime.utils.blender as wbl
from wrapanime.utils.vert_array import VertArray

def clip(v, vmin, vmax):
    """A basic clipper"""
    return min(vmax, max(vmin, v))

def plane_vertex(x, y, z=0.):
    """Interprets 2 values as cartesian coordinates"""
    return [y, x, z]

def cylinder_vertex(z, theta, rho=1.):
    """Interprets 2 values as cylindrical coordinates"""
    return [rho*cos(theta), rho*sin(theta), z]

def torus_vertex(theta_major, theta_minor, radius_minor=0.25):
    """Interprets 2 values as torus coordinates"""
    M = Matrix.Rotation(theta_major, 3, 'Z')
    V = radius_minor*Vector((cos(theta_minor), 0., sin(theta_minor)))
    W = M @ V
    return W + Vector((cos(theta_major), sin(theta_major), 0.))

def cone_vertex(rho, theta, slope=2.):
    """Interprets 2 values as polar coordinates"""
    return [rho*cos(theta), rho*sin(theta), rho*slope]

def sphere_vertex(phi, theta, radius=1.):
    """Interprets 2 values as spherical coordinates"""
    r = radius*cos(phi)
    return [r*cos(theta), r*sin(theta), radius*sin(phi)]

# -----------------------------------------------------------------------------------------------------------------------------
# Topology

class Topology():
    """Build the faces of a parameterized surface based on its topolgy.
    
    There are five possible topology:
    - PLANE      : topology is open on both x and y
    - CYLINDER   : topology is open on x and loop on y
    - TORUS      : topology loops both on x and y
    - CONE       : as CYLINDER plus a pole
    - SPHERE     : as CYLINDER plus two poles
    
    This class offers utilies to iter through the two dimensions.
    
    The 'vertex' method is initialized with one of the xxx_vertex
    based on the topology.
    
    Parameters
    ----------
    topology: str in ['PLANE', 'CYLINDER', 'TORUS', 'CONE', 'SPHERE']
        Topology code
    rings: int
        Number of rings (x dimension)
    segments: int
        Number of segments per rings (y dimension)
    size: float
        Amplitude of open dimension
    """

    TOPOLOGIES = ['PLANE', 'CYLINDER', 'TORUS', 'CONE', 'SPHERE']

    def __init__(self, topology, rings, segments, size=1.):
        self.check_topology(topology)
        self.topology  = topology
        self.rings     = clip(rings,    1, 1000)
        self.segments  = clip(segments, 1, 1000)
        self.size      = size

        self.poles = 0
        if self.topology == 'SPHERE':
            self.poles = 2
        elif self.topology == 'CONE':
            self.poles = 1

        if self.topology == 'PLANE':
            self._vertex = plane_vertex
            self.rings    = max(self.rings,    2)
            self.segments = max(self.segments, 2)
            
        elif self.topology == 'CYLINDER':
            self._vertex = cylinder_vertex
            self.rings    = max(self.rings, 2)

        elif self.topology == 'TORUS':
            self._vertex = torus_vertex
            self.rings    = max(self.rings, 2)

        elif self.topology == 'CONE':
            self._vertex = cone_vertex

        elif self.topology == 'SPHERE':
            self._vertex = sphere_vertex
            
    def vertex(self, x, y, value):
        return self._vertex(x, y, value)

    @classmethod
    def check_topology(Cls, topology):
        """Check that the topology is ok
        
        Parameters
        ----------
        topology: str in ['PLANE', 'CYLINDER', 'TORUS', 'CONE', 'SPHERE']
            The topology to test
            
        Raises
        ------
        WrapException: if topology is incorrect
        """
            
        if not topology in Cls.TOPOLOGIES:
            raise WrapException("Topology error",
                f"The topology '{topology}' is not valid. it must be in {Cls.TOPOLOGIES}"
                )

    @property
    def verts_count(self):
        """Number of vertices
        
        The number of vertices is rings*segments plus the number of poles (0, 1 or 2)
        """
        return self.rings*self.segments + self.poles

    def pole_index(self, pole):
        """Vertex index of the pole.
        
        Parameters
        ----------
        pole: int
            Pole index (1 or 2)
            
        Returns
        -------
        int
            Vertex index of the pole
        """
        return self.rings*self.segments + pole

    def pole_vertex(self, pole):
        """Vertex of the pole.
        
        Parameters
        ----------
        pole: int
            Pole index (1 or 2)
            
        Returns
        -------
        array of 3 floats
            Pole vertex
        """
        
        if self.topology == 'CONE':
            return [0., 0., 2*self.size]
        elif self.topology == 'SPHERE':
            if pole == 0:
                return [0., 0., 1.]
            else:
                return [0., 0., -1.]
            
    def pole_coord(self, pole):
        """Coordinates to compute the pole vertex.
        
        Parameters
        ----------
        pole: int
            Pole index (1 or 2)
            
        Returns
        -------
        couple of Floats
            x, y coordinates to compute the pole
        """
        if self.topology == 'CONE':
            return (0., 0.)
        elif self.topology == 'SPHERE':
            if pole == 0:
                return (pi/2, 0.)
            else:
                return (-pi/2, 0.)
        

    def ij(self, index):
        """Compute the i and j indices from the linear index.
        
        Parameters
        ----------
        index: int
            A valid vertex index in [0, verts_count["
            
        Returns
        -------
        2-tuple
            (i, j) parameters of the indexed vertex
        """
        return divmod(index, self.segments)

    def vert_index(self, i, j):
        """Linear index of (i, j) surface parameters.
        
        Parameters
        ----------
        i: int
            First parameter
        
        j: int
            Second Parameter
            
        Returns
        -------
        int
            The vertex linear index
        """
        return (i % self.rings)*self.segments + (j % self.segments)

    def rings_iter(self):
        """Iterator on the rings.
        
        Depending if the ring topology is closed (TORUS, SPHERE), the loop
        is on [0, 2*pi[ or on [-size/S, size/2]
        
        Returns
        -------
        float iterator
        """
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
            x0 = self.size
            dx = -self.size/self.rings
            return map(lambda i: x0 + i*dx, range(self.rings))
        
        elif self.topology == 'SPHERE':
            dag = pi/(self.rings+1)
            ag0 = pi/2-dag
            return map(lambda i: ag0 - i*dag, range(self.rings))

    def segments_iter(self):
        """Iterator on the segments.
        
        Depending if the segment topology is closed (all but PLANE), the loop
        is on [0, 2*pi[ or on [-size/S, size/2]
        
        Returns
        -------
        float iterator
        """
        
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

    def verts(self, f=None):
        """Computes the vertices of the surface.
        
        Parameters
        ----------
        f: function or None
            Two possible templates
                f(x, y) -> Vector
            or
                f(x y) -> Float
        
        Returns
        -------
        array of vectors
        """
        
        # If the function returns a single float, must compute
        # the vertex
        def vcompute(x, y):
            return self._vertex(x, y, f(x, y))
        
        def vcomp_st(x, y):
            return self._vertex(x, y, self.size)
        
        # Check the size of the function
        if f is None:
            if self.topology in ['TORUS', 'SPHERE']:
                calc = vcomp_st
            else:
                calc = self._vertex
        elif np.array(f(0., 0.)).size == 1:
            calc = vcompute
        else:
            calc = f
            
        # Resulting vertices

        verts = VertArray(self.verts_count)
        for i, (x, y) in zip(itertools.count(0), itertools.product(self.rings_iter(), self.segments_iter())):
            verts[i] = calc(x, y)
            
        for i in range(self.poles):
            x, y = self.pole_coord(i)
            verts[self.pole_index(i)] = calc(x, y)

        return verts

    def faces(self):
        """Computes the faces.
        
        The faces are arrays of vertex indices
        
        Returns
        -------
        array of array of int
        """
        
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
        """Computes the uv mappings.
        
        Returns
        -------
        array of array of 2-vectors
        """
        
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
            #pole = self.pole_index(0)
            for j in range(jmax):
                v = j*dv
                uvs.append([(v+dv/2, 1.), (v, 1.-du), (v+dv, 1.-du)])

        if self.poles > 1:
            #pole = self.pole_index(1)
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

    Store vertices in an array and the geometry faces by their indices.
    Once the geometry is built, the mesh is created with create_object method.

    Parameters
    ----------
    secure : boolean
        True for debugging. Performs advanced checks during the mesh construction.

    Attributes
    ----------

    secure : boolean
        Apply advances checkings when True
    verts : array of Vectors
        Store the actual vertices as Vector
    faces : array of tuples of int
        A face is a tuple of indices into the the verts array
    uvs : array of arrays of couples of floats
        The uv coordinates for each vertex of the face. The face the uv maps is the one
        with the same index. Hence uvs and faces must be the same length. An entry in uvs
        can be None if no map is defined for this face
    o_faces : array of tuples of int
        In secure mode only, store a copy of the faces where the vertices indices are
        ordered. When a new face is added, it is verified if the face was already created.
    """

    def __init__(self, secure=False):

        self.secure = secure
        self.verts  = VertArray() # The vertices
        self.faces  = [] # The faces : tuples of vertices
        self.uvs    = [] # uv mapping. Must be the size of faces

        if self.secure:
            self.o_faces = [] # Ordered faces

    # =============================================================================
    # Add a new vertex
    # =============================================================================

    def vert(self, v):
        """Add a new vertex.

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
            
        if True:
            self.verts.add(v)
        else:
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
            An array of vertices to add

        Returns
        -------
        array of int
            The indices of the newly created vertices
        """

        n = len(self.verts)
        if True:
            self.verts.add(vs)
        else:
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
            The uv map of the face. Can be None if no uv map is defined for this face.

        Raises
        ------
        WrapException
            Insertion is non consistent if secure mode
        """

        # ---------------------------------------------------------------------------
        # Security checks

        if self.secure:
            if len(f) <= 2:
                raise WrapException(
                    "Builder.face ERROR> A face must be made of at least 2 vertices",
                    f"faces: {f}"

                )

            for iv in f:
                if (iv < 0) or (iv >= len(self.verts)):
                    raise WrapException(
                        "Builder.face ERROR> One index in the face is not valid",
                        f"face: {f} --> index {iv} not in [0, {len(self.verts)}["
                    )

            if uv is not None:
                if len(uv) != len(f):
                    raise WrapException(
                        "Builder.face ERROR> The length of the face is not equal to the length of uv",
                        f"face: {f}",
                        f"uv: {uv}"
                        )

            o_f = [iv for iv in f]
            o_f.sort()
            if o_f in self.o_faces:
                raise WrapException(
                    "Builder.face ERROR> The face is already registered",
                    f"face: {f}"
                    )

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
        fs : array of tuples of int
            Array containing the faces to add
        uvs : array of tuple of couple of float
            Array containing the uv map of each face to add. The array must be of the same
            size as fs

        Raises
        ------
        WrapException
            When faces and uvs insertion is not consistent, in secure mode

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
                    print()
                    print()
                    raise WrapException(
                            "Builder.add_faces ERROR> The number of faces must be equal to the number of uvs",
                            f"faces: {len(fs)} --> {fs}",
                            f"uvs: {len(uvs)} --> {uvs}"
                            )

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
    # Surface with topology
    # =============================================================================

    @classmethod
    def FromFunction(cls, f=None, topology='PLANE', rings=10, segments=10, size=1., secure=False):
        """Return a MeshBuilder initialized with a given topology.
        
        Parameters
        ----------
        f: function or None
            Function of template: f(x, y) --> vertex
        topology: str in ['PLANE', 'CYLINDER', 'TORUS', 'CONE', 'SPHERE']
            Topology code
        rings: int
            Number of rings (x dimension)
        segments: int
            Number of segments per rings (y dimension)
        size: float, default 1.
            Used to generate default surface

        Returns
        -------
        MeshBuilder
            With vertices, faces and uvs read in the topology
        """
        
        topo = Topology(topology, rings, segments, size=size)
        
        builder = MeshBuilder(secure=secure)
        builder.add_verts(topo.verts(f))
        builder.add_faces(topo.faces(), topo.uvs())
        
        return builder

    # =============================================================================
    # Merge
    # =============================================================================

    def merge(self, other):
        """Merge the content of another builder within the builder.

        Parameters
        ----------
        other: MeshBuilder
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

        builder = MeshBuilder(secure=self.secure)
        builder.verts = [Vector(v) for v in self.verts]
        if clone_faces:
            builder.faces = [tuple(face) for face in self.faces]
            builder.uvs   = [None if mp is None else [uv for uv in mp] for mp in self.uvs]

        return builder
    
    # *****************************************************************************************************************************
    # Implementation in Blender meshes
    
    # =============================================================================
    # Implement the geometry into a mesh 
    # =============================================================================
    
    def to_mesh(self, mesh, uvs=True):
        """Create the geometry in a Blender mesh.
        
        Parameters
        ----------
        mesh: Blender Mesh
            The mesh into which to create the geometry
        uvs: bool
            Also created the uv mapping if True
            
        Returns
        -------
        Blender mesh
            The mesh parameter
        """
        
        
        # Create the new geometry
        mesh.from_pydata(self.verts.array, [], self.faces)
        
        # OLD algorithm
        if False:

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
        # Only if their exists not None uvs
        none_uvs = self.uvs.count(None)
        if none_uvs < len(self.uvs):

            # Create the uv layer
            uv_layer = mesh.uv_layers.new()

            # Rapid if uv are specified for all faces
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
        
        return mesh
    
    
    # =============================================================================
    # Update an existing object
    # =============================================================================
    
    def update_object(self, obj, remove_doubles=None):
        """Update the mesh of an existing Blender object.
        
        Parameters
        ----------
        obj: Blender object
            Object type must be 'MESH'
        remove_doubles: function or None
            Template is f(object)
            
        Returns
        -------
        Blender object
            The obj parameter
        """
        
        obj = wbl.get_object(obj, otype='MESH')
        
        # Get the mesh
        mesh = obj.data
        
        # Clear geometry
        obj.shape_key_clear()
        mesh.clear_geometry()
        
        # Create the new geometry in the mesh
        self.to_mesh(mesh, uvs=True)
        
        if remove_doubles is not None:
            remove_doubles_object(obj, remove_doubles)
            
        return obj
        

    # =============================================================================
    # Create the object
    # =============================================================================

    def create_object(self, name="Special", remove_doubles=None):
        """Create the blender object.

        Parameters
        ----------
        name: str
            Name of the object to create
        remove_doubles : function or None
            Template is f(object)

        Returns
        -------
        Object
            The created Blender object
        """

        # Create the mesh
        mesh = bpy.data.meshes.new(name=name)
        
        # Create the geometry
        self.to_mesh(mesh, uvs=True)

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
    # Check that an array of vertices has the right length
    # =============================================================================
    
    def check_vertices_length(self, length, caller="unknown"):
        """Check that the number of vertices matches a given length.
        
        Parameters
        ----------
        length: int
            The length to compare the number of vertices to
        caller: str
            Caller reference for error message
            
        Raises
        ------
        WrapException
            If the number of vertices isn't equal to the length parameter
        """    
    
        if len(self.verts) != length:
            raise WrapException(
                    f"MeshBuider.{caller} error: the numbers of vertices don't match",
                    f"Vertices of MeshBuilder: {len(self.verts)}"
                    f"Vertices to transform: {length}",
                    )
    
    # =============================================================================
    # Transform an object
    # =============================================================================
    
    def transform_object(self, obj):
        """Transform a mesh object by updating the vertices coordinates.
        
        The number of vertices must match
        
        Parameters
        ----------
        obj: Blender object
            Object type must be 'MESH'
        """
        
        obj = wbl.get_object(obj, otype='MESH')
        
        # Check that the numbers of vertices match
        self.check_vertices_length(len(obj.data.vertices), caller=f"transform_object('{obj.name}')")
        
        # Go
        #verts = np.array(self.verts).reshape(len(self.verts)*3)
        obj.data.vertices.foreach_set('co', self.verts.linear_array)

    # =============================================================================
    # Create a shape key
    # =============================================================================
    
    def to_shapekey(self, obj, name, step=None):
        """Create a shapekey with the current vertices.
        
        The number of vertices must match.
        
        Parameters
        ----------
        obj: Blender object
            Object type must be 'MESH'
        name: str
            Shapekey name
        step: int or None
            value to add to the shapekey name if not None
        
        """
        
        obj = wbl.get_object(obj, otype='MESH')
        
        # Check that the numbers of vertices match
        self.check_vertices_length(len(obj.data.vertices), caller=f"to_shapekey('{obj.name}')")
        
        # Shapekey
        sk = wbl.get_sk(obj, name, step)
        
        # Go
        #verts = np.array(self.verts).reshape(len(self.verts)*3)
        sk.data.foreach_set('co', self.verts.linear_array)
        
    
    # *****************************************************************************************************************************
    # Draw some basic figures

    # =============================================================================
    # Polygon
    # =============================================================================

    def polygon(self, radius=1, count=6, axis='Z'):
        """Draw a polygon.
        
        Parameters
        ----------
        radius: float
            Polygon radius
        count: int
            Number of edges / vertices
        axis: str of vector
            The axis perpendicular to the polygon plane
            
        Returns
        -------
        array of int
            The indices of the created vertices
        """
        
        dag = 2*pi/count
        if axis == 'X':
            return [(0., radius*cos(i*dag), radius*sin(i*dag)) for i in range(count)]
        elif axis == 'Y':
            return [(radius*sin(i*dag), 0., radius*cos(i*dag)) for i in range(count)]
        elif axis == 'Z':
            return [(radius*cos(i*dag), radius*sin(i*dag), 0.) for i in range(count)]
        else:
            q = wgeo.XY_rotation(axis)
            return [Vector((radius*cos(i*dag), radius*sin(i*dag), 0.)).rotate(q) for i in range(count)]

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
        
        # Vectors from center to vertices
        C  = self.face_center(face)
        Rs = [self.verts[iv]-C for iv in face]
        
        # Max one
        lx = [v.length for v in Rs]
        id_max = lx.index(max(lx))
        V0 = Rs[id_max].normalized()
        
        # Most "perp" one (including length)
        dx = [abs(V0.dot(v)) for v in Rs]
        id_p = dx.index(min(dx))
        V1 = Rs[id_p].normalized()

        perp = V0.cross(11)
        if V1.dot(V0) < 0:
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
        
        npt = np.array(translation)
        if npt.size == 3:
            size = self.verts._array.size
            self.verts._array = np.add(self.verts._array.reshape(size, 3), npt).reshape(size)
        else:
            raise WrapException("MeshBuilder translation error: impossible to translate with vector of size {nps.size}", translation)

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
    # scale
    # =============================================================================

    def scale(self, value):
        """Scale the mesh in all directions.

        Parameters
        ----------
        scale: vector or float
            the scale in x, y, z or single scale to apply in all directions
        """
        
        nps = np.array(value)
        if nps.size == 1:
            self.verts._array *= value
        elif nps.size == 3:
            size = self.verts._array.size
            self.verts._array = np.multiply(self.verts._array.reshape(size, 3), nps).reshape(size)
        else:
            raise WrapException("MeshBuilder scale error: impossible to scale with vector of size {nps.size}", value)


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

            If path is a function, it takes three parameters:
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

