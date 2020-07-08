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

import numpy as np
import itertools

from ..utils.errors import WrapException

# -----------------------------------------------------------------------------------------------------------------------------
# Topology

class Topology():
    """Build the faces and uvs corresponding to a given topology.
    
    There are five possible topology:
    - PLANE      : topology is open on both x and y
    - CYLINDER   : topology is open on x and loop on y
    - TORUS      : topology loops both on x and y
    - CONE       : as CYLINDER plus a pole
    - SPHERE     : as CYLINDER plus two poles
    
    Parameters
    ----------
    topology: str in ['PLANE', 'CYLINDER', 'TORUS', 'CONE', 'SPHERE']
        Topology code
    rings: int
        Number of rings (x dimension)
    segms: int
        Number of segments per ring (y dimension)
    """

    TOPOLOGIES = ['PLANE', 'CYLINDER', 'TORUS', 'CONE', 'SPHERE']

    def __init__(self, topology, rings, segms):
        self.check_topology(topology)
        self.topology  = topology
        self.rings     = min(rings,    1000)
        self.segms     = max(1, min(segms, 1000))
        
        # Number of poles
        self.poles = 0
        if self.topology == 'SPHERE':
            self.poles = 2
        elif self.topology == 'CONE':
            self.poles = 1
            
        # Rings & segms min
        if self.topology == 'PLANE':
            self.rings = max(self.rings,    2)
            self.segms = max(self.segms, 2)
            
        elif self.topology == 'CYLINDER':
            self.rings = max(self.rings, 2)

        elif self.topology == 'TORUS':
            self.rings = max(self.rings, 2)

        elif self.topology == 'CONE':
            self.rings = max(self.rings, 1)

        elif self.topology == 'SPHERE':
            self.rings = max(self.rings, 1)
            

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
    def ring_loops(self):
        return self.topology in ['TORUS']
    
    @property
    def segm_loops(self):
        return self.topology in ['CYLINDER', 'TORUS', 'CONE', 'SPHERE']

    @property
    def verts_count(self):
        """Number of vertices
        
        The number of vertices is rings*segms plus the number of poles (0, 1 or 2)
        """
        return self.rings*self.segms + self.poles

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
        return self.rings*self.segms + pole

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
        return divmod(index, self.segms)

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
        return (i % self.rings)*self.segms + (j % self.segms)
    
    def ring(self, ring_index):
        return np.arange(ring_index*self.segms, self.segms)
    
    def segm(self, segm_index):
        return np.arange(self.rings)*self.segms
    
    def triangles(self, points, top):
        return [[points[i], points[i+1], top] for i in range(len(points)-1)]
        

    def faces(self):
        """Computes the faces.
        
        The faces are arrays of vertex indices
        
        Returns
        -------
        array of array of int
        """
        
        # Faces without the poles
        imax = self.rings if self.ring_loops else self.rings-1
        jmax = self.segms if self.segm_loops else self.segms-1
        
        faces = []
        for i, j in itertools.product(range(imax), range(jmax)):
            faces.append([self.vert_index(i, j), self.vert_index(i+1, j), self.vert_index(i+1, j+1), self.vert_index(i, j+1)])

        # poles
        if self.poles > 0:
            faces += self.triangles(self.ring(0), self.pole_index(0))

        if self.poles > 1:
            faces += self.triangles(self.ring(self.rings-1), self.pole_index(1))

        return faces

    def uvs(self):
        """Computes the uv mappings.
        
        Returns
        -------
        array of array of 2-vectors
        """
        
        imax = self.rings if self.ring_loops else self.rings-1
        jmax = self.segms if self.segm_loops else self.segms-1
        
        du = 1./(imax + self.poles)
        dv = 1./jmax
        u0 = du if self.poles > 0 else 0.
        
        # Grid
        uvs = []
        for i, j in itertools.product(range(imax), range(jmax)):
            u = u0+i*du
            v = j*dv
            uvs.append([(v, 1.-u), (v, 1.-u-du), (v+dv, 1.-u-du), (v+dv, 1.-u)])
            

        # poles
        if self.poles > 0:
            for j in range(jmax):
                v = j*dv
                uvs.append([(v+dv/2, 1.), (v, 1.-du), (v+dv, 1.-du)])

        if self.poles > 1:
            for j in range(jmax):
                v = j*dv
                uvs.append([(v+dv/2, 0.), (v, du), (v+dv, du)])

        return uvs


