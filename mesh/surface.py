#!/usr/bin/env python

"""
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

Created: Jul 8 2020
Author: Alain Bernard
"""

import itertools
from math import sin, cos, pi, degrees
import numpy as np

"""
import sys
sys.path.append("..")

from utils.errors import WrapException

"""

from wrapanime.utils.errors import WrapException

#"""

# -----------------------------------------------------------------------------------------------------------------------------
# Coordinate system transformation

def xyz_xyz(x, y, z):
    return [x, y, z]
        
def sph_xyz(phi, theta, rho):
    rcphi = rho*cos(phi)
    return [rcphi*cos(theta), rcphi*sin(theta), rho*sin(phi)]

def cyl_xyz(z, theta, rho):
    return [rho*cos(theta), rho*sin(theta), z]

def pol_xyz(rho, theta, z):
    return [rho*cos(theta), rho*sin(theta), z]

def tor_xyz(theta_major, theta_minor, rho, radius=1.):
    
    c = cos(theta_major)
    s = sin(theta_major)
    M = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    
    V = rho * np.array([cos(theta_minor), 0., sin(theta_minor)])
    W = M.dot(V)
    
    return W + radius*np.array([c, s, 0.])

# -----------------------------------------------------------------------------------------------------------------------------
# A general surface iterator
# Iterates on two dimensions x and y
    
def surface_iterator(x0, x1, y0, y1, x_count=10, y_count=10, x_loop=False, y_loop=False):
    """An iterator on x an y parameters to compute a surface.
    
    When loop is set on a dimension, the iteration doesn't include
    the final value, otherwise it does.
    
    Parameters
    ----------
        x0: float
            Starting x value
        x1: float
            Endig x value
        y0: float
            Starting y value
        y1: float
            Ending x value
        x_count: int
            Number of vertices in the x dimension
        y_count: int
            Number of vertices in the y dimension
        x_loop: bool
            Last x index is followed by the first one
        y_loop: bool
            Last y index is followed by the first one
    """
    
    if x_loop:
        dx = (x1-x0)/x_count if x_count > 0 else 0.
    else:
        dx = (x1-x0)/(x_count-1) if x_count > 1 else 0.
        
    if y_loop:
        dy = (y1-y0)/y_count if y_count > 0 else 0.
    else:    
        dy = (y1-y0)/(y_count-1) if y_count > 1 else 0.
        
    return itertools.product(
            itertools.accumulate(itertools.chain([x0], itertools.repeat(dx, x_count-1))),
            itertools.accumulate(itertools.chain([y0], itertools.repeat(dy, y_count-1)))
            )
    
# -----------------------------------------------------------------------------------------------------------------------------
# Sphere iterator 
    
pis2 = pi/2.
twpi = 2*pi
    
def sph_iterator(x_count=17, y_count=32):
    return surface_iterator(pis2, -pis2, 0., twpi, x_count=x_count, y_count=y_count, x_loop=False, y_loop=True)

# -----------------------------------------------------------------------------------------------------------------------------
# Cylinder iterator 

def cyl_iterator(z0, z1, x_count=10, y_count=32):
    return surface_iterator(z0, z1, 0., twpi, x_count=x_count, y_count=y_count, x_loop=False, y_loop=True)
    
# -----------------------------------------------------------------------------------------------------------------------------
# Torus iterator

def tor_iterator(x_count=48, y_count=12):
    return surface_iterator(0., twpi, 0., twpi, x_count=x_count, y_count=y_count, x_loop=True, y_loop=True)

# -----------------------------------------------------------------------------------------------------------------------------
# Calc an uv surface

def calc_uv_surface(f, u0, u1, v0, v1, x_count, y_count):
    """Compute a surface parametered with two non dimensional parameters.
    
    Parameters
    ----------
    
    f: function f(u, v) -> vector
        The function called for each couple (u, v)
    u0: float
        Starting u value
    u1: float
        Ending u value
    v0: float
        Starting v value
    v1: float
        Ending v value
    x_count: int
        Number of vertices in the x dimension
    y_count: int
        Number of vertices in the y dimension
            
    Returns
    -------
    array of vertices
        The vertices of the surface
    """
    
    verts = np.empty((x_count*y_count, 3))
    for i, (u, v) in zip(itertools.count(), surface_iterator(u0, u1, v0, v1, x_count, y_count)):
        verts[i] = f(u, v)
    return verts

# -----------------------------------------------------------------------------------------------------------------------------
# Calc an uv surface

def calc_xyz_surface(f, x0, x1, y0, y1, x_count, y_count):
    """Compute a surface parametered by x, y coordinates.
    
    Parameters
    ----------
    
    f: function f(x, z) -> float
        Returns z for each (x, y)
    x0: float
        Starting x value
    x1: float
        Ending x value
    y0: float
        Starting y value
    y1: float
        Ending y value
    x_count: int
        Number of vertices in the x dimension
    y_count: int
        Number of vertices in the y dimension
            
    Returns
    -------
    array of vertices
        The vertices of the surface
    """
    
    verts = np.empty((x_count*y_count, 3))
    for i, (x, y) in zip(itertools.count(), surface_iterator(x0, x1, y0, y1, x_count, y_count)):
        verts[i] = [x, y, f(x, y)]
    return verts

# -----------------------------------------------------------------------------------------------------------------------------
# Calc a spherical surface

def calc_sph_surface(f, x_count, y_count, phi0=None, phi1=None, theta0=None, theta1=None):
    """Calc a surface parameterized by a spherical function.
    
    if phi or theta limits are not None, the dimension is not looped
    
    Parameters
    ----------
    
    f: function f(phi, theta) -> rho
        Returns rho for each (phi, theta)
    x_count: int
        Number of vertices in the x dimension
    y_count: int
        Number of vertices in the y dimension
    phi0: float
        If given, overrides the default pi/2
    phi1: float
        If given, overrides the default -pi/2
    theta0: float
        If given, overrides the default 0
    theta1: float
        If given, overrides the default 2*pi
            
    Returns
    -------
    array of vertices
        The vertices of the surface
    """
    
    
    y_loop = (theta0, theta1) == (None, None)
    
    phi0   =  pi/2 if phi0   is None else phi0
    phi1   = -pi/2 if phi1   is None else phi1
    theta0 =  0.   if theta0 is None else theta0
    theta1 =  2*pi if theta1 is None else theta1
        
    verts = np.empty((x_count*y_count, 3))
    for i, (phi, theta) in zip(itertools.count(), surface_iterator(phi0, phi1, theta0, theta1, x_count, y_count, x_loop=False, y_loop=y_loop)):
        verts[i] = sph_xyz(phi, theta, f(phi, theta))
    return verts

# -----------------------------------------------------------------------------------------------------------------------------
# Calc a spherical surface

def calc_cyl_surface(f, z0, z1, x_count, y_count, theta0=None, theta1=None):
    """Calc a surface parameterized by a cylindrical function.
    
    if theta limits are not None, the dimension is not looped
    
    Parameters
    ----------
    
    f: function f(z, theta) -> rho
        Returns rho for each (z, theta)
    z0: float
        Starting z value
    z1: float
        Ending z value
    x_count: int
        Number of vertices in the x dimension
    y_count: int
        Number of vertices in the y dimension
    theta0: float
        If given, overrides the default 0
    theta1: float
        If given, overrides the default 2*pi
            
    Returns
    -------
    array of vertices
        The vertices of the surface
    """

    y_loop = (theta0, theta1) == (None, None)
    theta0 =  0.   if theta0 is None else theta0
    theta1 =  pi   if theta1 is None else theta1
    
    verts = np.empty((x_count*y_count, 3))
    for i, (z, theta) in zip(itertools.count(), surface_iterator(z0, z1, theta0, theta1, x_count, y_count, x_loop=False, y_loop=y_loop)):
        verts[i] = cyl_xyz(z, theta, f(z, theta))
    return verts

# -----------------------------------------------------------------------------------------------------------------------------
# Calc a toric surface

def calc_tor_surface(f, x_count, y_count, major0=None, major1=None, minor0=None, minor1=None, radius=1.):
    """Calc a surface parameterized by a torus function.
    
    if phi or theta limits are not None, the dimension is not looped
    
    Parameters
    ----------
    
    f: function f(theta_major, theta_minor) -> minor_radius
        Returns rho for each (phi, theta)
    x_count: int
        Number of vertices in the x dimension
    y_count: int
        Number of vertices in the y dimension
    major0: float
        If given, overrides the default 0
    major1: float
        If given, overrides the default *pi
    minor0: float
        If given, overrides the default 0
    minor1: float
        If given, overrides the default 2*pi
    radius: float
        Major radius of the tor
            
    Returns
    -------
    array of vertices
        The vertices of the surface
    """
    
    
    x_loop = (major0, major1) == (None, None)
    y_loop = (minor0, minor1) == (None, None)
    
    major0 =  0.   if major0 is None else major0
    major1 =  2*pi if major1 is None else major1
    minor0 =  0.   if minor0 is None else minor0
    minor1 =  2*pi if minor1 is None else minor1
    
    verts = np.empty((x_count*y_count, 3))
    for i, (major, minor) in zip(itertools.count(), surface_iterator(major0, major1, minor0, minor1, x_count, y_count, x_loop=x_loop, y_loop=y_loop)):
        verts[i] = tor_xyz(major, minor, f(major, minor), radius=radius)
    return verts

# -----------------------------------------------------------------------------------------------------------------------------
# Calc a sphere

def calc_sphere(radius=1., x_count=17, y_count=32):
    
    verts = calc_sph_surface(lambda x, y: radius, x_count, y_count)
    
    # Merge the first and the last ring into poles
    pole0 = verts[0]
    pole1 = verts[-1]
    count = (x_count-2)*y_count
    verts = np.resize(verts[y_count:-y_count], count*3 + 6).reshape(count+2, 3)
    
    # And put them at the end
    verts[-2] = pole0
    verts[-1] = pole1
    
    return verts

# -----------------------------------------------------------------------------------------------------------------------------
# Calc a torus

def calc_torus(major_radius=1, minor_radius=0.25, x_count=42, y_count=12):
    return calc_tor_surface(lambda x, y: minor_radius, x_count, y_count, radius=major_radius)

# -----------------------------------------------------------------------------------------------------------------------------
# Calc a cone

def calc_cone(z0=-1., z1=1., slope=1., x_count=3, y_count=32):
    return calc_cyl_surface(lambda z, theta: abs(z*slope), z0, z1, x_count, y_count)

# -----------------------------------------------------------------------------------------------------------------------------
# Surface animator
    
class Surface():
    """Compute a surface from an function.
    
    The function can b of two types:
    - uv function:          it returns a vertex from the (u, v) parameters
    - coordinates function: it returns the third coordinate of two coordinates
    
    The supported coordinates are:
    - XYZ : f(x, y)       -> z
    - SPH : f(phi, theta) -> rho
    - CYL : f(z, theta)   -> rho
    - POL : f(rho, theta) -> z
    - TOR : f(amaj, ami   -> minor radius
    
    Note that the TORUS coordinates uses an additional parameter which is radius for majro radius
    
    In addition to the two parameters, the computation function can accept a third parameter t
    to manager the deformation with time
    
    Parameters
    ----------
    func: function
        Function of template f(x, y[, t]) -> vertex or float
    x0: float
        Starting x value
    x1: float
        Endig x value
    y0: float
        Starting y value
    y1: float
        Ending x value
    x_count: int
        Number of vertices in the x dimension
    y_count: int
        Number of vertices in the y dimension
    coords: str
        The coordinates to use in the computation
    x_loop: bool
        Last x index is followed by the first one
    y_loop: bool
        Last y index is followed by the first one
    """
    
    COORDS = ['UV', 'XYZ', 'SPH', 'CYL', 'POL', 'TOR']
    
    def __init__(self, func=lambda x, y, t=0.: t, x0=-1., x1=1., y0=-1., y1=1., x_count=10, y_count=10, coords='XYZ', x_loop=False, y_loop=False):
        if not coords in Surface.COORDS:
            raise WrapException(f"Surface initialization error: '{coords}' is an invalid code for coordinates.")
            
        self.func    = func
        
        self.x0      = x0
        self.x1      = x1
        self.y0      = y0
        self.y1      = y1
        self.x_count = min(max(x_count, 2), 1000)
        self.y_count = min(max(x_count, 2), 1000)
        self.coords  = coords
        self.x_loop  = x_loop
        self.y_loop  = y_loop
        self.poles   = []       # Rings indices which are contracted into a pole
        
        if coords == 'XYZ':
            self.to_xyz = xyz_xyz
        elif coords == 'SPH':
            self.to_xyz = sph_xyz
        elif coords == 'CYL':
            self.to_xyz = cyl_xyz
        elif coords == 'POL':
            self.to_xyz = pol_xyz

    # ----------------------------------------------------------------------------------------------------
    # repr
    
    def __repr__(self):
        s = f"<Surface> {self.x_count} x {self.y_count}"
        if len(self.poles) > 0:
            s += f" with {len(self.poles)} pole{'s' if len(self.poles)>1 else ''} {self.poles}"
        s += f" = {self.verts_count} vertices"
        if len(self.poles) > 0:
            s += f" ({self.verts_in_grid} + {len(self.poles)})"
        s += f"\nCoordinates: {self.coords} on [{self.x0:.2f}, {self.x1:.2f}, {self.y0:.2f}, {self.y1:.2f}]"
        s += f"\nx_loop: {self.x_loop}, y_loop: {self.y_loop}"
        
        return s
            
    # ----------------------------------------------------------------------------------------------------
    # Builders
        
    @classmethod
    def Uv(cls, func, u0=0., u1=1., v0=0., v1=1., x_count=10, y_count=10, x_loop=False, y_loop=False):
        """Create a uv surface"""
        return Surface(func, u0, u1, v0, v1, x_count, y_count, 'UV', x_loop, y_loop)
        
    @classmethod
    def Sphere(cls, radius=1., x_count=15, y_count=32):
        """Create a spherical surface
        
        Note that the two poles are put at the end of the grid
        """
        surf = Surface(lambda x, y, t=radius: t, pis2, -pis2, 0., twpi, x_count, y_count, 'SPH', False, True)
        surf.poles = [0, x_count-1]
        return surf
    
    @classmethod
    def Cylinder(cls, radius=1., z0=-1., z1=1., x_count=10, y_count=32):
        """Create a cylinder surface"""
        return Surface(lambda x, y, t=radius: t, z0, z1, 0., twpi, x_count, y_count, 'CYL', False, True)
    
    @classmethod
    def Disk(cls, radius=1., x_count=10, y_count=32):
        """Create a cylinder surface"""
        surf = Surface(lambda x, y, t=0: t, 0., radius, 0., twpi, x_count, y_count, 'POL', False, True)
        surf.poles = [0]
        return surf
    
    @classmethod
    def Torus(cls, major_radius=1., minor_radius=0.25, x_count=48, y_count=12):
        """Create a torus surface"""
        surf = Surface(lambda x, y, t=minor_radius: t, 0., twpi, 0., twpi, x_count, y_count, 'TOR', True, True)
        surf.radius = major_radius
        return surf
    
    # ----------------------------------------------------------------------------------------------------
    # Number of vertices
    
    @property
    def verts_count(self):
        """Number of vertices.
        
        When a pole is defined, it basically reduces the nuber of vertices
        of self.y_count-1
        """
        return (self.x_count-len(self.poles))*self.y_count + len(self.poles)

    @property
    def verts_in_grid(self):
        """Number of vertices in the grid (without the poles).
        
        When a pole is defined, it basically reduces the nuber of vertices
        of self.y_count-1
        """
        return (self.x_count-len(self.poles))*self.y_count
    
    # ----------------------------------------------------------------------------------------------------
    # Iterator
    
    def iterator(self):
        """Returns an iterator to compute the function.
        
        If poles exist, the iterator chain partial iterators and then
        adds the poles coordinates
        """
        # Has poles
        if len(self.poles) > 0:
            
            dx = (self.x1-self.x0)/(self.x_count-1)
            
            # Chained iterators
            surf_iters = []
            poles_iter = []
            ring0 = 0
            for pole_ring in itertools.chain(self.poles, [self.x_count]):
                if pole_ring > ring0:
                    surf_iters.append(
                            surface_iterator(self.x0 + ring0*dx, self.x0 + (pole_ring-1)*dx, self.y0, self.y1, pole_ring-ring0, self.y_count, False, self.y_loop)
                            )
                
                ring0 = pole_ring+1

                if pole_ring < self.x_count:
                    poles_iter.append((self.x0 + dx*pole_ring, self.y0))
                    
            if len(surf_iters) == 1:
                return itertools.chain(surf_iters[0], poles_iter)
            else:
                return itertools.chain(itertools.chain(tuple(surf_iters)), poles_iter)
        
        return surface_iterator(self.x0, self.x1, self.y0, self.y1, self.x_count, self.y_count, self.x_loop, self.y_loop)
    
    # ----------------------------------------------------------------------------------------------------
    # Surface computation
    
    def compute(self, verts, t=None):
        """Compute the vertices on a existing array of vertices.
        
        The attribute func must be initialized before calling this method
        
        Use build method the create the initial array
        
        Parameters
        ----------
        verts: array of vertices
            The number of vertices in the array must match the number of computed vertices
        t: float or None
            The time parameter. If None, the functions is simply called with f(x, y)
        """
        
        # Error
        if self.func is None:
            raise WrapException(
                    "Surface.compute error: the function surface.func must be initialized with a valid function."
                    )

        if len(verts) != self.verts_count:
            raise WrapException(
                    "Surface.compute error: the vertices array is not of the right size",
                    f"Computed vertices: {self.x_count} x {self.y_count} + {len(self.poles)} = {self.verts_count}"
                    f"Vertices array length: {len(verts)}"
                    )

        # Call the function with the time parameter
        if t is None:
            ff = lambda x, y, t: self.func(x, y)
        else:
            ff = self.func
            
        # Call depending upon the coords system
        if self.coords == 'UV':
            for i, (u, v) in zip(itertools.count(), self.iterator()):
                verts[i] = ff(u, v, t)
        elif self.coords == 'TOR':
            for i, (x, y) in zip(itertools.count(), self.iterator()):
                verts[i] = tor_xyz(x, y, ff(x, y, t), self.radius)
        else:
            for i, (x, y) in zip(itertools.count(), self.iterator()):
                verts[i] = self.to_xyz(x, y, ff(x, y, t))
                
    # ----------------------------------------------------------------------------------------------------
    # Initial build
                
    def build(self, f=None, t=None):
        """Create the vertices.
        
        Parameters
        ----------
        f: function of template f(x, y[, t])
            The function must return a vertex if the coords is 'UV' otherwirse a float
        t: float or None
            The time parameter. If None, the functions is simply called with f(x, y)
            
        Returns
        -------
        array of vertices
            The surface vertices
        """
        
        if f is not None:
            self.func = f
        
        verts = np.empty((self.verts_count, 3), np.float)
        self.compute(verts, t=t)
        return verts
    
    # ----------------------------------------------------------------------------------------------------
    # Faces
    
    def faces(self):
        faces = []
        
        i_count = self.x_count-1
        j_count = self.y_count-1
        if self.x_loop:
            i_count += 1
        if self.y_loop:
            j_count += 1
            
        
        ring_index = 0
        for i in range(i_count):

            # The current ring is a pole
            # Triangles with the next line
            if i in self.poles:
                # Next can't be a pole !
                if i+1 in self.poles:
                    raise WrapException(f"Surface topology error: the surface can't have to succesive poles: {self.poles}")
                    
                pole_index = self.verts_in_grid + self.poles.index(i)
                for j in range(j_count):
                    faces.append([pole_index, ring_index+(j+1)%self.y_count, ring_index+j])
                    
            # The next line is a pole
            # Triangles with the next pole
            elif i+1 in self.poles:
                
                pole_index = self.verts_in_grid + self.poles.index(i+1)
                
                for j in range(j_count):
                    faces.append([ring_index+j, ring_index+(j+1)%self.y_count, pole_index])
                    
                ring_index += self.y_count
                    
            # Two successive normal lines
            else:
                next_index = (ring_index + self.y_count) % self.verts_in_grid
                for j in range(j_count):
                    nj = (j+1)%self.y_count
                    faces.append([ring_index+j, ring_index+nj, next_index+nj, next_index+j])
                    
                ring_index += self.y_count
                
        return faces
    
    def uvs(self):
        
        # i, x mapped on v
        # j, y mapped on u
        
        i_count = self.x_count-1
        j_count = self.y_count-1
        if self.x_loop:
            i_count += 1
        if self.y_loop:
            j_count += 1
            
        dv = 1/i_count
        du = 1/j_count
        
        """
        if self.x_loop:
            dv = 1./self.x_count
            imax = self.x_count
        else:
            dv = 1./(self.x_count-1)
            imax = self.x_count-1
            
        if self.y_loop:
            du = 1./self.y_count
            jmax = self.y_count
        else:
            du = 1./(self.y_count-1)
            jmax = self.y_count-1
        """
        
        def uv_ij(i, j):
            return [du*j, dv*i]
        
        def uv_pole(i, j):
            return [du*(j+0.5), dv*i]
        
        def uv(i,j):
            if i in self.poles:
                return [uv_pole(i, j), uv_ij(i+1, j+1), uv_ij(i+1, j)]
            elif i+1 in self.poles:
                return [uv_ij(i, j), uv_ij(i, j+1), uv_pole(i+1, j)]
            else:
               return [uv_ij(i, j), uv_ij(i+1, j), uv_ij(i+1, j+1), uv_ij(i, j+1)] 
        
        uvs = [uv(i,j) for i, j in itertools.product(range(i_count), range(j_count))]
        
        if False:
            index = 0
            for i in range(i_count):
                s = "< "
                for j in range(j_count):
                    s += "["
                    for v in uvs[index]:
                        s += f"({v[0]:5.2f} {v[1]:5.2f}) "
                    s = s[:-1] + "] "
                    index += 1
                print(s + ">")
            
        return uvs
            
                
    
    
def test():
    def f(x, y, t):
        return t
    
    #surf = Surface(x_count=9, y_count=4, x_loop=False, y_loop=False)
    #surf = Surface.Sphere(x_count=5, y_count=4)
    surf = Surface.Cylinder(x_count=5, y_count=4)
    #surf = Surface.Torus(x_count=5, y_count=4)
    
    print("-"*100)
    print(surf)
    print(len(surf.faces()))
    print(len(surf.uvs()))
    

test()    
    
    
    
        
    
        


    
    
    
    
    
