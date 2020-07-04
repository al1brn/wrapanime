#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:32:18 2020

@author: alain
"""

import bpy
import numpy as np

# Get an array of vector property transformed by a Matrix
def get_transformed_array(coll, prop, M=None, rotation_only=False):
    count = len(coll)

    coords = np.empty(count*3, np.float)
    coll.foreach_get(prop, coords)

    if M is None:
        return coords.reshape(count, 3)

    if rotation_only:
        M = M.to_3x3().normalized().to_4x4()

    coords_4d = np.ones((count, 4), np.float)
    coords_4d[:, :-1] = coords.reshape(count, 3)

    coords = np.einsum('ij,aj->ai', M, coords_4d)[:, :-1]

    return coords.reshape(count, 3)

# Get the transformed vertices

def get_vertices(mesh, M = None):

    verts = mesh.vertices
    count = len(verts)

    coords = np.empty((count, 3), np.float)
    verts.foreach_get('co', np.reshape(coords, count * 3))

    if M is not None:
        coords_4d = np.ones((count, 4), np.float)
        coords_4d[:, :-1] = coords

        coords = np.einsum('ij,aj->ai', M, coords_4d)[:, :-1]

    return coords

def get_faces_centers(mesh, M = None):
    polys = mesh.polygons
    count = len(polys)

    coords = np.empty((count, 3), 'f')
    polys.foreach_get('center', np.reshape(coords, count * 3))

    if M is not None:
        coords_4d = np.ones((count, 4), 'f')
        coords_4d[:, :-1] = coords

        coords = np.einsum('ij,aj->ai', M, coords_4d)[:, :-1]

    return coords

def get_faces_normals(mesh, M = None):

    polys = mesh.polygons
    count = len(polys)

    coords = np.empty((count, 3), 'f')
    polys.foreach_get('normal', np.reshape(coords, count * 3))

    if M is not None:
        coords_4d = np.ones((count, 4), 'f')
        coords_4d[:, :-1] = coords

        coords = np.einsum('ij,aj->ai', M, coords_4d)[:, :-1]

    return coords

def get_edges(mesh, vertices=None, M=None):
    verts = vertices
    if verts is None:
        verts = get_vertices(mesh, M)

    return np.array([[verts[edge.vertices[0]], verts[edge.vertices[1]]] for edge in mesh.edges])


def set_vertices(mesh, vertices, mask='XYZ'):

    verts = mesh.vertices
    count = len(verts)

    if mask == 'XYZ':
        verts.foreach_set('co', np.reshape(vertices, count * 3))
        return

    # Existing coords
    coords = np.empty((count, 3), 'f')
    verts.foreach_get('co', np.reshape(coords, count * 3))

    for i in range(3):
        if 'XYZ'[i] in mask:
            coords[:,i] = coords[:,i]

    verts.foreach_set('co', np.reshape(coords, count * 3))
