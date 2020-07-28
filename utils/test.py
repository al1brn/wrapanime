#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 13:49:32 2020

@author: alain.bernard@loreal.com
"""
import math
import numpy as np

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

order = 'XYZ'
m = np.array([xxx
    [-0.73, -0.58, -0.36],
    [ 0.68, -0.63, -0.37],
    [-0.02, -0.51,  0.86]
    ])

m /= np.power(np.linalg.det(m), 1/3)
print("det", np.linalg.det(m))
e = rotationMatrixToEulerAngles(m)
print(e)

aaaa
mb = e_to_matrix(e, order)

print(_str(m, 2))
print(_str(e, 1, 'degrees'), order)
print(_str(mb, 2))
print("diff", np.linalg.norm(m.reshape(9) - mb.reshape(9)))
print(_str(m_rotate(m,  (1, 1, 1))))
print(_str(m_rotate(mb, (1, 1, 1))))

