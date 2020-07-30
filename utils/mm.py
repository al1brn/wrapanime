#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:13:18 2020

@author: alain.bernard@loreal.com
"""

# -----------------------------------------------------------------------------------------------------------------------------
# Transpose a matrix

def transp(m):
    t = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(3):
        for j in range(3):
            t[i][j] = m[j][i]
    return t

# -----------------------------------------------------------------------------------------------------------------------------
# Print a matrix

def print_mat(m):
    print('-'*70)
    for i in range(3):
        print(f"{m[i][0]:18} | {m[i][1]:18} | {m[i][2]:18}")
    print()
   
# -----------------------------------------------------------------------------------------------------------------------------
# Formal multiplication of two matrices

def matmul(ma, mb, sign = '*'):
   
    m = [["", "", ""], ["", "", ""], ["", "", ""]]
   
    for i in range(3):
        for j in range(3):
            for k in range(3):
                sa = ma[i][k]
                sb = mb[k][j]
                if sa != "" and sb != "":
                    sgn = 1
                    if sa[0] == "-":
                        sgn *= -1
                        sa = sa[1:]
                    if sb[0] == "-":
                        sgn *= -1
                        sb = sb[1:]
                       
                    if sa == "1":
                        s = sb
                    elif sb == "1":
                        s = sa
                    else:
                        s = f"{sa}{sign}{sb}"
                   
                    if sgn == 1:
                        if m[i][j] == "":
                            m[i][j] = s
                        else:
                            m[i][j] += " + " + s
                    else:
                        if m[i][j] == "":
                            m[i][j] = "-" + s
                        else:
                            m[i][j] += " - " + s
                           
    return m

# -----------------------------------------------------------------------------------------------------------------------------
# Base rotation matrices

M = {}

M['X'] = [["1", "", ""], ["", "cx", "-sx"], ["", "sx", "cx"]]
M['Y'] = [["cy", "", "sy"], ["", "1", ""], ["-sy", "", "cy"]]
M['Z'] = [["cz", "-sz", ""], ["sz", "cz", ""], ["", "", "1"]]

orders = ['XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX']

# -----------------------------------------------------------------------------------------------------------------------------
# The 6 euler matrices

def euler_matrix(order):
    return matmul(M[order[2]], matmul(M[order[1]], M[order[0]]))

# -----------------------------------------------------------------------------------------------------------------------------
# Dump the euler matrices
   
def dump_base():
   
    print('-'*100)
    print("Base matrices")
    print()

    for axis in 'XYZ':
        print(axis)
        print_mat(M[axis])
       
    print('-'*100)
    print("Euler matrices")
    print()

    for order in orders:
        print(order)
        print_mat(euler_matrix(order))
       
dump_base()


# -----------------------------------------------------------------------------------------------------------------------------
# Source code

def build_matrix(order, tab=4):
    blank = " "*tab
    m = euler_matrix(order)
    for i in range(3):
        for j in range(3):
            yield blank + f"m[:, {i}, {j}] = {m[i][j]}"
           
           
def build_transfo(tab=4):
    blank = " "* tab
    for i in range(len(orders)):
        order = orders[i]
        if i == 0:
            yield blank + f"if order == '{order}'':"
        else:
            yield blank + f"elif order == '{order}':"
       
        for line in build_matrix(order, 2*tab):
            yield line
        yield ""
           
for line in build_transfo():
    print(line)