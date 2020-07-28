#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:13:18 2020

@author: alain.bernard@loreal.com
"""

def pmat(m):
    print('-'*70)
    for i in range(3):
        print(f"{m[i][0]:18} | {m[i][1]:18} | {m[i][2]:18}")
    print()

def mmul(ma, mb):
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
                        s = f"{sa}.{sb}"
                    
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

                            
ma = [["1", "", ""], ["", "cos(a)", "-sin(a)"], ["", "sin(a)", "cos(a)"]]
mb = [["cos(b)", "", "sin(b)"], ["", "1", ""], ["-sin(b)", "", "cos(b)"]]
mc = [["cos(c)", "-sin(c)", ""], ["sin(c)", "cos(c)", ""], ["", "", "1"]]


M = {}

M['X'] = [["1", "", ""], ["", "cx", "-sx"], ["", "sx", "cx"]]
M['Y'] = [["cy", "", "sy"], ["", "1", ""], ["-sy", "", "cy"]]
M['Z'] = [["cz", "-sz", ""], ["sz", "cz", ""], ["", "", "1"]]

def morder(order):
    return mmul(M[order[2]], mmul(M[order[1]], M[order[0]]))

for axis in 'XYZ':
    print(axis)
    pmat(M[axis])
    
orders = ['XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX']
for order in orders:
    print(order)
    pmat(morder(order))
    

