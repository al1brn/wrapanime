#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 19:38:21 2020

@author: alain
"""

import timeit
import numpy as np
import sys

sys.path.append(".")


setup_curve = """

import numpy as np
import easing
import importlib
importlib.reload(easing)

from easing import WFCurve


count = 10
a = np.zeros(count*6, np.float).reshape(count, 6)
for i in range(count):
    a[i, 2] = i
a[:, 3] = np.random.random_sample(count)*3.
    
count -= 1

params = np.zeros(count*3).reshape(count, 3)

valids = [k for k in easing.Easing.CANONICS.keys()]
interps = [valids[np.random.randint(0, len(valids))] for i in range(count)]

modes   = [easing.Easing.EASINGS[np.random.randint(0, 2)] for i in range(count)]
    
curve = easing.WFCurve(a, params, interps, modes)

xs = np.random.random_sample(1000)*10. - 3.
"""

curve_s = "ys = curve(xs)"

t0 = timeit.timeit(stmt=curve_s, setup=setup_curve, number=1000, globals=None)

print(f"{t0:.3f}")









setup = """
import numpy as np
import bfunctions as bfunc
from math import sin

import importlib
importlib.reload(bfunc)

bf = bfunc.BFunction.Elastic(0., 7.)
count=1000
xs = np.random.random_sample(count)*10. - 3.
"""

stmt0 = """
ys = bf.comp(xs)
"""


stmt1 = """
ys = [bf(x) for x in xs]
"""


stmt2 = """
a = 3.
b = 5.
c = 7.
ys = [a*sin(b*x+c) for x in xs]
"""

stmt3 = """
a = 3.
b = 5.
c = 7.
ys = a*np.sin(b*xs+c)
"""
 
"""
t0 = timeit.timeit(stmt=stmt0, setup=setup, number=1000, globals=None)
t1 = timeit.timeit(stmt=stmt1, setup=setup, number=1000, globals=None)
t2 = timeit.timeit(stmt=stmt2, setup=setup, number=1000, globals=None)
t3 = timeit.timeit(stmt=stmt3, setup=setup, number=1000, globals=None)

print(f"Comp> np: {t0:.3f} | for: {t1:.3f} ({t0/t1*100:.2f}%) | [sin] {t2:.3f} ({t0/t2*100:.2f}%) | np.sin: {t3:.3f} ({t0/t3*100:.2f}%)")
"""
