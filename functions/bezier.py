import numpy as np

from math import cos, sin

# =============================================================================================================================
# FOR DEBUG
# Import matplotlib which is not included in Blender

""" 

import matplotlib.pyplot as plt

WrapException = Exception
    
"""

from wrapanime.utils.errors import WrapException

#"""

# =============================================================================================================================
# DEBUG: Dump bpoints
    
def _dump_bpoints(bpoints, title="Dump bpoints"):
    print('-'*100)
    print(title)
    print()
    
    def pline(i):
        def pvec(v):
            return f"[{v[0]:8.2f} {v[1]:8.2f} {v[2]:8.2f}]"
        return f"{i:3}> {pvec(bpoints[i, 0:3])} {pvec(bpoints[i, 3:6])} {pvec(bpoints[i, 6:9])}"
    
    lmax = 10
    if len(bpoints) <= 2*lmax:
        lmax = len(bpoints)
        etc = False
    else:
        etc = True
    
    for i in range(lmax):
        print(pline(i))
    
    if etc:
        print("...")
        for i in range(len(bpoints)-lmax, len(bpoints)):
            print(pline(i))
            
    print()
            

# =============================================================================================================================
# Compute a point on array of bezier points
    
def bcompute(bpoints, factor):
    """Return the location on a Bezier curve from a factor between 0 and 1
    
    Parameters
    ----------
    bezier: np.array of shape (n, 3, 3)
        The bezier curve control points
        
    factor: float
        A float between 0 and 1 representing the location on the curve
        
    Returns
    -------
    Vector
        The location on the curve
    """
    
    count = len(bpoints)

    index = int(factor * (count-1))
    delta = 1./(count-1)
    t = factor/delta - index

    if index < 0:
        C = bpoints[0, 3:6]
        L = bpoints[0, 0:3]
        return C - t*(L-C)

    if index >= count-1:
        C = bpoints[-1, 3:6]
        R = bpoints[-1, 6:9]
        return C + t*(R-C)

    t2 = t*t
    t3 = t2*t

    umt = 1.-t
    umt2 = umt*umt
    umt3 = umt2*umt
    
    a = bpoints.reshape(count*3, 3)
    i = index*3 + 1
    return umt3*a[i] + 3*t*umt2*a[i+1] + 3*umt*t2*a[i+2] + t3*a[i+3]


# =============================================================================================================================
# 3D Bezier : Bezier curve
#
# Points are stored in an array [Left Central Right] : ie 9 floats

class BezierCurve():
    
    def __init__(self, bpoints, t0=0., t1=1.):
        self.bpoints = self.check_bpoints(bpoints)
        self.t0      = t0
        self.length  = t1 - t0
        
    def __repr__(self):
        return f"BezierCurve[{len(self)} points, length {self.length:.1f}]"
        
    def _dump(self):
        _dump_bpoints(self.bpoints, self)
        
    @classmethod
    def check_bpoints(self, bpoints):
        a = np.array(bpoints)
        length, rem = divmod(a.size, 9)
        if rem != 0:
            raise WrapException(
                f"BezierCurve initialization error: the size of the array of control points is incorrect",
                f"The array of control points must include lines of triplets",
                f"The size of the array is {a.size} which is not a multiple of 9"
                )

        if length < 2:
            raise WrapException(
                "Bezier curve initialization error",
                f"Six control points at least are required: {length} is not valid."
                )
            
        return a.reshape(length, 9)
    
    # Compute
    def __call__(self, t):
        
        # Outside the interval
        
        if t <= self.t0:
            delta = self.length / (len(self)-1)
            
            C = self.bpoints[0, 3:6]
            L = self.bpoints[0, 0:3]
            return C - (t-self.t0)*(L-C)*3/delta

        if t >= self.t0 + self.length:
            delta = self.length / (len(self)-1)
            
            C = self.bpoints[-1, 3:6]
            R = self.bpoints[-1, 6:9]
            return C + (t - self.t0 - self.length)*(R-C)*3/delta
        
        # Ok, inside
            
        return bcompute(self.bpoints, (t - self.t0)/self.length)
    
    # Array implementation
    
    def __len__(self):
        return len(self.bpoints)
    
    def __getitem__(self, index):
        return self.bpoints[index]
    
    # Access to the points
    
    def left_point(self, index):
        return self.bpoints[index, 0:3]
    
    def point(self, index):
        return self.bpoints[index, 3:6]
    
    def right_point(self, index):
        return self.bpoints[index, 6:9]
    
    def set_left_point(self, index, P):
        self.bpoints[index, 0:3] = np.array(P)
        
    def set_point(self, index, P):
        self.bpoints[index, 3:6] = np.array(P)
        
    def set_right_point(self, index, P):
        self.bpoints[index, 6:9] = np.array(P)
        
    # Creation from a function
    
    @classmethod
    def FromFunction(cls, f, t0=0., t1=1., count=100):
        
        count  = min(1000, max(count, 2))
        length = t1-t0
        if abs(length) < 0.001:
            length = 0.001
            
        delta = length / (count-1)
        dt = delta/10.
        
        def derivative(t):
            return (np.array(f(t+dt)) - np.array(f(t-dt)))/2/dt
        
        bpoints = np.empty(count*9, np.float).reshape(count, 9)
        
        for i in range(count):
            t = t0 + i*delta
            
            P = np.array(f(t))
            D = derivative(t)
            
            bpoints[i, 0:3] = P - D/3*delta
            bpoints[i, 3:6] = P
            bpoints[i, 6:9] = P + D/3*delta
            
        return cls(bpoints, t0, t1)


# -----------------------------------------------------------------------------------------------------------------------------
# A test function
    
def test():
    
    def helix(t):
        z_speed  = 2.
        r_speed  = 1.
        w        = 1.  # Angular speed
        r0       = 1.
        
        r = r0  + r_speed*t
        
        return (r*cos(w*t), r*sin(w*t), z_speed*t)
    
    t0 = 0.
    t1= 10.
    bc = BezierCurve.FromFunction(helix, t0, t1)
    
    margin = 0.2
    amp = t1*(1+2*margin)
    
    count = 101
    xs = np.arange(count)/(count-1)*amp - margin*t1
    ys = np.array([bc(x) for x in xs])
    
    fig, ax = plt.subplots()
    ax.plot(xs, ys[:, 0])
    ax.plot(xs, ys[:, 1])
    ax.plot(xs, ys[:, 2])
    
    ax.set(xlabel='t', ylabel='curve(x, y, z)',
           title=bc)
    ax.grid()
    
    fig.savefig("test.png")
    plt.show()
    
    bc._dump()
    
#test()
