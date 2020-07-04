import numpy as np
from math import cos, sin, tan, log, exp
import itertools

#import matplotlib.pyplot as plt # Comment in production

#from wrapanime.utils.errors import WrapException

# =============================================================================================================================
# Function class

class Function():
    def __init__(self, f, derivative=None, name=None, dt=0.0001):
        self.f = f
        self._derivative = derivative
        self.dt = dt
        self.name = f.__name__ if name is None else name
        
    def __repr__(self):
        return f"Function: {self.name.replace('_', 't')}"

    def __call__(self, t):
        return self.f(t)
    
    def plot(self, t0, t1, count=500, fcomp=None):
        
        count = max(2, count)
        
        xs = np.arange(t0, t1, (t1-t0)/(count-1), dtype=np.float)
        ys = [self(x) for x in xs]
        
        fig, ax = plt.subplots()
        ax.plot(xs, ys)
        
        if fcomp is not None:
            ax.plot(xs, [fcomp(x) for x in xs])
        
        ax.set(xlabel='x', ylabel=type(self).__name__+'(x)',
               title=f"{self}")
        ax.grid()
        
        fig.savefig("test.png")
        plt.show()
        

    def derivative(self, t):
        if self.derivative is None:
            return (self.f(t+self.dt) - self.f(t-self.dt))/2/self.dt
        else:
            return self.derivative(t)
        
    def __neg__(self):
        return Function(lambda t: -self(t), name=f"-({self.name})")
    
    def __abs__(self):
        return Function(lambda t: abs(self(t)), name=f"abs({self.name})")
                        
    def __invert__(self):
        return Function(lambda t: 1./self(t), name=f"1/({self.name})")
                        

    def __mul__(self, other):
        return Function(lambda t: self(t) * other(t), name=f"({self.name}) * ({other.name})")
    def __truediv__(self, other):
        return Function(lambda t: self(t) / other(t), name=f"({self.name}) / ({other.name})")
    def __add__(self, other):
        return Function(lambda t: self(t) + other(t), name=f"{self.name} + {other.name}")
    def __sub__(self, other):
        return Function(lambda t: self(t) - other(t), name=f"{self.name} - ({other.name})")

    # ----- Power ** to implement functions composition

    def __pow__(self, other):
        name = self.name.replace('_', other.name)
        return Function(lambda t: self(other(t)), name=name)
    
    def length(self, t0, t1, count=1000):
        dt = (t1-t0)/count
        s = 0.
        y0 = self(t0)
        for t in itertools.accumulate(itertools.repeat(dt, count)):
            y1 = self(t0+t)
            s += abs(y1-y0)
            y0 = y1
        return s

    # return the Bezier control points
    def bezier_points(self, t0, t1, count=10):
        
        points = np.zeros((count, 3, 2), np.float) # Array of 3 vectors : point, left and right handles

        dt = (t1-t0)/(count-1)

        for i in range(count):
            t  = t0 + i*dt
            y  = self(t)
            yp = self.derivative(t)

            points[i][0] = [t,        y]
            points[i][1] = [t - dt/3, y - yp/3]
            points[i][2] = [t + dt/3, y + yp/3]

        return points
    
    # Predefinef functions
    
    @classmethod
    def Constant(Cls, a=1., name=None):
        return Function(lambda t: a, lambda t: 0., name=f"{a}" if name is None else name)

    @classmethod
    def Linear(Cls, a=1., b=0., name=None):
        return Function(lambda t: a*t + b, lambda t: a, name=f"({a}*(_)+{b})" if name is None else name)
    
    @classmethod
    def Sine(Cls, amp=1., w=1, phi=0., name=None):
        return Function(lambda t: amp*sin(w*t + phi), lambda t: w*amp*cos(w*t + phi), name=f"{amp}*sin({w}*(_)+{phi})" if name is None else name)

    @classmethod
    def Cosine(Cls, amp=1., w=1, phi=0., name=None):
        return Function(lambda t: amp*sin(w*t + phi), lambda t: -w*amp*sin(w*t + phi), name="cos(_)" if name is None else name)

    @classmethod
    def Tangent(Cls, w=1., phi=0., name=None):
        return Function(lambda t: tan(w*t + phi), lambda t: w/(cos(t)**2), name="tan(_)" if name is None else name)

    @classmethod
    def Log(Cls, name=None):
        return Function(log, lambda t: 1/t, name="log(_)" if name is None else name)

    @classmethod
    def Exp(Cls, a=1., b=1., name=None):
        return Function(lambda t: a*exp(b*t), lambda t: a*b*exp(b*t), name="exp(_)" if name is None else name)
    
    @classmethod
    def Power(Cls, n=1, name=None):
        return Function(lambda t: t**n, lambda t: n*t**(n-1), name=f"(_)**{n}" if name is None else name)
    

# =============================================================================================================================
# Parameterized curve
# Return vertices rather than reals

class PCurve():
    def __init__(self, f, name=None, dt=0.0001):
        self.f = f
        self.dt = dt
        self.name = f.__name__ if name is None else name
        
    def __repr__(self):
        return f"PCurve: {self.name.replace('_', 't')}"

    def __call__(self, t):
        return self.f(t)
    
    def plot(self, t0, t1, count=500, fcomp=None):
        
        count = max(2, count)
        
        xs = np.arange(t0, t1, (t1-t0)/(count-1), dtype=np.float)
        ys = [self(x) for x in xs]
        ys = np.array(ys)
        
        fig, ax = plt.subplots()
        ax.plot(xs, ys[:, 0])
        ax.plot(xs, ys[:, 1])
        ax.plot(xs, ys[:, 2])
        
        if fcomp is not None:
            ax.plot(xs, [fcomp(x) for x in xs])
        
        ax.set(xlabel='x', ylabel=type(self).__name__+'(x)',
               title=f"{self}")
        ax.grid()
        
        fig.savefig("test.png")
        plt.show()
        

    def derivative(self, t):
        return (np.array(self.f(t+self.dt)) - np.array(self.f(t-self.dt)))/2/self.dt
        
    
    def length(self, t0, t1, count=1000):
        dt = (t1-t0)/count
        s = 0.
        P0 = np.array(self(t0))
        for t in itertools.accumulate(itertools.repeat(dt, count)):
            P1 = np.array(self(t0+t))
            s += np.linalg.norm(P1-P0)
            P0 = P1
        return s

    # return the Bezier control points
    def bezier_points(self, t0, t1, count=10):
        
        points = np.zeros((count, 3, 3), np.float) # Array of 3 vectors : point, left and right handles

        dt = (t1-t0)/(count-1)

        for i in range(count):
            t = t0 + i*dt
            P = self(t)
            D = self.derivative(t)

            points[i][0] = P
            points[i][1] = P - D/3.
            points[i][2] = P + D/3.

        return points
    
    # Predefinef functions
    
    @classmethod
    def Line(Cls, O=(0., 0., 0.), V=(1., 0., 0.), name=None):
        o = np.array(O)
        v = np.array(V)
        return Function(lambda t: o + v*t, name=f"line" if name is None else name)

    @classmethod
    def Helix(Cls, O=(0., 0., 0.), w=1., radius=1., vz=0., name=None):
        o = np.array(O)
        if name is None:
            if abs(vz) < 0.0001:
                name = "Circle"
            else:
                name = "Helix"
        return Function(lambda t: o + np.array([radius*cos(w*t), radius*sin(w*t), vz*t]), name=name)
    
    @classmethod
    def Trochoid(Cls, O=(0., 0., 0.), V=(1., 0., 0.), w=1., radius=1., vz=0., name=None):
        o = np.array(O)
        v = np.array(V)
        if name is None:
            if abs(vz) < 0.0001:
                name = "Trochoid"
            else:
                name = "Helicoidal trochoid"
        return Function(lambda t: o + v*t + np.array([radius*cos(w*t), radius*sin(w*t), vz*t]), name=name)
    
    @classmethod
    def Spiral(Cls, O=(0., 0., 0.), w=1., vr=1., vz=0., name=None):
        o = np.array(O)
        if name is None:
            if abs(vz) < 0.0001:
                name = "Spiral"
            else:
                name = "Helicoidal spiral"
        return Function(lambda t: o + np.array([vr*t*cos(w*t), vr*t*sin(w*t), vz*t]), name=name)
    



