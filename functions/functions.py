import numpy as np
from math import cos, sin, tan, log, exp, pi
import itertools
import inspect

#import matplotlib.pyplot as plt # Comment in production

#from wrapanime.utils.errors import WrapException


#import matplotlib.pyplot as plt

WrapException = Exception


# =============================================================================================================================
# Root function fo Function and Bezier Function
# Managing extrapolation outside an interval [x_min, x_max]

class RootFunction():
    
    extrapolations = ['CONSTANT', 'LINEAR', 'CYCLIC']
    
    def __init__(self, name="root", extrapolation='CONSTANT'):
        if not extrapolation in self.extrapolations:
            raise WrapException(f"Function initialization error: extrapolation '{extrapolation}' is not valid",
                                f"Valid codes are: {self.extrapolations}."
                                )
            
        self.extrapolation = extrapolation
        self.name          = name
        self.dx            = 0.001
        
        # User space
        
        self.y0      = 0.
        self.y_scale = 1.
        self.x0      = 0.
        self.x_scale = 1.
        
        # Interval (extrapolation outside this interval)
        self.x_min   = np.NINF
        self.x_max   = np.inf
        
        
    def __repr__(self):
        return f"Function({self.name})"
    
    # =============================================================================================================================
    # Dump
    
    def _dump(self):
        print('-'*50)
        print(f"Dump> {self}")
        print()
        print("--- User space")
        print(f"x0:      {self.x0}")
        print(f"x_scale: {self.x_scale}")
        print(f"y0:      {self.y0}")
        print(f"y_scale: {self.y_scale}")
        print(f"x_min:   {self.x_min}")
        print(f"x_max:   {self.x_max}")
        print()
    
        
    # =============================================================================================================================
    # Plot the function

    def _plot(self, count=100, margin=0., tangents=False, points=False, fcomp=None):
        
        x0 = self.x0
        x1 = x0 + self.x_scale
        amp = x1-x0
        
        x0 -= margin*amp
        x1 += margin*amp
        dx = (x1-x0)/(count-1)
        
        xs = np.arange(x0, x1+dx, dx, dtype=np.float)
        ys = self(xs)

        fig, ax = plt.subplots()
        
        if tangents:
            for pts in self:
                ax.plot([pts[0], pts[2], pts[4]], [pts[1], pts[3], pts[5]], '-o', color='black')
        
        ax.plot(xs, ys)
        
        if fcomp is not None:
            ax.plot(xs, [fcomp(x) for x in xs])
            
        if points:
            xp = self.bpoints[:, 2]
            yp = self.bpoints[:, 3]
            ax.plot(xp, yp, 'o')
            
        
        ax.set(xlabel='x', ylabel='bfunction(x)',
               title=f"{self}")
        ax.grid()
        
        fig.savefig("test.png")
        plt.show()
        
        
    # =============================================================================================================================
    # Canonical computation
    
    def compute(self, x):
        return x
    
    # =============================================================================================================================
    # Derivative
    
    def derivative(self, x):
        dxi = 0.5/self.dx
        return (self.compute(x+self.dx)- self.compute(x-self.dx))*dxi
    
    # =============================================================================================================================
    # User space computation
    
    def __call__(self, x):
        
        if hasattr(x, '__len__'):
            length = len(x)
            xs = np.array((x-self.x0)/self.x_scale)
        else:
            length = 1
            xs = np.full((1,), (x-self.x0)/self.x_scale, np.float)
        
        # Cyclic extrapolation
        if self.extrapolation == 'CYCLIC':
            xs = self.x_min + (xs - self.x_min)/(self.x_max - self.x_min)
            
        # Result
        ys = np.empty(length, np.float)
            
        # Points below 
        i_inf = np.where(xs <= self.x_min)[0]
        if len(i_inf) > 0:
            xm = (self.x_min-self.x0)/self.x_scale
            slope = 0. if self.extrapolation == 'CONSTANT' else self.derivative(xm)
            ys[i_inf] = self.compute(xm) + (xs[i_inf]-self.x_min)*slope
        
        # Points above
        i_sup = np.where(xs >= self.x_max)[0]
        if len(i_sup) > 0:
            xm = (self.x_max-self.x0)/self.x_scale
            slope = 0. if self.extrapolation == 'CONSTANT' else self.derivative(xm)
            ys[i_sup] = self.compute(self.x_max) + (xs[i_sup]-self.x_max)*self.right_slope
        
        # Remaining points
        idx = np.delete(np.arange(length), np.append(i_inf, i_sup))
        
        ys[idx] = self.compute(xs[idx])
        
        return self.y0 + ys*self.y_scale

# =============================================================================================================================
# Function class

class Function(RootFunction):
    def __init__(self, func, derivative=None, name=None):
        
        name = func.__name__ if name is None else name
        super().__init__(name)
        
        self.func        = func
        self._derivative = derivative
        
    def __repr__(self):
        return f"Function: {self.name.replace('_', 'x')}"
    
    def _dump(self):
        super()._dump()
        print("--- Function")
        try:
            print(inspect.getsource(self.func))
        except:
            print("No source code available")
        print()
        print("--- Derivative")
        if self._derivative is None:
            print("None")
        else:
            print(self.derivative.__name__)
            try:
                print(inspect.getsource(self._derivative))
            except:
                print("No source code available")
        print()
    
    # =============================================================================================================================
    # Computation
    
    def derivative(self, x):
        if self._derivative is None:
            return super().derivative(self, x)
        else:
            return self._derivative(x)

    def compute(self, x):
        return self.func(x)
    
    # =============================================================================================================================
    # Operations on functions
        
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

    # =============================================================================================================================
    # Power ** to implement functions composition

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
    
    # =============================================================================================================================
    # Predefinef functions
    
    @classmethod
    def Constant(cls, a=1., name=None):
        return cls(lambda x: a, lambda x: 0., name=f"{a}" if name is None else name)

    @classmethod
    def Linear(cls, a=1., b=0., name=None):
        return cls(lambda x: a*x + b, lambda x: a, name=f"linear" if name is None else name)
    
    @classmethod
    def Sine(cls, amp=1., omega=1, phi=0., name=None):
        return cls(lambda x: amp*np.sin(omega*x + phi), lambda x: omega*amp*np.cos(omega*x + phi), name=f"sin" if name is None else name)

    @classmethod
    def Cosine(cls, amp=1., omega=1, phi=0., name=None):
        return cls(lambda x: amp*np/sin(omega*x + phi), lambda x: -omega*amp*np.sin(omega*x + phi), name="cos" if name is None else name)

    @classmethod
    def Tangent(cls, omega=1., phi=0., name=None):
        def tang(x):
            return np.tan(omega*x + phi)
        def dtang(x):
            t = tang(x)
            return 1. + omega*t*t
            
        return cls(tang, dtang, name="tan" if name is None else name)

    @classmethod
    def Inverse(cls, name=None):
        return cls(lambda x: 1/x, lambda x: -1/x/x, name="inv" if name is None else name)

    @classmethod
    def Square(cls, name=None):
        return cls(lambda x: x*x, lambda x: 2*x, name="square" if name is None else name)

    @classmethod
    def SquareRoot(cls, name=None):
        return cls(lambda x: np.sqrt(x), lambda x: 0.5/np.sqrt(x), name="square" if name is None else name)

    @classmethod
    def Pow3(cls, name=None):
        return cls(lambda x: x*x*x, lambda x: 3*x*x, name="pow3" if name is None else name)

    @classmethod
    def Log(cls, name=None):
        return cls(lambda x: np.log(x), lambda x: 1/x, name="log" if name is None else name)

    @classmethod
    def Exp(cls, a=1., b=1., name=None):
        return cls(lambda x: a*np.exp(b*x), lambda x: a*b*exp(b*x), name="exp" if name is None else name)
    
    @classmethod
    def Power(cls, n=1, name=None):
        return cls(lambda x: np.power(x, n), lambda x: n*np.power(x, n-1), name=f"pow{n}" if name is None else name)
    
    
#f = Function.Power(n=7)
#f._dump()
#f._plot()
    

# =============================================================================================================================
# Parameterized curve
# Return vertices rather than floats

class PCurve():
    def __init__(self, f, name=None, dt=0.0001):
        self.f = f
        self.dt = dt
        self.name = f.__name__ if name is None else name
        
    def __repr__(self):
        return f"PCurve: {self.name.replace('_', 't')}"

    def __call__(self, t):
        return np.array(self.f(t))
    
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

            points[i, 0] = P
            points[i, 1] = P - D/3.
            points[i, 2] = P + D/3.

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
    
# -----------------------------------------------------------------------------------------------------------------------------
# Mix two values

class Mixer(Function):
    def __init__(self, f0, f1, mapper=None, name=None):
        self.f0     = f0
        self.f1     = f1
        self.mapper = mapper
        self.name   = name
        self.factor = 0.5
        
    def __repr__(self):
        return f"Mixer({self.f0} <{self.mapper}> {self.f1})" if self.name is None else self.name
        
    def __call__(self, t):
        p = self.mapper(self.factor)
        return self.f0(t)*(1.-p) + self.f1(t)*p
    
# -----------------------------------------------------------------------------------------------------------------------------
# Mix two vectors

class CurveMixer(PCurve):
    
    def __init__(self, f0, f1, mapper=None, name=None):
        self.f0     = f0
        self.f1     = f1
        self.mapper = mapper
        self.name   = name
        self.factor = 0.5
        
    def __repr__(self):
        return f"CurveMixer({self.f0} <{self.mapper}> {self.f1})" if self.name is None else self.name
    
    def __call__(self, factor, t):
        if self.mapper is None:
            p = self.factor
        else:
            p = self.mapper(factor)
        return np.array(self.f0(t))*(1.-p) + np.array(self.f1(t))*p
    

