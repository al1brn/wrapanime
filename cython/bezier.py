import numpy as np

from math import cos, pi, exp

# =============================================================================================================================
# FOR DEBUG
# Import matplotlib which is not included in Blender

"""

import matplotlib.pyplot as plt


def error_header(*args):

    msg  = "\n\n" + "="*80 + "\n"
    msg += "Animation wrapper error\n\n"

    title = args[0] if args else "Undocument error"
    msg += "::: {}\n\n".format(title)

    if args:
        for i in range(1, len(args)):
            msg += "{}\n".format(args[i])

    return msg

class WrapException(Exception):
    def __init__(self, title, *args):
        self.message = error_header(title, *args)

        #print(traceback.format_exc())

    def __str__(self):
        return self.message
    
"""

# =====

# Comment for debug        

from wrapanime.utils.errors import WrapException

# -----------------------------------------------------------------------------------------------------------------------------
# Clip utility

def clip(v, v0, v1):
    """Clip the value within an interval"""
    return min(max(v0, v1), max(min(v0, v1), v))

# -----------------------------------------------------------------------------------------------------------------------------
# Functions

def damping(x, x0=0., x1=1., amp=1., power=5):
    return amp*exp(-clip((x-x0)/(x1-x0), 0., 1.)*power)

# =============================================================================================================================
# 3D Bezier : Beizer curve

# -----------------------------------------------------------------------------------------------------------------------------
# Check if a function bezier array is ok
# Return a proper array of shape (n, 3, 3)

def check_bezier_curve_points(points):
    """Check if an array has the shape of a Bezier curve
    
    Parameters
    ----------
    points: array-like
        The array of float of shape (n, 3, 2) representing the Bezier control points
        
    Raises
    ------
        WrapException if the points array is not a valid Bezier curve
        
    Returns
    -------
    np.array of shape (n, 3, 3)
        The np.array
    """
    
    points = np.array(points)
    if points.size %9 != 0:
        raise WrapException(
            "Bezier curve initialization error",
            f"Control vertices must be given by triplets. The size of resulting array must be a multiple of 9. {points.size} is not a multiple of 6"
            )
    points = points.reshape(points.size//9, 3, 3)
    shape = points.shape
    if shape[0] < 2:
        raise WrapException(
            "Bezier curve initialization error",
            f"Six control points at least are required: {shape[0]} is not valid."
            )

    return points    



# -----------------------------------------------------------------------------------------------------------------------------
# Return the location on a Bezier curve
# Ensure the bezier control points are acceptable, by calling check_bezier_curve_points
    
def bezier_loc(bezier, factor):
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
    
    count = bezier.shape[0]

    index = int(factor * (count-1))
    dt = 1./(count-1)
    t = factor/dt - index

    if index < 0:
        P0 = bezier[0][0]
        P1 = bezier[0][1]
        return P0 + t*(P1-P0)/dt

    if index >= count-1:
        P0 = bezier[count-1][0]
        P1 = bezier[count-1][2]
        return P0 - t*(P1-P0)/dt

    P0 = bezier[index][0]
    P1 = bezier[index][2]
    P2 = bezier[index+1][1]
    P3 = bezier[index+1][0]

    t2 = t*t
    t3 = t2*t

    umt = 1.-t
    umt2 = umt*umt
    umt3 = umt2*umt

    return umt3*P0 + 3*t*umt2*P1 + 3*umt*t2*P2 + t3*P3

# -----------------------------------------------------------------------------------------------------------------------------
# Bezier curve
    
class BezierCurve():
    def __init__(self, points, x0=0., x1=1.):
        self.points = check_bezier_curve_points(points)
        self.x0     = x0
        self.length = x1-x0
        
    def __call__(self, x):
        return bezier_loc(self.points, (x-self.x0)/self.length)
    
    @classmethod
    def FromFunction(Cls, f, x0=0., x1=1., count=100):
        count  = max(count, 10)
        length = x1-x0
        if abs(length) < 0.001:
            length = 0.001
            
        dx = length / (count-1)
        dt = dx/10.
        
        def derivative(x):
            return (np.array(f(x+dt)) - np.array(f(x-dt)))/2/dt
        
        points = np.zeros((count, 3, 3), np.float)
        
        for i in range(count):
            x = x0 + i*dx
            
            P = np.array(f(x))
            D = derivative(x)
            L = P - D/3*dx
            R = P + D/3*dx
            
            points[i, 0] = P
            points[i, 1] = L
            points[i, 2] = R
            
        return BezierCurve(points, x0=x0, x1=x1)


# =============================================================================================================================
# 2D Bezier : Bezier function


# -----------------------------------------------------------------------------------------------------------------------------
# Check if a function bezier array is ok
# Return the input space

def check_bezier_func_points(points):
    """Check if an array has the shape of a functional Bezier curve
    
    Parameters
    ----------
    points: array-like
        The array of float of shape (n, 3, 2) representing the Bezier control points
        
    Raises
    ------
        WrapException if the points array is not a valid Bezier curve
        
    Returns
    -------
    np.array of shape (n, 3, 2)
        The np.array
    """
    
    points = np.array(points)
    if points.size %6 != 0:
        raise WrapException(
            "Bezier function initialization error",
            f"Control points must be given by triplets. The size of resulting array must be a multiple of 6. {points.size} is not a multiple of 6"
            )
    points = points.reshape(points.size//6, 3, 2)
    shape = points.shape
    if shape[0] < 2:
        raise WrapException(
            "Bezier function initialization error",
            f"Six control points at least are required: {shape[0]} is not valid."
            )

    x0 = points[0][0][0]
    x  = x0
    for i in range(1,shape[0]):
        x1 = points[i][0][0]
        if x1 < x+0.0001:
            raise WrapException(
                "Bezier function initialization error",
                "Bezier points must be ordered by their absissa",
                "Point number {} ({}) is less than previous one ({})".format(i, points[i][0], points[i-1][0])
                )
    return points


# -----------------------------------------------------------------------------------------------------------------------------
# Bezier function x --> is in [x0 x1]
# Abscisses intervals are not necessarily equal
# Abscisses must be in growing order

def bezier_func(points, x):
    """Eval teh value of a function approximated by a Bezier curve.
    
    The bezier curve abscissa are not necessarilty evenly spaced.
    On the contrary, the bezier_func algorithm supposes that the intervals
    between abscissa are equal
    
    Parameters
    ----------
    points: np.array of shape (n, 3, 2)
        Bezier control points
        
    x: float
        Input of the function
        
    Returns
    -------
    float
        Value of the function at abscissa x
    """
    
    count = points.shape[0]
    
    x0 = points[0, 0, 0]
    if x <= x0:
        dx = points[1, 0, 0] - points[0, 0, 0]
        tg = 3*(points[0, 0, 1] - points[0, 1, 1])/dx
        return points[0, 0, 1] + tg*(x-x0)

    x1 = points[count-1, 0, 0]
    if x >= x1:
        dx = points[count-1, 0, 0] - points[count-2, 0, 0]
        tg = 3*(points[count-1, 2, 1] - points[count-1, 0, 1])/dx
        return points[count-1, 0, 1] + tg*(x-x1)
    
    # Ok, we are in the middle
    index = 0
    for i in range(1, count):
        if x <= points[i, 0, 0]:
            index = i-1
            break
        
    P0 = points[index,   0]
    y1 = points[index,   2, 1]
    y2 = points[index+1, 1, 1]
    P3 = points[index+1, 0]
    
    dx = P3[0] - P0[0]
    t = (x - P0[0])/dx
    t2 = t*t
    t3 = t2*t

    umt = 1.-t
    umt2 = umt*umt
    umt3 = umt2*umt

    return umt3*P0[1] + 3*umt2*t*y1 + 3*umt*t2*y2 + t3*P3[1]


# -----------------------------------------------------------------------------------------------------------------------------
# Bezier function x --> y# eval is in [x0 x1]
# Abscissa intervals must be equal
# CAUTION: no control is performed on the points array
    
def even_bezier_func(points, x):
    """Return the location on a Bezier curve from a factor between 0 and 1
    
    Parameters
    ----------
    bezier: np.array of shape (n, 3, 3)
        The bezier curve control points. Abscissa of locations are evenly spaced.
        
    x: float
        Input of the function
        
    Returns
    -------
    float
        Value of the function at abscissa x
    """
    
    count = points.shape[0]
    
    # x is out the [min abs - max abs] interval
    
    x0 = points[0, 0, 0]
    if x <= x0:
        dx = points[1, 0, 0] - points[0, 0, 0]
        tg = 3*(points[0, 0, 1] - points[0, 1, 1])/dx
        return points[0, 0, 1] + tg*(x-x0)

    x1 = points[count-1, 0, 0]
    if x >= x1:
        dx = points[count-1, 0, 0] - points[count-2, 0, 0]
        tg = 3*(points[count-1, 2, 1] - points[count-1, 0, 1])/dx
        return points[count-1, 0, 1] + tg*(x-x1)

    # Ok: x belongs to the curve interval
    
    factor = (x-x0)/(x1-x0)
    index = int(factor * (count-1))
    
    P0 = points[index,   0]
    y1 = points[index,   2, 1]
    y2 = points[index+1, 1, 1]
    P3 = points[index+1, 0]
    
    dx = P3[0] - P0[0]
    t = (x - P0[0])/dx
    t2 = t*t
    t3 = t2*t

    umt = 1.-t
    umt2 = umt*umt
    umt3 = umt2*umt

    return umt3*P0[1] + 3*umt2*t*y1 + 3*umt*t2*y2 + t3*P3[1]

# -----------------------------------------------------------------------------------------------------------------------------
# A test function
    
def some_test():

    parab = [
                [ [-1.00,  1.00], [-1.33,  1.66], [-0.66,  0.33] ],
                [ [ 0.00,  0.00], [-0.33,  0.00], [ 0.33,  0.00] ],
                [ [ 1.00,  1.00], [ 0.66,  0.33], [ 1.33,  1.66] ]
            ]
    
    print('-'*50)
    parab = np.array(parab)
    x = 0.
    print("{:2}> {:5.2f} --> {:5.2f}".format(0, x, even_bezier_func(parab, x)))
    print()
    
    count = 10
    for i in range(count):
        x = -1. + 3/(count-1)*i
        print("{:2}> {:5.2f} --> {:5.2f}".format(i, x, even_bezier_func(parab, x)))
    print('-'*50)
    
    count = 100
    x0 = -2
    amp = 4
    dx = amp/(count-1)
    
    xs = np.arange(x0, x0+amp, dx, dtype=np.float)
    ys = [even_bezier_func(parab, x) for x in xs]
    
    fig, ax = plt.subplots()
    ax.plot(xs, ys)
    
    ax.set(xlabel='time (s)', ylabel='voltage (mV)',
           title='About as simple as it gets, folks')
    ax.grid()
    
    fig.savefig("test.png")
    plt.show()
    
    
# -----------------------------------------------------------------------------------------------------------------------------
# A Mapper transforms a value within an input interval into a value within an output interval
# The root Mapper doesn't transform the value but clips it within the given interval

class ClipMapper():
    def __init__(self, in_space= [0., 1.], clamp=True):
        self.in_space = in_space
        self.in_amp   = self.in_space[1]-self.in_space[0]
        self.clamp    = clamp
        self.in_sum   = self.in_space[0] + self.in_space[1]
        self.reversed = False

        if abs(self.in_amp) < 0.00001:
            raise WrapException(
                "Mapper initilization error",
                "Input space amplitude can not be nul: {}".format(in_space)
            )
            
    def clip(self, x):
        """Clip the value in the input space is clamp is True"""
        if self.clamp:
            return clip(x, self.in_space[0], self.in_space[1])
        else:
            return x
        
    def compute(self, x):
        """Maps the value
        
        By default, simply clip the given value"""
        return self.clip(x)

    def __call__(self, x):
        if self.reversed:
            return self.compute(self.in_sum - x)
        else:
            return self.compute(x)
    
    def plot(self, points=False, fcomp=None):
        """Plot the mapper using matplotlib"""
        
        marge = 0.0
        x0 = self.in_space[0] - self.in_amp*marge
        amp = self.in_amp*(1. + 2*marge)
        count = 500
        dx = amp/(count-1)
        xs = np.arange(x0, x0+amp, dx, dtype=np.float)
        ys = [self(x) for x in xs]
        
        fig, ax = plt.subplots()
        ax.plot(xs, ys)
        
        if fcomp is not None:
            ax.plot(xs, [fcomp(x) for x in xs])
            
        
        if points and hasattr(self, 'points'):
            pts = self.points[:, 0, 1]
            ax.plot(self.points[:, 0, 0], pts, 'o')
            ax.plot(self.points[:, 1, 0], self.points[:, 1, 1], '.')
            ax.plot(self.points[:, 2, 0], self.points[:, 2, 1], '.')
            #ax.plot(xs, [any_bezier_func(self.points, x) for x in xs])
        
        ax.set(xlabel='x', ylabel=type(self).__name__+'(x)',
               title='Mapper curve')
        ax.grid()
        
        fig.savefig("test.png")
        plt.show()

# -----------------------------------------------------------------------------------------------------------------------------
# A linear mapper between in and out space
        
class LinearMapper(ClipMapper):
    """Linear mapper which maps values from an input space to an output space"""
    
    def __init__(self, in_space=[0., 1.], out_space=[0., 1.], clamp=True):
        super().__init__(in_space, clamp)
        self.out_space = out_space
        self.in0       = self.in_space[0]
        self.out0      = self.out_space[0]

        out_amp        = self.out_space[1]-self.out_space[0]
        self.slope     = out_amp/self.in_amp

    def compute(self, x):
        return self.out0 + self.slope*(self.clip(x)-self.in0)

# -----------------------------------------------------------------------------------------------------------------------------
# A general mapper using a bezier function between the input and output spaces

class Mapper(ClipMapper):
    """The mapper uses a Bezier function between the input and the output spacess
    
    Use class methods to initialize a mapper with predefined functions
    
    The bezier function is venly computed. Make sure to initialize the mapper
    with locations points evenly spaced.
    
    Parameters
    ----------
    points: array like of shape (n, 3, 2)
        The Bezier curve control points
        
    clamp: boolean, default = True
        Clamp the value within the given spaces
    """
    def __init__(self, points, even=True, clamp=True):

        points   = check_bezier_func_points(points)
        in_space = [points[0, 0, 0], points[points.shape[0]-1, 0, 0]]

        super().__init__(in_space, clamp)

        self.points = points
        self.even   = even

    def compute(self, x):
        """Compute the value function with the even Bezier function"""
        if self.even:
            return even_bezier_func(self.points, self.clip(x))
        else:
            return bezier_func(self.points, self.clip(x))
        
    @classmethod
    def to_3D(self, perp='Y'):
        if perp == 'X':
            return np.insert(self.points, 0, 0., axis=1)
        elif perp == 'Y':
            return np.insert(self.points, 1, 0., axis=1)
        else:
            return np.insert(self.points, 2, 0., axis=1)
        
    
    # Class methods returning predefined mappers
    @classmethod
    def SCurve(Cls, x0, x1, y0, y1, clamp=True):
        """A Bezier S curve"""
        dx = (x1-x0)/3
        points = [
                    [[x0, y0], [x0-dx, y0], [x0+dx, y0]],
                    [[x1, y1], [x1-dx, y1], [x1+dx, y1]]
                ]
        return Mapper(points, clamp=clamp)
    
    @classmethod
    def FromFunction(Cls, f, x0, x1, count=10, clamp=True):
        """From a given function"""
        
        if count < 5:
            count = 5
        dx = (x1-x0)/(count-1)
        dt = dx/10.
        
        def derivative(x):
            return (f(x+dt) - f(x-dt))/2/dt
        
        points = []
        
        dx3 = dx/3
        for i in range(count):
            x  = x0 + i*dx
            y  = f(x)
            P  = [x, y]
            yp = derivative(x)
            L  = [x - dx3, y - yp*dx3]
            R  = [x + dx3, y + yp*dx3]
            points.append([P, L, R])
            
        return Mapper(points, clamp=clamp)
    
    # Damping
    @classmethod
    def ExpDamping(Cls, x0, x1, y0, y1, power=4, clamp=True):
        
        def f(x):
            return y1 + damping(x, x0, x1, y0-y1, power=power)
        
        return Mapper.FromFunction(f, x0, x1, count=20, clamp=clamp)
    
    # Damping
    @classmethod
    def PowDamping(Cls, x0, x1, y0, y1, power=4, clamp=True):
        
        def f(x):
            return y1 + damping(x, x0, x1, y0-y1, power=power)
        
        return Mapper.FromFunction(f, x0, x1, count=20, clamp=clamp)
    
    
    # Elastic
    @classmethod
    def Elastic(Cls, x0, x1, y0, y1, periods=3, power=3, clamp=True):
        
        w = 2*pi*periods/(x1-x0)
        amp = y0-y1
        phi = w*x0
        
        def f(x):
            return y1 + damping(x, x0, x1, amp, power=power)*cos(w*x - phi)
        
        count = periods*4+1
        return Mapper.FromFunction(f, x0, x1, count=count, clamp=clamp)
    
    @classmethod
    def Parabola(Cls, x0, x1, y0, y1, half=False, clamp=True):
        
        h = y0-y1
        
        if half:
            d = x1-x0
            
            P0 = [x0,      y1+h]
            L0 = [x0-d/3,  y1+h]
            R0 = [x0+d/3,  y1+h]
            
            P1 = [x1,      y1]
            L1 = [x1-d/3,  y1+h/1.5]
            R1 = [x1+d/3,  y1-h/1.5]
        else:
            d = (x1-x0)/2
            
            # By multiplying the vertical tangents by 2, the top is reached by
            # the Bezier computation
        
            P0 = [x0,     y1]
            L0 = [x0-d/3, y1 - h/1.5*2]
            R0 = [x0+d/3, y1 + h/1.5*2]
            
            P1 = [x1,     y1]
            L1 = [x1-d/3, y1 + h/1.5*2]
            R1 = [x1+d/3, y1 - h/1.5*2]
        
        return Mapper([P0, L0, R0, P1, L1, R1], clamp=clamp)
        

    # Bounces
    @classmethod
    def Bounces(Cls, x0, x1, y0, y1, bounces=3, clamp=True):
        
        # No bounce, just a parabolic fall
        
        if bounces < 1:
            return Cls.Parabola(x0, x1, y0, y1, half=True, clamp=clamp)
        
        # After the last bounce, the height must be a % of the initial height
        # a^(n+1) = 0.01
        a = 0.1**(1/(bounces+1))
        
        # The weights of the bounces decrease geometrically
        # The total length is given in parameter
        # l0/2 + l1 + L2 + ... ln = length
        # li = l0*a^i
        # l0(1/2 + a + a^2 + ... a^n) = length
        # l0.((1-a^(n+1))/(1-a) - 1/2) = 1ength
        length = x1-x0
        l0 = 2*length*(1-a)/(1 + a - 2*a**(bounces+1))
        
        # A bounce is parbola's top and right
        # Loop is initiated with previous point before the initial point
        
        x = -l0/2
        h = (y0-y1)
        l = l0
        
        # Initial fall: top to down
        P = [x0,       y0]
        L = [x0 - l/6, y0]
        R = [x0 + l/6, y0]
        points = [P, L, R]
        
        for i in range(bounces+1):
            
            x += l
            P = [x,     y1]
            L = [x-l/3, y1 + h/1.5*2]
            if i==0:
                L[1] = y1 + h/1.5
            
            l *= a
            h *= a*a
            R = [x+l/3, y1 + h/1.5*2]
            if i == bounces:
                R[1] = 0.
            
            points += [P, L, R]
            
        return Mapper(points, even=False, clamp=clamp)

def other_test():

    #Mapper.Bezier(0., 10., 3., 0., clamp=False).plot()      
    #Mapper.ExpDamping(0., 10., 3., 0., power=4, clamp=False).plot()      
    #Mapper.PowDamping(0., 10., 3., 0., power=4, clamp=False).plot()      
    #Mapper.Elastic(0., 10., 3., 0., periods=4, power=4, clamp=False).plot()      
    #Mapper.Parabola(-1, 1., 1., 0., half=False, clamp=False).plot(points=False, fcomp=lambda x: -x*x + 1)
    Mapper.Bounces(0., 20., 10., 0., bounces=5, clamp=False).plot(points=False)      
    
    #Mapper.FromFunction(lambda x:-x*x  + 1, -1., 1., count=20, clamp=False).plot(points=False)


