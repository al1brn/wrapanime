import inspect
import itertools
import numpy as np

from math import sin, cos, pi, tan, exp, log, sqrt

# =============================================================================================================================
# FOR DEBUG
# Import matplotlib which is not included in Blender

#"""

import matplotlib.pyplot as plt

WrapException = Exception

"""

from wrapanime.utils.errors import WrapException
    
"""

# ----------------------------s-------------------------------------------------------------------------------------------------
# Clip utility

def clip(v, v0, v1):
    """Clip the value within an interval"""
    return min(max(v0, v1), max(min(v0, v1), v))

# -----------------------------------------------------------------------------------------------------------------------------
# A general mapper using a bezier function between the input and output spaces

class BFunction():
    """A function implemented with a Bezier curve.
    
    The bezier points are stored by triplets in the intuitive order:
        - left control point (P3 or previous interval)
        - control point (P0 of next interval and P4 of previous interval)
        - right control point (P1 of next interval)
        
    The attribute bpoints is managed as a numpy array og shape (n, 6) floats.
    
    The BFunction is initialized with a valid bpoints array
    
    It can also be initialized with a xyt or xylr array.
    xylr contains lines of 4 floats:
        - xy for the central control point
        - l and r for the left and right tangent
    xyt is the same but with a single tangent for both sides
    
    The extrapolation can be: CONSTANT, LINEAR or CYLIC
    The class comes with predefined functions which can be initialized through
    some class methods.
    
    The class offers some function to manager the points:
        - scale along x or y
        - translate along x or y
        - invert to symetrized the function
        - insert a point at a given x value
        - cut to reduce the function to a given interval
        
    Two computation algorithms are used
        - compute_at(t): The standard cubic polynom (1-t)^3 + t(1-t)^2 + t^2(1-t) + t^3
          is applied once on the y values of the control points. t is considered as
          being the abscissa
        - dicho_compute(x): The cubic polynom is applied on the 4 control points. It returns a point
          with the y value for a given x wich generally differs from the parameter t.
          The operation is iterated by dichotomy to reach a point the abscissa of which
          is close to the parameter x.
        
    The first algorithm is used by default. The second one is used when a vertical tangent is used.
    
    CAUTION: when a vertical tangent is used, come utility functions can crash.
    
    Class attributes
    ----------------
    resolution : float
        The minimum distance between two x values. Two values closer that resolution are
        considered equal
    dx_der     : float
        Used to computhe the derivatives of a function. Must be less than resolution
    interpolations: array of strings
        Valide interpolations codes.
    
    Parameters
    ----------
    bpoints: array like of shape (n, 6)
        A vali Bezier curve of control points. n must be at least 2. The x values must be given
        in growing order.
    name: str
        Used in __repr__
    extrapolation: str, default 'CONSTANT'
        A valid extrapolation code: CONSTANT, LINEAR, CYCLIC
    """
    
    resolution = 0.003 # Minimum distance between abscissa (used in Tangent !)
    dx_der     = 0.001 # Used to compute derivative

    interpolations = [
        'LINEAR', 'SINUSOIDAL',
        'QUADRATIC', 'CUBIC', 'QUARTIC', 'QUINTIC',
        'EXPONENTIAL', 'CIRCULAR',
        'ELASTIC', 'BOUNCE']
    
    # =============================================================================================================================
    # Default initilization
    
    def __init__(self, bpoints=None, name="BFunction", extrapolation='CONSTANT'):
        
        self.bpoints       = BFunction.check_bpoints(bpoints)
        self.name          = name
        self.extrapolation = extrapolation
        self.constant      = False # Special behavior: y0 before x1, y1 after
        self.vert_tangents = False # To deal with vertical tangents
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # Check that a bpoints array is valid
        
    @classmethod
    def check_bpoints(cls, bpoints):
        """Check that a bpoint array is valid.
        
        Return a valid array or raise an exception if not possible.
        
        Parameters
        ----------
        bpoints: array with a size multiple of 6 and greater than 12
            The bpoints array to check
            
        Raise
        -----
        WrapException: if the parameter is not a valide bpoints array
        
        Returns
        -------
        array of floats of shape (n, 6)
            A valid bpoints array
        """
        
        a = np.array(bpoints)
        length, rem = divmod(a.size, 6)
        if rem != 0:
            cls._dump_bpoints(bpoints)
            raise WrapException(
                    "Bezier points array size not valid: it should be a multiple of 6 : 3 control 2D points",
                    f"Received array {a.shape} has size {a.size}"
                    )
        if length < 2:
            cls._dump_bpoints(bpoints)
            raise WrapException(
                    "Bezier points array size not valid: need at least two triplets of control points.",
                    f"Received array {a.shape} has only {length} triplet"
                    )
            
        a = a.reshape(length, 6)
        
        # ---- Check the x values are in growing order
        deltas = a[1:len(a)-1, 2] - a[0:len(a)-2, 2]
        
        if len(np.where(deltas<=0)[0]) > 0:
            cls._dump_bpoints(bpoints)
            raise WrapException(
                    "Bezier points array is not valid: The abscissa must be in growing order.",
                    f"Absicssa: {a[:, 2]} are not properly sorted.",
                    f"Deltas: {deltas}"
                    )
        # ok
        return a
    
    # Not sure it is usefull !
    def sort(self):
        self.bpoints = self.bpoints[np.argsort(self.bpoints[:, 2])]
        
    # =============================================================================================================================
    # Introspection

    def __repr__(self):
        return f"{self.name}({len(self)}) [{self.x0:.2f} {self.x1:.2f}]"
    
    @classmethod
    def _dump_bpoints(cls, bpoints):
        """Properly print a bpoints array (for debug)"""
        print("bpoints> number of lines:", len(bpoints))
        missed = True
        for i in range(len(bpoints)):
            p = bpoints[i]
            if (i < 10) or (i>len(bpoints)-10):
                print(f"{i:3d}> [{p[0]:6.2f} {p[1]:6.2f}] [{p[2]:6.2f} {p[3]:6.2f}] [{p[4]:6.2f} {p[5]:6.2f}]")
            else:
                if missed:
                    print("...")
                    missed = False

    @classmethod
    def _dump_xylr(cls, xylr):
        """Properly print an xylr array (for debug)"""
        print("xylr> Number of lines:", len(xylr))
        missed = True
        for i in range(len(xylr)):
            p = xylr[i]
            if (i < 10) or (i>len(xylr)-10):
                print(f"{i:3d}> [{p[0]:6.2f} {p[1]:6.2f}] l:{p[2]:6.2f} r:{p[3]:6.2f}]")
            else:
                if missed:
                    print("...")
                    missed = False

    @classmethod
    def _dump_xyt(cls, xyt):
        """Properly print an xyt array (for debug)"""
        print("xyt> number of lines:", len(xyt))
        missed = True
        for i in range(len(xyt)):
            p = xyt[i]
            if (i < 10) or (i>len(xyt)-10):
                print(f"{i:3d}> [{p[0]:6.2f} {p[1]:6.2f}] t:{p[2]:6.2f}")
            else:
                if missed:
                    print("...")
                    missed = False

    def _dump(self):
        print(f"Dump: {self}")
        print("-"*30)
        self._dump_bpoints(self.bpoints)
        print("-"*30)
        self._dump_xylr(self.xylr)
        
    # =============================================================================================================================
    # Fundamental initialization class methods
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Initialize from a xylr array : x, y, left tangent, right tangent

    @classmethod
    def FromXylr(cls, xylr, name="Xylr", extrapolation='CONSTANT'):
        """Initialize from an xylr array of floats.
        
        Xylr array gives left an right tangents for each point of the function.
        Controls points are computed with these tangents and the intervals
        between the abscissa
        
        Parameters
        ----------
        xylr: arrayf of float of shape (n, 4)
            The array to initialize the class
        extrapolation: str
            A valid extrapolation mode
            
        Returns
        -------
        BFunction
            A BFunction the controls points are computed with the xylr array.
        """
        
        # --------------------------------------------------
        # Sort the points

        xylr = np.array(xylr)
        length, rem = divmod(xylr.size, 4)
        if rem != 0:
            raise WrapException(
                    "BFunction FromXylr error: bad array size",
                    "The xylr parameter must be an array of shape (?, 4)."
                    f"The size of xylr is {xylr.size}."
                    )
            
        xylr = xylr.reshape(length, 4)
        
        # Sort
        xylr = xylr[np.argsort(xylr[:, 0])]
        
        # --------------------------------------------------
        # Remove the values wich are too close
        
        index = 1
        while index < len(xylr):
            if (xylr[index, 0] - xylr[index-1, 0]) < cls.resolution:
                xylr = np.delete(xylr, index, axis=0)
            else:
                index += 1
                
        if len(xylr) < 2:
            raise WrapException("BFunction FromXylr error: there are not enough points in the xylr array!")
                
        # --------------------------------------------------
        # Let's build the bezier points

        bpoints = np.empty(len(xylr)*6).reshape(len(xylr), 6)
        
        # (x, y) points
        bpoints[:, 2:4] = xylr[:, 0:2]
        
        # left and right deltas
        deltas = xylr[1:, 0] - xylr[:len(xylr)-1, 0]
        l_deltas = np.insert(deltas, 0,           deltas[0],             axis=0)
        r_deltas = np.insert(deltas, len(deltas), deltas[len(deltas)-1], axis=0)
        
        # left control points
        bpoints[:, 0] = xylr[:, 0] - l_deltas/3
        bpoints[:, 1] = xylr[:, 1] - l_deltas/3*xylr[:, 2]

        # right control points
        bpoints[:, 4] = xylr[:, 0] + r_deltas/3
        bpoints[:, 5] = xylr[:, 1] + r_deltas/3*xylr[:, 3]
            
        return cls(bpoints, name=name, extrapolation=extrapolation)
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Initialize from an xyt array : x, y, tangent
    
    @classmethod
    def FromXyt(cls, xyt, name="Xyt", extrapolation='CONSTANT'):
        """Initialize from an xyt array of floats.
        
        Xyt array gives the tangent for each point of the function.
        Controls points are computed with these tangents and the intervals
        between the abscissa
        
        Parameters
        ----------
        xyt: arrayf of float of shape (n, 3)
            The array to initialize the class
        extrapolation: str
            A valid extrapolation mode
            
        Returns
        -------
        BFunction
            A BFunction the controls points are computed with the xyt array.
        """
        
        xylr = np.insert(xyt, 3, xyt[:, 2], axis=1)
        return cls.FromXylr(xylr, name=name, extrapolation=extrapolation)
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Initialize from a function between two values
    
    @classmethod
    def FromFunction(cls, f, x0=0., x1=1., name=None, count=10, derivative=None, extrapolation='CONSTANT'):
        """Initialize from a function on an interval.
        
        The derivative of the function can be passed as a complementary argument.
        If not, the derivative is computed using the class attribute dx_der.
        
        Parameters
        ----------
        f: function of template f(float) -> float
            The function to used to build the Bezier curve
        x0: float
            Starting x value
        x1: float
            Ending x value
        name: str
            Name of the curve.
        count: int
            Number of points to compute
        derivative: function of template f(float) -> float
            The derivative to used to get the tangent at x abscissa
        extrapolation: str
            A valid extrapolation mode

        Returns
        -------
        BFunction
            A BFunction the controls points are computed from the function f.
        """
        
        # Control the number of points
        count = min(max(2, count), 10000)
        
        # Minimum distance
        x1 = max(x0 + cls.resolution, x1)
        
        # Prepare the resulting xylr array
        xylr = np.empty(count*4, np.float).reshape(count, 4)
        
        # Abscissa
        delta = (x1-x0)/(count-1)
        xylr[:, 0] = x0 + delta*np.arange(count)
        
        # Function values
        xylr[:, 1] = [f(x) for x in xylr[:, 0]]
        
        # If the derivative is not passe, it must be computed
        def comp_der(x):
            return (f(x+cls.dx_der) - f(x-cls.dx_der))*.5/cls.dx_der
        
        derivative = comp_der if derivative is None else derivative
        
        # Compute the derivatives
        xylr[:, 2] = [derivative(x) for x in xylr[:, 0]]
        xylr[:, 3] = xylr[:, 2].copy()
        
        # OK
        name = f.__name__ if name is None else name
        return cls.FromXylr(xylr, name=name, extrapolation=extrapolation)
    
    # =============================================================================================================================
    # Function initialization class methods
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Linear
    
    @classmethod
    def Linear(cls, x0=0., x1=1., y0=1., y1=1., extrapolation='CONSTANT'):
        """Linear interpolation.
        
        Parameters
        ----------
        x0: float
            Starting x value
        x1: float
            Ending x value
        y0: float
            Starting y value
        y1: float
            Ending y value
        extrapolation: str
            A valid extrapolation mode

        Returns
        -------
        BFunction
            A BFunction the controls points are computed with the xylr array.
        """

        xyt = np.empty(6, np.float).reshape(2, 3)
        xyt[0, 0] = x0
        xyt[1, 0] = x1
        xyt[0, 1] = y0
        xyt[1, 1] = y1
        try:
            xyt[0, 2] = (y1-y0)/(x1-x0)
        except:
            raise WrapException(
                    "BFunction Linear error: the arguments don't allow to compute a linear interpolation",
                    f"x values: x0={x0:.3f} to x1={x1:.3f}",
                    f"y values  y0={y0:.3f} to y1={y1:.3f}"
                    )
        
        xyt[1, 2] = xyt[0, 2]
        
        return cls.FromXyt(xyt, name="linear", extrapolation=extrapolation)
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Constant

    @classmethod
    def Constant(cls, x0=0., x1=1., y0=1., y1=1.):
        """Constant interpolation.
        
        Returns y0 if x < x1 else y1.
        It is not possible to approximate tjis constant interpolation with a bezier curve.
        The computataton is hooked with the contant attribute.
        
        Parameters
        ----------
        x0: float
            Starting x value
        x1: float
            Ending x value
        y0: float
            Starting y value
        y1: float
            Ending y value
        extrapolation: str
            A valid extrapolation mode

        Returns
        -------
        BFunction
        """
        
        bf = cls.Linear(x0, x1, y0, y1)
        bf.name = f"constant({y0:.2f} -> {y1:.2f})"
        bf.constant = True
        
        return bf
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Default Bezier S curve
    
    @classmethod
    def S(cls, x0=0., x1=1., y0=1., y1=1., extrapolation='CONSTANT'):
        """S curve interpolation.
        
        Parameters
        ----------
        x0: float
            Starting x value
        x1: float
            Ending x value
        y0: float
            Starting y value
        y1: float
            Ending y value
        extrapolation: str
            A valid extrapolation mode

        Returns
        -------
        BFunction
        """
        
        xyt = np.empty(6, np.float).reshape(2, 3)
        xyt[0, 0] = x0
        xyt[1, 0] = x1
        xyt[0, 0] = y0
        xyt[1, 0] = y1
        xyt[0, 2] = 0.
        xyt[1, 2] = 0.

        return cls.FromXyt(xyt, name="S curve", extrapolation=extrapolation)
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Circular

    @classmethod
    def Circular(cls, x0=0., x1=1., y0=1, y1=0., extrapolation='CONSTANT'):
        """Circular interpolation.
        
        The circular interpolation ends with a vertical tangent.
        It sets the attribute vert_tangents to True in order
        to use the dicho algorithm for the computations.
        
        Parameters
        ----------
        x0: float
            Starting x value
        x1: float
            Ending x value
        y0: float
            Starting y value
        y1: float
            Ending y value
        extrapolation: str
            A valid extrapolation mode

        Returns
        -------
        BFunction
        """
        
        # Build a vertical tangent
        tg = 0.55

        bp = np.empty(12, np.float).reshape(2, 6)
        bp[0, :] = [-tg,   1., 0., 1.,  tg,   1.]
        bp[1, :] = [  1., tg,  1., 0.,   1., -tg]
        bf = BFunction(bp, extrapolation='CONSTANT')
        bf.vert_tangents = True   # Use dicho_compute rather than compute_at
        bf.name = "circular"
        
        bf.scale_x(x1-x0)
        bf.translate_x(x0)
        
        bf.scale_y(y0-y1)
        bf.translate_y(y1)
        
        return bf
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # Sine curve
    
    @classmethod
    def Sine(cls, x0=0., x1=2*pi, y0=0., amp=1., omega=1., phi=0., extrapolation='CONSTANT'):
        """Sinusoidal interpolation.
        
        Computes y0 + amp*sin(omega*x + phi)
        
        Parameters
        ----------
        x0: float
            Starting x value
        x1: float
            Ending x value
        y0: float
            The central y value
        amp: float
            Sinusoidal curve amplitude
        omega: float
            Angular rotation speed
        phi: float
            Phase
        extrapolation: str
            A valid extrapolation mode

        Returns
        -------
        BFunction
        """

        # Canonical sine
        xyt = np.empty(15, np.float).reshape(5, 3)
        xyt[:, 0] = np.arange(5)*pi/2
        xyt[:, 1] = np.sin(xyt[:, 0])
        xyt[:, 2] = np.cos(xyt[:, 0])
        
        bf = cls.FromXyt(xyt, extrapolation=extrapolation)
        
        bf.cyclic_cut(x0, x1, omega=omega, phi=phi)
        
        bf.scale_y(amp)
        bf.translate_y(y0)
            
        # Name
        samp = '' if abs(amp-1.)   < 0.001 else f"{amp:.2f}*"
        some = '' if abs(omega-1.) < 0.001 else f"{omega:.2f}*"
        sphi = '' if abs(phi)      < 0.001 else f" + {phi:.2f}"
        bf.name = f"{samp}sin({some}x{sphi})"
            
        # Done
        return bf

    # -----------------------------------------------------------------------------------------------------------------------------
    # Cosine
    
    @classmethod
    def Cosine(cls, x0=0., x1=2*pi, y0=-1., amp=1., omega=1., phi=0., extrapolation='CONSTANT'):
        """Sinusoidal interpolation with cosine.
        
        Computes y0 + amp*cos(omega*x + phi)
        
        Parameters
        ----------
        x0: float
            Starting x value
        x1: float
            Ending x value
        y0: float
            The central y value
        amp: float
            Sinusoidal curve amplitude
        omega: float
            Angular rotation speed
        phi: float
            Phase
        extrapolation: str
            A valid extrapolation mode

        Returns
        -------
        BFunction
        """
        
        return cls.Sine(x0, x1, y0, amp, omega, phi=phi+pi/2)
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Tangent    
    
    @classmethod
    def Tangent(cls, x0, x1, omega=1., phi=0., extrapolation='CONSTANT'):
        """Tangent function interpolation.
        
        Computes tant(omega*x + phi)
        
        Parameters
        ----------
        x0: float
            Starting x value
        x1: float
            Ending x value
        omega: float
            Angular rotation speed
        phi: float
            Phase
        extrapolation: str
            A valid extrapolation mode

        Returns
        -------
        BFunction
        """
        
        if (x0 <= -pi/2 + cls.resolution) or (x1 >= pi/2 - cls.resolution):
            raise WrapException(
                "BFunction Tangent function error: tangent is only defined between -pi/2 and pi/2",
                f"Impossible to compute the tangent on the interval [{x0:.4f} {x1:.4f}]"
                )
        
        def f(x):
            return tan(omega*x + phi)
        
        def fp(x):
            t = f(x)
            return omega*(1+t*t)
        
        some = '' if abs(omega-1.) < 0.001 else f"{omega:.2f}*"
        sphi = '' if abs(phi)      < 0.001 else f" + {phi:.2f}"
        name = f"tan({some}x{sphi})"
        
        # The number of points depends upon the max y value
        
        ym = max(abs(f(x0)), abs(f(x1)))
        count = max(20, 2*int(ym))
        
        return cls.FromFunction(f, x0, x1, name=name, count=count, derivative=fp)
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Exponential
    
    @classmethod
    def Exponential(cls, x0=0., x1=1., y0=0., y1=1., extrapolation='CONSTANT'):
        """Exponential interpolation.
        
        Parameters
        ----------
        x0: float
            Starting x value
        x1: float
            Ending x value
        y0: float
            Starting y value
        y1: float
            Ending y value
        extrapolation: str
            A valid extrapolation mode

        Returns
        -------
        BFunction
        """
        
        bf = cls.FromFunction(exp, 0., 5., name="exp", count=10, extrapolation=extrapolation)
        
        bf.scale_x(x1-x0)
        bf.translate_x(x0 - bf.x0)
        
        bf.scale_y((y1-y0)/(bf.y1-bf.y0))
        bf.translate_y(y0 - bf.y0)
        
        return bf

    # -----------------------------------------------------------------------------------------------------------------------------
    # A utility returning a damping function which can be used in other computations
    
    @classmethod
    def exp_damping(cls, x0, x1, y0, y1):
        """A utility returning a damping function which can be used in other computations.
        
        Parameters
        ----------
        x0: float
            Starting x value
        x1: float
            Ending x value
        y0: float
            Starting y value
        y1: float
            Ending y value
            
        Returns
        -------
        function of template f(float) -> float
        """
        
        # X = x1-x0
        # Y = y0-y1
        # k = epsilon
        
        # a.e^(b.0) = Y
        # a.e^(b.X) = k
        # a = Y
        # b = log(k/a)/X
        
        try:
            a = y0-y1
            b = log(0.01/abs(a))/(x1-x0)
        except:
            raise WrapException(
                    "BFunction exp damping error: impossible to compute a demping with these parameters:",
                    f"x values: x0={x0:.3f} to x1={x1:.3f}",
                    f"y values  y0={y0:.3f} to y1={y1:.3f}"
                    )
        
        def damp(x):
            return a*exp(b*(x-x0))
        
        return damp
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Elastic
    
    @classmethod
    def Elastic(cls, x0=0., x1=1., y0=1., y1=0., periods=6):
        """Elastic interpolation.
        
        When y value varies from y0 to y1, it oscillates with a decreasing amplitude.
        The extrapolation is necessarily CONSTANT.
        
        Parameters
        ----------
        x0: float
            Starting x value
        x1: float
            Ending x value
        y0: float
            Starting y value
        y1: float
            Ending y value
        periods: int
            Number of oscillations between x0 and x1

        Returns
        -------
        BFunction
        """

        # The damping function
        damp = cls.exp_damping(x0, x1, y0, y1)
        
        # At least one period !
        periods = max(1, periods)
        
        # Angular velocity
        w = (periods+0.25)*2*pi/(x1-x0)
        
        def f(x):
            return cos(w*(x-x0))*damp(x)
        
        name = f"Elastic({periods})"
        bf = cls.FromFunction(f, x0, x1, name=name, count=periods*4+2, extrapolation='CONSTANT')

        # Ensure it finishes exactly on 0
        bf.bpoints[-1, 3] = 0.
        bf.bpoints[-1, 5] = 0.
        
        return bf
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Elastic
    
    @classmethod
    def Power(cls, x0=0., x1=1., y0=1., y1=0., n=2, extrapolation='CONSTANT'):

        """Power interpolation.
        
        Parameters
        ----------
        x0: float
            Starting x value
        x1: float
            Ending x value
        y0: float
            Starting y value
        y1: float
            Ending y value
        n: int
            Power exponent

        Returns
        -------
        BFunction
        """
        
        # X = x1-x0
        # Y = y0-y1
        
        # a + b.0^n = Y
        # a + b.X^n = 0
        
        # a = Y
        # b = -a/X^n
        
        a = y0 - y1
        try:
            b = -a/(x1-x0)**n
        except:
            raise WrapException(
                    "BFunction Power error: Impossible to compute a power interpolation with these parameters",
                    f"x-values: x0={x0:.3f} to x1={x1:.3f}",
                    f"y-values: y0={y0:.3f} to y1={y1:.3f}"
                    )
            
        def f(x):
            return a + b*(x-x0)**n
            
        def fp(x):
            return n*b*(x-x0)**(n-1)
            
            
        count = 2 + max(0, n-3)
        xyt = np.empty(count*3, np.float).reshape(count, 3)
        xyt[:, 0] = x0 + (x1-x0)/(count-1)*np.arange(count)
        xyt[:, 1] = [f(x)  for x in xyt[:, 0]]
        xyt[:, 2] = [fp(x) for x in xyt[:, 0]]
        
        return cls.FromXyt(xyt, name=f"power({n})", extrapolation=extrapolation)
            
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Bounces
    
    @classmethod
    def Bounces(cls, x0=0., x1=1., y0=1., y1=0., bounces=4):
        """Bounces interpolation.
        
        When y value varies from y0 to y1, it bounces several times with
        decreasing amplitudes.
        The extrapolation is necessarily CONSTANT.
        
        Parameters
        ----------
        x0: float
            Starting x value
        x1: float
            Ending x value
        y0: float
            Starting y value
        y1: float
            Ending y value
        bounces: int
            Number of bounces between x0 and x1

        Returns
        -------
        BFunction
        """
        
        # One bounce occurs on the horizontal length delta
        # The equation is :
        # y = a.x - bx^2 
        # 0 at x=0 and x=delta --> 0 = a - b.delta
        # b = a/delta
        # y = a(x -x^2/delta)
        #
        # The value a gives the shape of parabola
        #
        # The tangents are
        # yp = a(1 - 2x/delta)
        # yp(0) = a
        # yp(delta) = -a
        #
        # The height of the parabola is at delta/2
        # h = a(delta - delta/4) = 3/4.a.delta
        #
        # The initial height is y0-y1
        # a = 4/3(y0-y1)/delta0
        #
        # 
        
        # No bounce, just a parabolic fall
        if bounces < 1:
            return cls.Power(x0, x1, y0, y1, n=2)
        
        # Computes 
        # After n bounces, the height must be neglictible
        # eps = q**(bounces+1)
        eps = 0.01
        q = eps**(1/(bounces+1))
        
        a = 4
        
        # Let's build a canonical bounces + 1 bounces
        count = bounces + 2
        xylr = np.empty(count*4, np.float).reshape(count, 4)
        xylr[0,  0] = 0.
        xylr[1:, 0] = [x for x in itertools.accumulate([q**i for i in range(count-1)])]
        xylr[:,  1] = 0.
        # tangents
        xylr[:,  2] = [-a*q**(i-1) for i in range(count)]
        xylr[:,  3] = [ a*q**i     for i in range(count)]
        
        # Adjust last rigth tangent to zero
        xylr[count-1, 3] = 0.
        
        # Insert the top of the first bounce (the true initial point)
        xylr = np.insert(xylr, 1, [0.5, 1., 0., 0.],axis=0)
        
        # Remove the initial point
        xylr = np.delete(xylr, 0, axis=0)
        
        # We've got our bounces
        bf = cls.FromXylr(xylr)
        
        # Let's scale and translate x
        bf.scale_x((x1-x0)/(bf.x1-bf.x0))
        bf.translate_x(x0 - bf.x0)
        
        # Y scale and translation
        bf.scale_y(y0-y1)
        bf.translate_y(y1)
        
        # Done
        return bf
    
    # =============================================================================================================================
    # Interpolation by code

    @classmethod
    def Interpolation(cls, x0=0., x1=1., y0=1., y1=0., interpolation='S', invert=False, extrapolation='CONSTANT'):
        """Interpolation Bezier function is returned by it Blender code.
        
        Parameters
        ----------
        x0: float
            Starting x value
        x1: float
            Ending x value
        y0: float
            Starting y value
        y1: float
            Ending y value
        interpolation: str
            A valid Blender interpolation code
        invert:
            Invert the curve
        extrapolation: str
            A valid extrapolation mode

        Returns
        -------
        BFunction
        """

        if interpolation == 'LINEAR':
            bf = cls.Linear(x0, x1, y0, y1, extrapolation=extrapolation)
        elif interpolation == 'SINUSOIDAL':
            omega = pi/(x1-x0)/2
            phi  = -omega*x0
            bf = cls.Sine(x0, x1, y0, y1-y0, omega=omega, phi=phi, extrapolation=extrapolation)
        elif interpolation == 'QUADRATIC':
            bf = cls.Power(x0, x1, y0, y1, n=2, extrapolation=extrapolation)
        elif interpolation == 'CUBIC':
            bf = cls.Power(x0, x1, y0, y1, n=3, extrapolation=extrapolation)
        elif interpolation == 'QUARTIC':
            bf = cls.Power(x0, x1, y0, y1, n=4, extrapolation=extrapolation)
        elif interpolation == 'QUINTIC':
            bf = cls.Power(x0, x1, y0, y1, n=5, extrapolation=extrapolation)
        elif interpolation == 'EXPONENTIAL':
            bf = cls.Exponential(x0, x1, y0, y1, extrapolation=extrapolation)
        elif interpolation == 'CIRCULAR':
            bf = cls.Circular(x0, x1, y0, y1)
            
        elif interpolation == 'ELASTIC':
            bf = cls.Elastic(x0, x1, y0, y1, periods=6)
        elif interpolation == 'BOUNCE':
            bf = cls.Bounces(x0, x1, y0, y1, bounces=8)
            
        else:
            raise WrapException(f"BFunction Interpolation code '{interpolation}' is unknown")

        if invert:
            bf.inverse()
            
        return bf
    
    # =============================================================================================================================
    # Array implementation
    
    def __len__(self):
        return len(self.bpoints)
    
    def __getitem__(self, index):
        return self.bpoints[index]
    
    def __setitem__(self, index, value):
        self.bpoints[index] = np.array(value).reshape(6)
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # Index of a value

    def prev_index(self, x):
        """Returns the index the right interval of which he value belongs.
        
        If the argument is close to an indexed point (< resolution), its  index is returned.
        
        Parameters
        ----------
        x: float
            The x-value to handle
        
        Returns
        -------
        int of None
            - point[index] <= x < point[index+1] (with resolution)
            - None if x < x0
        """

        if abs(x-self.x0) < self.resolution:
            return 0
    
        if x < self.x0:
            return None
        
        for i in range(1, self.bpoints.shape[0]):
            
            if abs(x-self.x(i)) < self.resolution:
                return i
            
            if x < self.x(i):
                return i-1
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # X, y and derivatives array
    
    @property
    def xylr(self):
        """Returns the control points as 4-uples: x, t, left and right tangents"""
        
        if self.vert_tangents:
            raise WrapException(
                "BFunction xylr error: impossible to compute tangents when the function has vertical tangents",
                f"{self}"
                )
        
        res = np.empty(len(self)*4, np.float).reshape(len(self), 4)
        a = self.bpoints
        
        res[:, 0:2] = a[:, 2:4]
        res[:, 2]   = (a[:, 1] - a[:, 3])/(a[:, 0] - a[:, 2])
        res[:, 3]   = (a[:, 5] - a[:, 3])/(a[:, 4] - a[:, 2])
        
        return res
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Repeat the points along the x axis
    
    def repeat(self, count=1):
        """Repeat the points along the x axis.
        
        The last point is supposed to be at the same y than the first one.
        The points from 1 to len(self)-1 are cloned along the x axis with
        a translation of (x1-x0).
        
        Parameters
        ----------
        count: int
            Number of times to repeat
        """
        
        block = BFunction(self.bpoints)
        delta = self.x1 - self.x0
        new_length = len(self) + count*(len(self)-1)
        a = np.resize(self.bpoints, new_length*6).reshape(new_length, 6)
        for i in range(1, count+1):
            block.translate_x(delta)
            a[1 + i*(len(self)-1): 1 + (i+1)*(len(self)-1), :] = block.bpoints[1:]
            
        self.bpoints = a.reshape(new_length, 6)
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # Merge two function
    # Don't know when it could be useful;, just for fun
        
    def merge(self, bpoints):
        """Merge with another array of bpoints."""
        
        xylr0 = self.xylr
        xylr1 = BFunction(bpoints).xylr
        xylr  = np.append(xylr0, xylr1).reshape(len(xylr0)+len(xylr1), 4)
        self.bpoints = BFunction.FromXylr(xylr).bpoints
    
    
    # =============================================================================================================================
    # Reading and setting
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Access to the control points
    
    
    def left(self, index):
        """Left control point at index."""
        return self.bpoints[index, 0:2]
        
    def point(self, index):
        """Central control point at index."""
        return self.bpoints[index, 2:4]
    
    def right(self, index):
        """Right control point at index."""
        return self.bpoints[index, 4:6]

    def set_left(self, index, bpoints):
        """Set the value of the left control point at index."""
        self.bpoints[index, 0:2] = np.array(bpoints).reshape(2)
        
    def set_point(self, index, bpoints):
        """Set the value of the central control point at index."""
        self.bpoints[index, 2:4] = np.array(bpoints).reshape(2)
    
    def set_right(self, index, bpoints):
        """Set the value of the right control point at index."""
        self.bpoints[index, 4:6] = np.array(bpoints).reshape(2)
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # Access to the x and y values
    
    def x(self, index):
        """x value of the central point at index."""
        return self.bpoints[index, 2]
    
    @property
    def x0(self):
        """Starting x value of the function."""
        return self.bpoints[0, 2]
    
    @property
    def x1(self):
        """Ending x value of the function."""
        return self.bpoints[-1, 2]
    
    # y interval
    def y(self, index):
        """y value of the central point at index."""
        return self.bpoints[index, 3]
    
    @property
    def y0(self):
        """Starting y value of the function."""
        return self.bpoints[0, 3]
    
    @property
    def y1(self):
        """Ending y value of the function."""
        return self.bpoints[-1, 3] 
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Intervals are named deltas
    
    @property
    def left_deltas(self):
        """Returns an array with the left intervals.
        
        The left interval of the first index is equal to its right interval.
        """
        deltas = self.bpoints[1:, 2] - self.bpoints[:len(self)-1, 2]
        return np.insert(deltas, 0, deltas[0], axis=0)

    @property
    def right_deltas(self):
        """Returns an array with the right intervals.
        
        The right interval of the last index is equal to its left interval.
        """
        deltas = self.bpoints[1:, 2] - self.bpoints[:len(self)-1, 2]
        return np.insert(deltas, len(deltas), deltas[len(deltas)-1], axis=0)
    
    def left_delta(self, index):
        """The left interval at index."""
        if index == 0:
            return self.bpoints[1, 2] - self.bpoints[0, 2]
        
        return self.bpoints[index, 2] - self.bpoints[index-1, 2]

    def right_delta(self, index):
        """The right interval at index."""
        if index == len(self)-1:
            return self.bpoints[len(self)-1, 2] - self.bpoints[len(self)-2, 2]
        
        return self.bpoints[index+1, 2] - self.bpoints[index, 2]
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Tangents
    
    def left_tangent(self, index):
        """Returns the left tangent at an index."""
        P = (self.left(index)-self.point(index))
        return P[1]/P[0]
    
    def right_tangent(self, index):
        """Returns the right tangent at an index."""
        P = (self.right(index)-self.point(index))
        return P[1]/P[0]
    
    def set_left_tangent(self, index, value):
        """Set the left tangent."""
        delta = self.left_delta(index)
        self.bpoints[index, 0] = self.bpoints[index, 2] - delta/3
        self.bpoints[index, 1] = self.bpoints[index, 3] - delta/3*value

    def set_right_tangent(self, index, value):
        """Set the right tangent."""
        delta = self.right_delta(index)
        self.bpoints[index, 4] = self.bpoints[index, 2] + delta/3
        self.bpoints[index, 5] = self.bpoints[index, 3] + delta/3*value
        
    def set_tangent(self, index, value):
        """Set the tangent at a given index."""
        self.set_left_tangent(index, value)
        self.set_right_tangent(index, value)
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Smooth vs angular points
    # is smooth
    
    def is_smooth(self, index):
        """Returns True if the points is smooth, ie not angular."""
        
        A = self.right(index) - self.left(index)
        D = self.point(index) - self.left(index)
        try:
            K = D/A
        except:
            return False
        return abs(K[1]-K[0]) < 0.001
    
    @property
    def smoothes(self):
        """Returns the smoothness of control points."""
        return [self.is_smooth(i) for i in range(len(self))]
    
    
    # =============================================================================================================================
    # Utility methods

    # -----------------------------------------------------------------------------------------------------------------------------
    # Scale the intervals
    
    # Adjust derivatives
    def scale_left(self, index, scale):
        """Adjust the left control point of a given scale.
        
        Can be used with the left interval is modified.
        """
        L = self.left(index)
        P = self.point(index)
        self.set_left(P + (L-P)*scale)

    def scale_right(self, index, scale):
        """Adjust the right control point of a given scale.
        
        Can be used with the right interval is modified.
        """
        R = self.right(index)
        P = self.point(index)
        self.set_right(P + (R-P)*scale)
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # Scale along x
    
    def scale_x(self, scale):
        """Applies a scale factor along the x axis.
        
        If the scale is negative, the order of the points are reversed
        """
        
        length = len(self)
        a = self.bpoints.reshape(length*3, 2)
        a[:, 0] *= scale
        
        # Points must reordered if negative
        if scale < 0:
            a = np.flip(a, axis=0)
        
        self.bpoints = a.reshape(length, 6)

        # Points must reordered if negative
        if scale < 0 and False:
            self.bpoints = np.flip(self.bpoints, axis=0)
            for i in range(len(self)):
                P = self.bpoints[i, 0:2].copy()
                self.bpoints[i, 0:2] = self.bpoints[i, 4:6]
                self.bpoints[i, 4:6] = P
            

    def scale_y(self, scale):
        """Applies a scale factor along the y axis."""

        length = len(self)
        a = self.bpoints.reshape(length*3, 2)
        a[:, 1] *= scale
        self.bpoints = a.reshape(length, 6)

    def translate_x(self, delta):
        """Applies a translation along the x axis."""
        
        length = len(self)
        a = self.bpoints.reshape(length*3, 2)
        a[:, 0] += delta
        self.bpoints = a.reshape(length, 6)

    def translate_y(self, delta):
        """Applies a translation along the y axis."""

        length = len(self)
        a = self.bpoints.reshape(length*3, 2)
        a[:, 1] += delta
        self.bpoints = a.reshape(length, 6)
        
    def inverse(self):
        """Inverse the function.
        
        Inversion changes the effect from left to right.
        This is equivalent to invert ease out and in in Blender.
        
        Inversion consists in applying -1 scale on x and y axis.
        Then, the function is replaced by translation in the
        initial rectangle: x0, x1, y0, y1.
        """
        
        x0 = self.x0
        y0 = self.bpoints[0, 3]
        
        self.scale_x(-1)
        self.translate_x(x0-self.x0)
        
        self.scale_y(-1)
        self.translate_y(y0-self.y0)
        
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Inert a point at a given abscissa
    
    def insert_x(self, x):
        """Insert a new point at the abscissa x.
        
        If the abscissa is close (close than resolution) to an existing point, nothing happens.
        
        When the new point is inserted, the interval where it is inserted is split in two smaller
        intervals. It is necessary to adjust the left and right control points of the neighbours.
        
        Parameters
        ----------
        x: float
            The abscissa where to inserr the new point.
        """
        
        xylr = self.xylr
        
        # Look for the insertion index
        index = len(xylr)
        for i in range(len(xylr)):
            
            # Close to an existing one
            if abs(x - xylr[i, 0]) < self.resolution:
                return
            
            # Stop if below the current x value
            if x < xylr[i, 0]:
                index = i
                
        # Insert the point
        tg = self.derivative(x)
        xylr = np.insert(xylr, index, [x, self(x), tg, tg], axis=0)
        
        # build the new array of bpoints
        bf = BFunction.FromXylr(xylr)
        
        self.bpoints= bf.bpoints

    # -----------------------------------------------------------------------------------------------------------------------------
    # Cut into an interval

    def cut(self, x0, x1):
        """Cut the function to a new interval.
        
        The new interval can be anywhere. The y values are computed
        using the extrapolation mode.
        
        When y value varies from y0 to y1, it bounces several times with
        Parameters
        ----------
        x0: float
            Starting x value
        x1: float
            Ending x value
        """
        
        # Insert the two points
        self.insert_x(x0)
        self.insert_x(x1)
        
        i0 = self.prev_index(x0)
        i1 = self.prev_index(x1)+1
        
        if i0 == 0:
            nbf = BFunction(np.array(self.bpoints))
        else:
            nbf = BFunction(np.delete(self.bpoints, np.arange(i0), axis=0))
            
        i1 -= i0
        if i1 < len(nbf.bpoints):
            nbf.bpoints = np.delete(nbf.bpoints, np.arange(i1, len(nbf.bpoints)), axis=0)
            
        self.bpoints = nbf.bpoints
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # Cylic curve cutting
    
    def cyclic_cut(self, x0, x1, omega=1., phi=0.):
        """Cut a periodical curve at a given interval.
        
        The Function is initialized in its canonical on one perio (typically 2*pi).
        This function is scale to fit with the required omega
        It is translated to take the phase into account.
        This part is reproduced in order to cover the required interval
        Then the result function is cut at the interval.
        
        Parameters
        ----------
        x0: float
            Starting x value
        x1: float
            Ending x value
        omega: float
            Angular rotation speed
        phi: float
            Phase

        Returns
        -------
        BFunction
        """
        
        base_period = self.x1 - self.x0
        
        # Takes phase into account
        x_trans = -phi/omega
        x0 -= x_trans
        x1 -= x_trans
        
        # Period
        period = base_period/omega
        
        self.scale_x(1/omega)
        
        # Starts before x0
        per_count, rem = divmod(x0, period)
        self.translate_x(per_count*period)
        
        # end after x1
        rep = int((x1-x0)/period)+1
        #print(period, per_count, rem, rep)
        self.repeat(rep)
        
        # Cut between x0 and x1
        self.cut(x0, x1)
        
        # Phase
        self.translate_x(x_trans)
                    
    # =============================================================================================================================
    # Computation
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Compute within an interval
    
    def compute_at(self, index, t):
        """Compute the Bezier value within an interval.
        
        Naming L the left point of the interval and R the right one, 
        the algorithm supposes that x is close to (1-t)L + tR.
        This is not exactly true.
        
        Depending of the curve, the number of control points must be increased
        to gain in precision. For instance this is the case with the tangent curve.
        
        An alternative to increase the number of control points, the alternative
        algorithm can be used : dicho_compute.
        
        Parameters
        ----------
        index: int
            The index of the left point of the interval. Must be <= len(self)-2
        t: float
            The computation parameters:  0 <= t <= 1
            
        Returns
        -------
        float
            The y value
        """
            
        t2 = t*t
        t3 = t2*t
    
        umt = 1.-t
        umt2 = umt*umt
        umt3 = umt2*umt
        
        a = self.bpoints.reshape(self.bpoints.size)
        i = index*6 + 3
    
        return umt3*a[i] + 3*t*umt2*a[i+2] + 3*t2*umt*a[i+4] + t3*a[i+6]
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Compute the coordinates of the point within an interval
    
    def compute_P(self, index, t):
        """Compute the Bezier point within an interval.
        
        Parameters
        ----------
        index: int
            The index of the left point of the interval. Must be <= len(self)-2
        t: float
            The computation parameters:  0 <= t <= 1
            
        Returns
        -------
        array of 2 floats
            The coordinates of the point on the curve
        """
        
        t2 = t*t
        t3 = t2*t
    
        umt = 1.-t
        umt2 = umt*umt
        umt3 = umt2*umt
        
        a = self.bpoints.reshape(self.bpoints.size//2, 2)
        i = index*3
        
        return umt3*a[i+1] + 3*t*umt2*a[i+2] + 3*t2*umt*a[i+3] + t3*a[i+4]
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Dichotomic algorithm
    
    def dicho_compute(self, index, x):
        """Compute the Bezier value for agiven abscissa.
        
        The x value is within the interval starting at the given index.
        
        The Bezier parameter t is computed such as the abscissa of the
        computed point is equal to the argument x.
        
        This allows to managed vertical tangents.
        
        Parameters
        ----------
        index: int
            The index of the left point of the interval. Must be <= len(self)-2
        x: float
            A value within the interval starting at index.
            
        Returns
        -------
        float
            The y value
        """
        
        t0 = 0.
        t1 = 1.
        for i in range(15):
            t = (t0 + t1)/2
            P = self.compute_P(index, t)
            if abs(P[0]-x) < self.resolution:
                return P[1]
            if P[0] < x:
                t0 = t
            else:
                t1 = t
        return P[1]
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Call implementation

    def __call__(self, x):
        """Returns the function value at a given abscissa.
        
        The function manages the extrapolation of the function outside its definition interval.
        
        Within its definition interval, the function looks for the interval the value belongs to.
        One of the two computation algorithm is then called.
        
        Note that a ugly patch manages the "constant" behavior.
        
        Parameters
        ----------
        x: float
            The input of the function
            
        Returns
        -------
        float
            The output of the function
        """
        
        x0 = self.x0
        x1 = self.x1
        
        # Cyclic extrapolation
        if self.extrapolation == 'CYCLIC':
            if x < x0 or x > x1:
                x = x0 + ((x-x0) % (x1-x0))
                
        # Constant patch
        if self.constant:
            if x < x1:
                return self.bpoints[0, 3]
            else:
                return self.bpoints[1, 3]
        
        # Extrapolation before
        if x <= x0:
            if self.extrapolation == 'CONSTANT'  or self.vert_tangents:
                return self.y0
            
            tg = self.left_tangent(0)
            return self.y0 + tg*(x-self.x0)
            
        # Extrapolation after
        if x >= x1:
            if self.extrapolation == 'CONSTANT' or self.vert_tangents:
                return self.y1
            
            tg = self.right_tangent(len(self)-1)
            return self.y1 + tg*(x-self.x1)
            
        
        # In the middle, let's find the interval where is x
        for i in range(1, self.bpoints.shape[0]):
            if x < self.bpoints[i, 2]:
                index = i-1
                break
            
        # Algorithm depends upon if vertical tangents exist
        if self.vert_tangents:
            return self.dicho_compute(index, x)
        else:
            return self.compute_at(index, (x-self.bpoints[index, 2])/self.right_delta(index))
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # Derivative
    
    def derivative(self, x):
        return (self(x + self.dx_der) - self(x - self.dx_der))*0.5/self.dx_der
        
    
    # =============================================================================================================================
    # Plot the function

    def _plot(self, count=100, margin=0., tangents=False, points=False, fcomp=None):
        x0 = self.x0
        x1 = self.x1
        amp = x1-x0
        x0 -= margin*amp
        x1 += margin*amp
        dx = (x1-x0)/(count-1)
        
        xs = np.arange(x0, x1+dx, dx, dtype=np.float)
        #ys = [self(x) for x in xs]
        ys = self.comp(xs)

        Ps = []
        for i in range(0, len(self)-1):
            t_count = 20
            for j in range(t_count):
                t = j/(t_count-1)
                Ps.append(self.compute_P(i, t))
        
        fig, ax = plt.subplots()
        
        if tangents:
            for pts in self:
                ax.plot([pts[0], pts[2], pts[4]], [pts[1], pts[3], pts[5]], '-o', color='black')
        
        ax.plot(xs, ys)
        
        if fcomp is not None:
            ax.plot(xs, [fcomp(x) for x in xs])
            
        if False:
            xp = [P[0] for P in Ps]
            yp = [P[1] for P in Ps]
            ax.plot(xp, yp, color='red')
            
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
    # Compute the bezier values for an array 
        
    def comp(self, x):
        
        # Cyclic extrapolation
        if self.extrapolation == 'CYCLIC':
            xs = self.x0 + (np.array(x) - self.x0)/(self.x1 - self.x0)
        else:
            xs = np.array(x)
        ys = np.empty(len(xs), np.float)
            
        # Constant patch
        if self.constant:
            return np.where(xs < self.x1, self.y0, self.y1)
        
        if self.vert_tangents:
            return np.array([self(x) for x in xs])
        
        # Points below 
        i_inf = np.where(xs <= self.x0)[0]
        ys[i_inf] = (xs[i_inf]-self.x0)*self.left_tangent(0)
        
        # Points above
        i_sup = np.where(xs >= self.x1)[0]
        ys[i_sup] = (xs[i_sup]-self.x1)*self.right_tangent(len(self)-1)   
        
        # Remaining points
        idx = np.delete(np.arange(len(xs)), np.append(i_inf, i_sup))
        
        # Duplicaton of the xs for each of the bezier points
        axs = np.full((len(self), len(idx)), xs[idx]).transpose()
        
        # Deltas
        deltas = np.full((len(idx), len(self)), self.right_deltas)
        
        # Differences
        diffs = (axs-self.bpoints[:, 2]) / deltas

        ix, tx = np.where(np.logical_and(np.greater_equal(diffs, 0), np.less(diffs, 1)))

        # differences in a linear array
        ts = diffs[ix, tx]
        
        vals = self.compute_at(tx, ts)
        
        ys[idx] = vals
        
        return ys
    
    # -----------------------------------------------------------------------------------------------------------------------------
    # Dichotomic algorithm
    
    def dicho_comp(self, index, x):
        """Compute the Bezier value for a given abscissa.
        
        The x value is within the interval starting at the given index.
        
        The Bezier parameter t is computed such as the abscissa of the
        computed point is equal to the argument x.
        
        This allows to managed vertical tangents.
        
        Parameters
        ----------
        index: int
            The index of the left point of the interval. Must be <= len(self)-2
        x: float
            A value within the interval starting at index.
            
        Returns
        -------
        float
            The y value
        """
        
        xs = np.array(x)
        
        t0 = np.zeros(len(xs))
        t1 = np.ones(len(xs))
        ix = np.arange(len(xs))
        
        for i in range(15):
            t = (t0[ix] + t1[ix])/2
            P = self.compute_P(index[ix], t[ix])
            
            if abs(P[0]-x) < self.resolution:
                return P[1]
            if P[0] < x:
                t0 = t
            else:
                t1 = t
        return P[1]
    
    
        
        
        

# =============================================================================================================================
# Let's do some tests    
        
def test():
    
    def bfs(x0=0., x1=1., y0=0., y1=1., amp=1., omega=1., phi=0., extrapolation='CONSTANT'):
        """
            def Linear(cls, x0=0., x1=1., y0=1., y1=1., extrapolation='CONSTANT'):
            def Constant(cls, x0=0., x1=1., y0=1., y1=1.):
            def S(cls, x0=0., x1=1., y0=1., y1=1., extrapolation='CONSTANT'):
            def Circular(cls, x0=0., x1=1., y0=1, y1=0., extrapolation='CONSTANT'):
            def Sine(cls, x0=0., x1=2*pi, y0=0., amp=1., omega=1., phi=0., extrapolation='CONSTANT'):
            def Cosine(cls, x0=0., x1=2*pi, y0=-1., amp=1., omega=1., phi=0., extrapolation='CONSTANT'):
            def Tangent(cls, x0, x1, omega=1., phi=0., extrapolation='CONSTANT'):
            def Exponential(cls, x0=0., x1=1., y0=0., y1=1., extrapolation='CONSTANT'):
            def Elastic(cls, x0=0., x1=1., y0=1., y1=0., periods=6):
            def Power(cls, x0=0., x1=1., y0=1., y1=0., n=2, extrapolation='CONSTANT'):
            def Bounces(cls, x0=0., x1=1., y0=1., y1=0., bounces=4):
        """

        yield BFunction.Linear(x0, x1, y0, y1, extrapolation=extrapolation)
        yield BFunction.Constant(x0, x1, y0, y1)
        yield BFunction.S(x0, x1, y0, y1, extrapolation=extrapolation)
        yield BFunction.Circular(x0, x1, y0, y1, extrapolation=extrapolation)
        
        yield BFunction.Sine(x0, x1, y0, amp, omega, phi, extrapolation=extrapolation)
        yield BFunction.Cosine(x0, x1, y0, amp, omega, phi, extrapolation=extrapolation)
        yield BFunction.Tangent(x0, x1, omega, phi, extrapolation=extrapolation)
        
        yield BFunction.Exponential(x0, x1, y0, y1, extrapolation=extrapolation)
        #yield BFunction.Elastic(x0, x1, y0, y1)
        #yield BFunction.Power(x0, x1, y0, y1, extrapolation=extrapolation)
        #yield BFunction.Bounces(x0, x1, y0, y1)
        
    invert   = False
    points   = False
    tangents = False
    x0 = 0.
    x1 = 1.
    y0 = 1.
    y1 = 0.
    amp   = 1.
    omega = pi
    phi   = 0.
    extrap = 'CONSTANT'
    margin = 0.
    cut = False
    
    for bf in itertools.chain(
            bfs(x0, x1, y0, y1, amp, omega, phi, extrapolation=extrap),
            [BFunction.Power(x0, x1, y0, y1, n, extrapolation=extrap) for n in range(1,7)],
            [BFunction.Elastic(x0, x1, y0, y1, periods=2*i) for i in range(4)],
            [BFunction.Bounces(x0, x1, y0, y1, bounces=2*i) for i in range(4)],
            ):

        if invert:
            bf.inverse()
            
        if extrap == 'CYCLIC':
            bf.extrapolation = extrap
            
        if cut:
            if not bf.vert_tangents:
                bf.cut(-2., 2.)
            
        if not bf.vert_tangents:
            bf.insert_x(0.33)
            
        bf._plot(margin=margin, count=1000, tangents=tangents, points=points, fcomp=None)
    
#test()    
            
            
            
        
        
class temp():   
        
    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, index):
        return self.points[index]

    def compute(self, x):
        """Compute the value function with the even Bezier function"""
        if self.even:
            return even_bezier_func(self.points, self.clip(x))
        else:
            return bezier_func(self.points, self.clip(x))
        
    def resize_x(self, x0, x1):
        scale = (x1-x0)/(self.in_space[1] - self.in_space[0])
        self.points *= scale
        self.points += (x0-self.in_space[0])
        self.in_space = [self.points[0, 0, 0], self.points[-1, 0, 0]]
        
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
    if points.size % 6 != 0:
        raise WrapException(
            "Bezier function initialization error",
            f"Control points must be given by triplets. The size of resulting array must be a multiple of 6. {points.size} is not a multiple of 6"
            )
    np.reshape(points, (points.size//6, 3, 2))
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


def other_test():

    #Mapper.Bezier(0., 10., 3., 0., clamp=False).plot()      
    #Mapper.ExpDamping(0., 10., 3., 0., power=4, clamp=False).plot()      
    #Mapper.PowDamping(0., 10., 3., 0., power=4, clamp=False).plot()      
    #Mapper.Elastic(0., 10., 3., 0., periods=4, power=4, clamp=False).plot()      
    #Mapper.Parabola(-1, 1., 1., 0., half=False, clamp=False).plot(points=False, fcomp=lambda x: -x*x + 1)
    Mapper.Bounces(0., 20., 10., 0., bounces=5, clamp=False).plot(points=False)      
    
    #Mapper.FromFunction(lambda x:-x*x  + 1, -1., 1., count=20, clamp=False).plot(points=False)


