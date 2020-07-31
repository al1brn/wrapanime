#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 09:01:56 2020

@author: alain
"""

import numpy as np
from math import pi

# FOR DEV

import matplotlib.pyplot as plt

###

# =============================================================================================================================
# Useful constants

zero  = 1e-6
twopi = pi*2
hlfpi = pi/2

# =============================================================================================================================
# Easings canonic functions
# Parameter is betwwen 0 and 1. Return from 0 to 1

def f_constant(t):
    y = np.ones_like(t)
    try:
        y[np.where(t<1)[0]] = 1.
    except:
        return 0. if t < 1 else 1.
    return y
   
def f_linear(t):
    return t

def f_bezier(t):
    return f_linear(t)

def f_sine(t):
    return 1 - np.cos(t * hlfpi)
        
def f_quadratic(t):
    return t*t

def f_cubic(t):
    return t*t*t
        
def f_quartic(t):
    t2 = t*t
    return t2*t
        
def f_quintic(t):
    t2 = t*t
    t3 = t2*t
    return t3*t2
        
def f_exponential(t):
    return 1. - np.power(2, -10 * t)
    
def f_circular(t):
    return 1 - np.sqrt(1 - t*t)

def f_back(t, factor):
    return t*t*((factor + 1)*t - factor)
    
def f_elastic(t):
    amplitude    = 0.
    period       = 0.
    
    if period == 0:
        period = .3
        
    if (amplitude == 0) or (amplitude < 1.):
        amplitude = 1.
        s         = period/4
    else:
        s = period / twopi * np.sin(1/amplitude)
        
    t -= 1
    
    return -amplitude * np.power(2, 10*t) * np.sin((t-s)*twopi / period)

    
def f_bounce(t, a, xi, di, ci):
    """The number of bounces is the length of params.
    
    Let's name n the number of bounces after the half initial one.
    Each bounces is half of the previouse one. Let's note q the length
    of bounce 0 (the half one).
    The total length is L = q + q/2 + q/4 + ... -q/2
    Hence: L/q = (1-q/2^(n+1))/(1-1/2) - 1/2 = 3/2 - 1/2^n
    We want L = 1, hence 1/q = 3/2 - 1/2^n
    
    Let's note: d = q/2
    
    The equation of the initial parabola (half one) is: y = a(d^2 - x^2)
    At x= 0, y = 1, hence: a = 4/q^2
    
    Each xi is given by: xi = q(3/2-1/2^i)
    
    The parameters are the following:
        - a  : float -> used to compute the parabola a*(t^2 - ??)
        - xi : [0, q/2, q, ... 3/2 - 1/2^i ... 1]
        - di : [q/2, q/2,  ... xi+1 - xi]
        - ci : [0.,  3/2q, ... xi + di/2]
    
    These parameters are computed at initialization time
    
    NOTE that the ease in falls from right to left !
    The parameters must be initialized in consequence :-)
"""
    # Number of bounces
    
    n = len(di)

    # Duplicaton of the t for each of the intervals starting abscissa
    # NOTE: 1-t because the computation is made on falling from 1 to 0
    # but the default ease in is time reversed from 0 to 1
    
    ats = np.full((n, len(t)), 1-t).transpose()
    
    # Distances to the parabola centers
    y = ats - ci
    
    # Parabola equation
    # a and di are supposed to be correctly initialized :-)
    y = a*y*y + di
    
    # Return the max values (normally, only one positive value per line)
    
    return np.max(y, axis=1)


# =============================================================================================================================
#

class Interpolation():
    """Interpolation function from an interval towards another interval.
    
    Interpolated values are computed depending upon an interpolation code.
    
    Parameters
    ----------
    x0: float
        User min x value
    x1: float
        User max x value
    y0: float
        User min y value
    y1: float
        User max y value
    interpolation: str
        A valid code for interpolation in 
    mode: str
        A valid code for easing mode in [AUTO', 'EASE_IN', 'EASE_OUT', 'EASE_IN_OUT'
    """
    
    EASINGS = ['AUTO', 'EASE_IN', 'EASE_OUT', 'EASE_IN_OUT']
    
    INTERPS = {
        'CONSTANT'  : {'func': f_constant,    'auto': 1, 'tangents': [0, 0]},
        'LINEAR'    : {'func': f_linear,      'auto': 1, 'tangents': [1, 1]},
        'BEZIER'    : {'func': f_bezier,      'auto': 1, 'tangents': [0, 0]},
        'SINE'      : {'func': f_sine,        'auto': 1, 'tangents': [0, 1]},
        'QUAD'      : {'func': f_quadratic,   'auto': 1, 'tangents': [0, 1]},
        'CUBIC'     : {'func': f_cubic,       'auto': 1, 'tangents': [0, 1]},
        'QUART'     : {'func': f_quartic,     'auto': 1, 'tangents': [0, 1]},
        'QUINT'     : {'func': f_quintic,     'auto': 1, 'tangents': [0, 1]},
        'EXPO'      : {'func': f_exponential, 'auto': 1, 'tangents': [10*np.log(2), 0]},
        'CIRC'      : {'func': f_circular,    'auto': 1, 'tangents': [0, 0]},
        'BACK'      : {'func': f_back,        'auto': 1, 'tangents': [0, 0]},
        'BOUNCE'    : {'func': f_bounce,      'auto': 2, 'tangents': [0, 0]},
        'ELASTIC'   : {'func': f_elastic,     'auto': 2, 'tangents': [0, 0]},
        }

    def __init__(self, x0=0., x1=1., y0=0., y1=1., interpolation='BEZIER', easing='AUTO'):
        
        self.x0             = x0
        self.Dx             = x1-x0   # delta x
        self.y0             = y0      # 
        self.Dy             = y1-y0   # delta y
        
        # Interpolation
        self._interpolation = ""
        self.interpolation  = interpolation
        
        # Easing
        self._easing        = 0
        self.easing         = easing
        
    # ---------------------------------------------------------------------------
    # A user friendly representation
        
    def __repr__(self):
        return f"Easing({self.interpolation}) [{self.x0:.2f} {self.x0+self.Dx:.2f}] -> [{self.y0:.2f} {self.y0+self.Dy:.2f}]"

    # ---------------------------------------------------------------------------
    # Initialize from two Blender KeyFrame points

    @classmethod
    def FromKFPoints(cls, kf0, kf1):
        interp = Interpolation(
            kf0.co.x, kf1.co.x, kf0.co.y, kf1.co.y,
            interpolation=kf0.interpolation, easing=kf0.easing)
        interp.set_bpoint(1, kf0.handle_right)
        interp.set_bpoint(2, kf1.handle_left)
        
        interp.amplitude = kf0.amplitude
        interp.back      = kf0.back
        interp.period    = kf0.period
        interp.comp_bounces()
        return interp
        
    # ---------------------------------------------------------------------------
    # Initializers
    
    @classmethod
    def Constant(cls, x0=0., x1=1, y0=0., y1=1., easing='AUTO'):
        return Interpolation(x0, x1, y0, y1, 'CONSTANT', easing)
        
    @classmethod
    def Linear(cls, x0=0., x1=1, y0=0., y1=1., easing='AUTO'):
        return Interpolation(x0, x1, y0, y1, 'LINEAR', easing)

    @classmethod
    def Bezier(cls, x0=0., x1=1, y0=0., y1=1., easing='AUTO', P1=(1/3, 0.), P2=(2/3, 1.)):
        interp = Interpolation(x0, x1, y0, y1, 'BEZIER', easing)
        interp.set_bpoint(1, P1)
        interp.set_bpoint(2, P2)
        return interp
    
    @classmethod
    def Sine(cls, x0=0., x1=1, y0=0., y1=1., easing='AUTO'):
        return Interpolation(x0, x1, y0, y1, 'SINE', easing)
            
    @classmethod
    def Quadratic(cls, x0=0., x1=1, y0=0., y1=1., easing='AUTO'):
        return Interpolation(x0, x1, y0, y1, 'QUAD', easing)
            
    @classmethod
    def Cubic(cls, x0=0., x1=1, y0=0., y1=1., easing='AUTO'):
        return Interpolation(x0, x1, y0, y1, 'CUBIC', easing)
            
    @classmethod
    def Quartic(cls, x0=0., x1=1, y0=0., y1=1., easing='AUTO'):
        return Interpolation(x0, x1, y0, y1, 'QUART', easing)
            
    @classmethod
    def Quintic(cls, x0=0., x1=1, y0=0., y1=1., easing='AUTO'):
        return Interpolation(x0, x1, y0, y1, 'QUINT', easing)
            
    @classmethod
    def Exponential(cls, x0=0., x1=1, y0=0., y1=1., easing='AUTO'):
        return Interpolation(x0, x1, y0, y1, 'EXPO', easing)
        
    @classmethod
    def Circular(cls, x0=0., x1=1, y0=0., y1=1., easing='AUTO'):
        return Interpolation(x0, x1, y0, y1, 'CIRC', easing)
    
    @classmethod
    def Back(cls, x0=0., x1=1, y0=0., y1=1., easing='AUTO', factor=1.70158):
        interp = Interpolation(x0, x1, y0, y1, 'BACK', easing)
        interp.factor = factor
        return interp
    
    @classmethod
    def Bounce(cls, x0=0., x1=1, y0=0., y1=1., easing='AUTO', n=3):
        interp = Interpolation(x0, x1, y0, y1, 'BOUNCE', easing)
        interp.comp_bounces(n)
        return interp
        
    @classmethod
    def Elastic(cls, x0=0., x1=1, y0=0., y1=1., easing='AUTO'):
        return Interpolation(x0, x1, y0, y1, 'ELASTIC', easing)

    # ---------------------------------------------------------------------------
    # Interpolation property
    
    @property
    def interpolation(self):
        return self._interpolation
    
    @interpolation.setter
    def interpolation(self, value):
        if not value in self.INTERPS.keys():
            raise WrapException(
                f"Easing initialization error: invalid interpolation {value}.",
                f"Valid codes are {self.INTERPS.keys()}"
                )
            
        self._interpolation = value
        self._canonic  = self.INTERPS[value]['func']
        self._auto     = self.INTERPS[value]['auto']
        self._tangents = self.INTERPS[value]['tangents']
        
        # Specific Parameters 
        self.amplitude = 0.
        self.back      = 0.
        self.period    = 0.
        self.factor    = 1.70158
        
        self.comp_bounces()
        
        # bezier points
        self.init_bezier()
        
    # ---------------------------------------------------------------------------
    # Easing property
    
    @property
    def easing(self):
        return self._easing
        
    @easing.setter
    def easing(self, value):
        if not value in self.EASINGS:
            raise WrapException(
                f"Easing initialization error: invalid easing mode {value}. Valid modes are {self.EASINGS}"
                )
            
        self._easing = value
            
    # ---------------------------------------------------------------------------
    # Easing mode
    
    @property
    def easing_mode(self):
        return self._auto if self._easing == 'AUTO' else self._easing
            
    # ---------------------------------------------------------------------------
    # Canonic computation
    # Can be overriden for custom easings
        
    def canonic(self, t):
        if self.interpolation == 'BOUNCE':
            return f_bounce(t, self.a, self.xi, self.di, self.ci)
        elif self.interpolation == 'BACK':
            return f_back(t, self.factor)
        else:
            return self._canonic(t)
        
    # ---------------------------------------------------------------------------
    # Bezier points initialization
        
    def init_bezier(self, P1=(1/3, 0.), P2=(2/3, 1.)):

        self._bpoints       = np.zeros((4, 2), np.float)
        
        self.set_bpoint(0, (self.x0, self.y0))
        self.set_bpoint(1, P1)
        self.set_bpoint(2, P2)
        self.set_bpoint(3, (self.x0 + self.Dx, self.y0 + self.Dy))
                        
    def get_bpoint(self, index):
        return self._bpoints[index]

    def set_bpoint(self, index, P):
        self._bpoints[index] = np.array(P)
        
    # ---------------------------------------------------------------------------
    # Initialization specific to bounces
        
    def comp_bounces(self, n=3):
        
        r = 2 # Default
        
        # All but the first half one
        
        n      = min(7, max(0, n))
        
        qinv   = 1.5 - 1/r**n
        q      = 1/qinv
        xi     = np.array([q*(1.5 - 1/r**i) for i in range(n+1)])
        xi[-1] = 1
        di     = xi[1:] - xi[:-1]
        a      = 4*qinv*qinv

        self.a  = -a 
        self.xi = np.insert(xi[:-1], 0, 0)                # 0, q/2, q, ... 
        self.di = np.insert(a * di * di / 4, 0, 1)  # Parabola equation : a*x*x + di
        self.ci = np.insert(xi[:-1] + di/2, 0, 0)
        
    # ---------------------------------------------------------------------------
    # Interpolation
        
    def __call__(self, x):
        
        # Normalized the abscissa between 0 and 1
        ts = (np.array(x) - self.x0)/self.Dx
        
        # A single value
        single = len(ts.shape) == 0
        if single:
            ts = np.array([ts])
        
        # Points outside the interval
        i_inf = np.where(ts <= 0)[0]
        i_sup = np.where(ts >= 1)[0]
        
        # Abscissas exist outside
        outside = (len(i_inf) + len(i_sup)) > 0
        if outside:
            ys = np.empty_like(ts)
            
            ys[i_inf] = self.y0
            ys[i_sup] = self.y0 + self.Dy
            
            idx = np.delete(np.arange(len(ys)), np.concatenate((i_inf, i_sup)))
            
            t = ts[idx]
            
        else:
            t = ts
            
        # Compute on the required abscissa
        if self.interpolation == 'BEZIER':
            t2   = t*t
            t3   = t2*t
            umt  = 1-t
            umt2 = umt*umt
            umt3 = umt2*umt
            
            vals = umt3*self.x0 + umt2*t*self._bpoints[1, 0] + umt*t2*self._bpoints[2, 0] + t3*self.x1
            
        else:
            mode = self.easing_mode
    
            if mode == 'EASE_IN':
                vals = self.y0 + self.Dy*self.canonic(t)
            
            elif mode == 'EASE_OUT':
                vals = self.y0 + self.Dy*(1-self.canonic(1-t))
            
            else:
                t *= 2
                
                if len(t.shape) > 0:
                    y = np.empty_like(t)
                    
                    inf = np.where(t<=1)[0]
                    sup = np.delete(np.arange(len(t)), inf)
                    
                    y[inf] = self.canonic(t[inf])/2
                    y[sup] = 1 - self.canonic(2-t[sup])/2
                    
                    vals = self.y0 + self.Dy*y
                
                else:
                    if t <= 1:
                        vals = self.y0 + self.Dy*self.canonic(t)/2
                    else:
                        vals = self.y0 + self.Dy*(1 - self.canonic(2-t))/2
                    
        # The results
        if outside:
            ys[idx] = vals
        else:
            ys = vals
            
        # Single result
        if single:
            return ys[0]
        else:
            return ys
                
    # ---------------------------------------------------------------------------
    # Interval
        
    @property
    def x1(self):
        return self.x0 + self.Dx
    
    @property
    def y1(self):
        return self.y0 + self.Dy
    
    # ---------------------------------------------------------------------------
    # Tangents
                
    @property
    def left_tangent(self):
        if self.interpolation == 'BEZIER':
            dx = self._bpoints[1, 0] - self._bpoints[1, 0]
            dy = self._bpoints[1, 1] - self._bpoints[1, 1]
            if abs(dx) < 1e6:
                return 0.
            else:
                return dy/dx
        else:
            tg = self.Dy/self.Dx
            
            mode = self.easing_mode
            
            if mode == 'EASE_IN':
                return tg*self._tangents[0]
            elif mode == 'EASE_OUT':
                return tg*(1-self._tangents[0])
            if mode == 'EASE_IN_OUT':
                return tg*self._tangents[0]/2
            
        return 0.
        
    @property
    def right_tangent(self):
        if self.interpolation == 'BEZIER':
            dx = self._bpoints[3, 0] - self._bpoints[2, 0]
            dy = self._bpoints[3, 1] - self._bpoints[2, 1]
            if abs(dx) < 1e6:
                return 0.
            else:
                return dy/dx
        else:
            tg = self.Dy/self.Dx
            
            mode = self.easing_mode
            
            if mode == 'EASE_IN':
                return tg*self._tangents[1]
            elif mode == 'EASE_OUT':
                return tg*(1-self._tangents[1])
            if mode == 'EASE_IN_OUT':
                return tg*self._tangents[1]/2
            
        return 0.
    
    # ---------------------------------------------------------------------------
    # _plot for development
    
    def _plot(self, count=100, margin=0., fcomp=None):
        
        x0 = self.x0
        x1 = self.x0 + self.Dx
        amp = x1-x0
        
        x0 -= margin*amp
        x1 += margin*amp
        dx = (x1-x0)/(count-1)
        
        xs = np.arange(x0, x1+dx, dx, dtype=np.float)
        
        fig, ax = plt.subplots()
        
        def splot(mode):
            mmode = self.mode
            self.mode = mode
            ys = self(xs)
            self.mode = mmode
            
            ax.plot(xs, ys)
            
        #splot(0)
        splot(1)
        #splot(2)

        
        if fcomp is not None:
            ax.plot(xs, [fcomp(x) for x in xs])
        
        ax.set(xlabel='x', ylabel='easing',
               title=f"{self}")
        ax.grid()
        
        fig.savefig("test.png")
        plt.show()
    
# =============================================================================================================================
# A curve
        
class WFCurve():
    """A fcurve Blender compatible.
    
    The Fcurve is a series of successive interpolations. Each interpolation
    occupies an interval
    
    Parameters
    ----------
    bpoints: array(n, 3, 2) of float
        The bpoints of the fcurve
    params: 
        Parameters
    funcs
    modes
    """
    
    def __init__(self):
        self.interpolations = []
        self.extrapolation = 'CONSTANT'
        
    def __repr__(self):
        s = ""
        for interp in self.interpolations:
            s += f"{interp.x0:.2f} '{interp.interpolation}' "
        s = "[" + s + f"{self.x1:.2f}]"
            
        return f"WFCurve({len(self)})\n{s}"
    
    # ---------------------------------------------------------------------------
    # Initialize from a Blender fcurve
            
    def FromFCurve(self, fcurve):
        
        wfc = WFCurve()
        wfc.extrapolation = fcurve.extrapolation
        
        for i in range(len(fcurve.keyframe_points)-1):
            kf0 = fcurve.keyframe_points[i]
            kf1 = fcurve.keyframe_points[i+1]
            wfc.append(Interpolation.FromKFPoints(kf0, kf1))
            
        return wfc
    
    # ---------------------------------------------------------------------------
    # As an array of interpolations
    
    def __len__(self):
        return len(self.interpolations)
    
    def __getitem__(self, index):
        return self.interpolations[index]

    def __setitem__(self, index, value):
        self.interpolations[index] = value
    
    # ---------------------------------------------------------------------------
    # Append a new interpolation
    
    def append(self, interp):
        
        if len(self) == 0:
            self.interpolations = [interp]
            return interp
        
        if abs(interp.x0 - self.x1) > zero:
            raise WrapException(
                "WFCurve append error: the x0 of a new interpolation must equal the x1 to the last one.",
                f"WFCurve: {self}",
                f"Interpolation to insert: {interp}",
                f"WFCurve.x1 = {self.x1:.2f}, Interpolation.x0 = {interp.x0:.2f}"
                )
            
        interp.x0 = self.x1
        interp.y0 = self.y1
        self.interpolations.append(interp)
        
        return interp
    
        
    @property
    def x0(self):
        """Starting x value of the function."""
        if len(self) == 0:
            return 0.
        return self[0].x0
    
    @property
    def x1(self):
        """Ending x value of the function."""
        if len(self) == 0:
            return 1.
        return self[-1].x1
    
    @property
    def y0(self):
        """Starting y value of the function."""
        if len(self) == 0:
            return 0.
        return self[0].y0
    
    @property
    def y1(self):
        """Ending y value of the function."""
        if len(self) == 0:
            return 1.
        return self[-1].y1
    
    @property
    def deltas(self):
        return np.array([itp.Dx for itp in self.interpolations])
    
    @property
    def x0s(self):
        return np.array([itp.x0 for itp in self.interpolations])

    @property
    def x1s(self):
        return np.array([itp.x1 for itp in self.interpolations])

    # ====================================================================================================
    # Call
            
    def __call__(self, x):

        # ---------------------------------------------------------------------------
        # Empty curve
        
        if len(self) == 0:
            return np.array(x)
        
        # ---------------------------------------------------------------------------
        # A single value
        
        if np.array(x).size == 1:
            if x <= self.x0:
                return self.y0 + (x-self.x0)*self[0].left_tangent
            if x >= self.x1:
                return self.y1 + (x-self.x1)*self[-1].right_tangent
            
            for interp in self.interpolations:
                if interp.x0 + interp.Dx >= x:
                    return interp(x)
                
        # ---------------------------------------------------------------------------
        # Not that many values
        
        if len(x) < 100:
            return [self(X) for X in x]
        
        # ---------------------------------------------------------------------------
        # Vectorisation
        
        # Cyclic extrapolation
        if self.extrapolation == 'CYCLIC':
            xs = self.x0 + (np.array(x) - self.x0)/(self.x1 - self.x0)
        else:
            xs = np.array(x)
            
        # The resulting array
        ys = np.full(len(xs), 9., np.float)
        
        # ----- Points which are below the definition interval
        
        i_inf = np.where(xs <= self.x0)[0]
        ys[i_inf] = self.y0 + (xs[i_inf]-self.x0)*self.interpolations[0].left_tangent
        
        # ----- Points which are above the definition interval

        i_sup = np.where(xs >= self.x1)[0]
        ys[i_sup] = self.y1 + (xs[i_sup]-self.x1)*self.interpolations[-1].right_tangent

        # ----- Remaining points are within the definition interval
        
        idx = np.delete(np.arange(len(xs)), np.concatenate((i_inf, i_sup)))
        
        # Duplicaton of the xs for each of the bezier points
        axs = np.full((len(self), len(idx)), xs[idx]).transpose()

        # Deltas
        deltas = np.full((len(idx), len(self)), self.deltas)
        
        # Differences
        diffs = (axs - self.x0s)
        ix, tx = np.where(np.logical_and(np.greater_equal(diffs, 0), np.less(diffs, deltas)))

        # differences in a linear array (not useful here)
        # ts = diffs[ix, tx]
            
        # Array of the remaining x
        rem_x = xs[idx]
        
        # Let's loop on the easing to compute on the x which are in the interval
        # Note the this algorithm supposes that the number of easings is low
        # Compared to the number of x to compute
        
        interpolations = self.interpolations
        yints = np.full(len(idx), 8, np.float)
        for i in range(len(interpolations)):
            
            i_filter = np.where(tx==i)[0]
            vals = interpolations[i](rem_x[i_filter])
            
            yints[i_filter] = vals
            
        ys[idx] = yints
        
        return ys
    
    # ====================================================================================================
    # Call
    
    def _plot(self, count=100, margin=0., fcomp=None):
        
        x0 = self.x0
        x1 = self.x1
        amp = x1-x0
        
        x0 -= margin*amp
        x1 += margin*amp
        dx = (x1-x0)/(count-1)
        
        xs = np.arange(x0, x1+dx, dx, dtype=np.float)
        
        fig, ax = plt.subplots()
        ys = self(xs)
        ax.plot(xs, ys)
            
        if fcomp is not None:
            ax.plot(xs, [fcomp(x) for x in xs])
        
        ax.set(xlabel='x', ylabel='easing',
               title=f"{self}"[:60])
        ax.grid()
        
        fig.savefig("test.png")
        plt.show()
        
    
def test_c(count=10):
    
    interps = list(Interpolation.INTERPS.keys())
    easings = Interpolation.EASINGS
    
    wfc = WFCurve()
    x0 = 0.
    y0 = 0.
    y = np.random
    for i in range(count):
        x1 = x0 + 1
        y1 = y0 + (np.random.random_sample()-0.5)*2
        itp = Interpolation(
            x0, x1, y0, y1,
            interps[np.random.randint(len(interps))],
            easings[np.random.randint(len(easings))],
            )
        wfc.append(itp)
        x0 = x1
        y0 = y1
        
    print(wfc)
    wfc._plot(count=1000)

#test_c()
        
        
    