#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 09:01:56 2020

@author: alain
"""

import numpy as np
from math import pi

import matplotlib.pyplot as plt


twopi = pi*2
hlfpi = pi/2

# =============================================================================================================================
# The canoninc easing function
# Parameter is betwwen 0 and 1. Return from 0 to 1
#
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
    

"""
CAUTION: OUT
			if ((t/=d) < (1/2.75)) {
				return c*(7.5625*t*t) + b;
			} else if (t < (2/2.75)) {
				return c*(7.5625*(t-=(1.5/2.75))*t + .75) + b;
			} else if (t < (2.5/2.75)) {
				return c*(7.5625*(t-=(2.25/2.75))*t + .9375) + b;
			} else {
				return c*(7.5625*(t-=(2.625/2.75))*t + .984375) + b;
			}
"""
         
class Easing():
    """Generic easing.
    
    Override canonic for personalized easing functions
    
    Modes are stored as int
    - -1    : AUTO
    -  0    : IN
    -  1    : OUT
    -  other: INOUT
    """
    
    EASINGS = ['AUTO', 'EASE_IN', 'EASE_OUT', 'EASE_IN_OUT']
    CANONICS   = {
        'CONSTANT'  : f_constant,
        'LINEAR'    : f_linear,
        'BEZIER'    : f_bezier,
        'SINE'      : f_sine,
        'QUAD'      : f_quadratic,
        'CUBIC'     : f_cubic,
        'QUART'     : f_quartic,
        'QUINT'     : f_quintic,
        'EXPO'      : f_exponential,
        'CIRC'      : f_circular,
        'BACK'      : f_back,
        'BOUNCE'    : f_bounce,
        'ELASTIC'   : f_elastic
        }
    TANGENTS = {
        'CONSTANT'  : [0, 0],
        'LINEAR'    : [1, 1],
        'BEZIER'    : [0, 0],
        'SINE'      : [0, 1],
        'QUAD'      : [0, 1],
        'CUBIC'     : [0, 1],
        'QUART'     : [0, 1],
        'QUINT'     : [0, 1],
        'EXPO'      : [10*np.log(2), 0],
        'CIRC'      : [0, 0],
        'BACK'      : [0, 0],
        'BOUNCE'    : [0, 0],
        'ELASTIC'   : [0, 0]
            }

    def __init__(self, x0=0., x1=1., y0=0., y1=1., func='LINEAR', mode=0):
        self.x0     = x0
        self.Dx     = x1-x0   # d
        self.y0     = y0      # b
        self.Dy     = y1-y0   # c
        self.easing = mode
        
        self.amplitude = 0.
        self.back      = 0.
        self.period    = 0.
        self.factor    = 1.70158  # ?
        
        if type(func) is str:
            self.code = func
            self._canonic = self.CANONICS[func]
            self._tangents = self.TANGENTS[func]
        else:
            self.code = 'CUSTOM'
            self._canonic = func
            self._tangents = [0, 0]
            
        self.comp_bounces()
            
    def __repr__(self):
        return f"Easing({self.code}) [{self.x0:.2f} {self.x0+self.Dx:.2f}] -> [{self.y0:.2f} {self.y0+self.Dy:.2f}]"
        
    @property
    def easing(self):
        return self.EASINGS[self.mode]
        
    @easing.setter
    def easing(self, value):
        if type(value) is int:
            self.mode = max(-1, min(2, value))
        else:
            self.mode = self.EASINGS.index(value)-1
        
    def canonic(self, t):
        if self.code == 'BOUNCE':
            return f_bounce(t, self.a, self.xi, self.di, self.ci)
        else:
            return self._canonic(t)
        
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
        
        
    def __call__(self, x):
        
        # Normalized abscissa between 0 and 1
        ts = (np.array(x) - self.x0)/self.Dx
        
        # A single value
        single = type(ts) is not np.ndarray
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
            
            idx = np.delete(np.arange(len(ys)), np.append(i_inf, i_sup))
            
            t = ts[idx]
            
        else:
            t = ts
            
        # Compute on the required abscissa

        if self.mode == 0:
            vals = self.y0 + self.Dy*self.canonic(t)
        
        elif self.mode == 1:
            vals = self.y0 + self.Dy*(1-self.canonic(1-t))
        
        else:
            t *= 2
            
            if hasattr(t, '__len__'):
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
            
        # Fake array !
            
        if single:
            return ys[0]
        else:
            return ys
                
                
    @property
    def left_tangent(self):
        tg = self.Dy/self.Dx
        if self.mode == 0:
            return tg*self._tangents[0]
        elif self.mode == 1:
            return tg*(1-self._tangents[0])
        if self.mode == 2:
            return tg*self._tangents[0]/2
            
        return 0.
        
    @property
    def right_tangent(self):
        tg = self.Dy/self.Dx
        if self.mode == 0:
            return tg*self._tangents[1]
        elif self.mode == 1:
            return tg*(1-self._tangents[1])
        if self.mode == 2:
            return tg*self._tangents[1]/2
            
        return 0.
    
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
    
    
class Linear(Easing):
    pass    

class Constant(Easing):
    def canonic(self, t):
        y = np.ones_like(t)
        y[np.where(t<1)[0]] = 0
        return y
    
class Sine(Easing):
    def canonic(self, t):
        return 1 - np.cos(t * hlfpi)
        
class Quadratic(Easing):
    def canonic(self, t):
        return t*t
        
class Cubic(Easing):
    def canonic(self, t):
        return t*t*t
        
class Quartic(Easing):
    def canonic(self, t):
        t2 = t*t
        return t2*t
        
class Quintic(Easing):
    def canonic(self, t):
        t2 = t*t
        t3 = t2*t
        return t3*t2
        
class Exponential(Easing):
    def canonic(self, t):
        y = 1. - np.power(2, -10 * t)
        y[np.where(t==0)[0]] = 0.
        y[np.where(t==1)[0]] = 1.
        return y
    
class Circular(Easing):
    def canonic(self, t):
        return 1 - np.sqrt(1 - t*t)

class Back(Easing):
    def canonic(self, t):
        return t*t*((self.factor + 1)*t - self.factor)
    
class Elastic(Easing):
    def canonic_DEV(self, t):
        amp    = 0.4
        period = 0.3
        
        y = -amp*np.power(2, 10*(t-1))*np.sin((t-self.factor)*twopi/period)
        
        if hasattr(y, '__len__'):
            y[np.where(t==0)[0]] = 0
            y[np.where(t==1)[0]] = 1
        else:
            if t == 0:
                return 0.
            if t == 1:
                return 1.
            
        return y
    
class Bounce(Easing):
    pass
        
            

#if (t==0) return b
#if ((t/=d)==1) return b+c;
"""
if (!p) p=d*.3;
if (!a || a < Math.abs(c)) { a=c; s=p/4; }:
    
    
else s = p/PI_M2 * Math.asin (c/a);
return -(a*Math.pow(2,10*(t-=1)) * Math.sin( (t*d-s)*PI_M2/p )) + b;
"""    
    
        
def test_e():     
    e = Easing(y0=1, y1=0, func='BOUNCE', mode='EASE_IN')
    e._plot(count=1000)
    print(e(-0.5))
    print(e(0.))
    print(e(0.5))
    print(e(1.))
    print(e(1.5))
    

test_e()

"""
		/*
		Elastic
		---------------------------------------------------------------------------------
		*/
		public static function easeInElastic (t:Number, b:Number, c:Number, d:Number, a:Number=undefined, p:Number=undefined):Number
		{
			var s:Number;
			if (t==0) return b;  if ((t/=d)==1) return b+c;  if (!p) p=d*.3;
			if (!a || a < Math.abs(c)) { a=c; s=p/4; }
			else s = p/PI_M2 * Math.asin (c/a);
			return -(a*Math.pow(2,10*(t-=1)) * Math.sin( (t*d-s)*PI_M2/p )) + b;
		}
		public static function easeOutElastic (t:Number, b:Number, c:Number, d:Number, a:Number=undefined, p:Number=undefined):Number
		{
			var s:Number;
			if (t==0) return b;  if ((t/=d)==1) return b+c;  if (!p) p=d*.3;
			if (!a || a < Math.abs(c)) { a=c; s=p/4; }
			else s = p/PI_M2 * Math.asin (c/a);
			return (a*Math.pow(2,-10*t) * Math.sin( (t*d-s)*PI_M2/p ) + c + b);
		}
		public static function easeInOutElastic (t:Number, b:Number, c:Number, d:Number, a:Number=undefined, p:Number=undefined):Number
		{
			var s:Number;
			if (t==0) return b;  if ((t/=d/2)==2) return b+c;  if (!p) p=d*(.3*1.5);
			if (!a || a < Math.abs(c)) { a=c; s=p/4; }
			else s = p/PI_M2 * Math.asin (c/a);
			if (t < 1) return -.5*(a*Math.pow(2,10*(t-=1)) * Math.sin( (t*d-s)*PI_M2/p )) + b;
			return a*Math.pow(2,-10*(t-=1)) * Math.sin( (t*d-s)*PI_M2/p )*.5 + c + b;
		}


		/*
		Bounce
		---------------------------------------------------------------------------------
		*/
		public static function easeInBounce (t:Number, b:Number, c:Number, d:Number):Number
		{
			return c - easeOutBounce (d-t, 0, c, d) + b;
		}
		public static function easeOutBounce (t:Number, b:Number, c:Number, d:Number):Number
		{
			if ((t/=d) < (1/2.75)) {
				return c*(7.5625*t*t) + b;
			} else if (t < (2/2.75)) {
				return c*(7.5625*(t-=(1.5/2.75))*t + .75) + b;
			} else if (t < (2.5/2.75)) {
				return c*(7.5625*(t-=(2.25/2.75))*t + .9375) + b;
			} else {
				return c*(7.5625*(t-=(2.625/2.75))*t + .984375) + b;
			}
		}
		public static function easeInOutBounce (t:Number, b:Number, c:Number, d:Number):Number
		{
			if (t < d/2) return easeInBounce (t*2, 0, c, d) * .5 + b;
			else return easeOutBounce (t*2-d, 0, c, d) * .5 + c*.5 + b;
		}
        
"""

class WFCurve():
    
    def __init__(self, bpoints, params, funcs, modes):
        
        # ----- Bpoints
        
        a = np.array(bpoints)
        
        count = a.size // 6
        self.bpoints = a.reshape(count, 6)
        
        # ----- Params
        
        count -= 1
        self.params = np.array(params).reshape(count, 3)
        
        # ----- Array of easings

        self.easings = []
        
        a = self.bpoints
        for i in range(count):
            easing = Easing(a[i, 2], a[i+1, 2], a[i, 3], a[i+1, 3], func=funcs[i], mode=modes[i])
            self.easings.append(easing)
            
        self.extrapolation = 'CONSTANT'
        
    def __repr__(self):
        s = ""
        for easing in self.easings:
            s += f"{easing.x0:.2f} '{easing.code}' "
        s = "[" + s + f"{self.x1:.2f}]"
            
        return f"WFCurve({len(self.bpoints)})\n{s}"
            
    def FromFCurve(self, fcurve):
        
        # --- Get the points

        count = len(fcurve.keyframe_points)
        bpoints = np.empty(count*6, np.float).reshape(count, 6)

        vx = np.empty(2*count, np.float)
        
        fcurve.keyframe_points.foreach_get('co', vx)
        bpoints[:, 2:4] = vx.reshape(count, 2)
        
        fcurve.keyframe_points.foreach_get('handle_left', vx)
        bpoints[:, 0:2] = vx.reshape(count, 2)
        
        fcurve.keyframe_points.foreach_get('handle_right', vx)
        bpoints[:, 4:6] = vx.reshape(count, 2)
        
        # --- Get the parameters
        vx = np.empty(count, np.float)
        params = np.empty(count*3, np.float).reshape(count, 3)
        
        fcurve.keyframe_points.foreach_get('amplitude', vx)
        params[:, 0] = vx.copy()
        
        fcurve.keyframe_points.foreach_get('back', vx)
        params[:, 1] = vx.copy()
        
        fcurve.keyframe_points.foreach_get('period', vx)
        params[:, 2] = vx.copy()
        
        # ----- Interpolation and modes
        
        interps = []
        modes   = []
        for kf in fcurve.keyframe_points:
            interps.append(kf.interpolation)
            modes.append(kf.easing)
            
        return WFCurve(bpoints, params, interps, modes)
            
        
    @property
    def x0(self):
        """Starting x value of the function."""
        return self.bpoints[0, 2]
    
    @property
    def x1(self):
        """Ending x value of the function."""
        return self.bpoints[-1, 2]
    
    @property
    def y0(self):
        """Starting y value of the function."""
        return self.bpoints[0, 3]
    
    @property
    def y1(self):
        """Ending y value of the function."""
        return self.bpoints[-1, 3]
    
    @property
    def deltas(self):
        return self.bpoints[1:, 2] - self.bpoints[:len(self.bpoints)-1, 2]
    
    # ====================================================================================================
    # Array implementation
    
    def __len__(self):
        return len(self.easings)
    
    def __getitem__(self, index):
        return self.easings[index]
    
    # ====================================================================================================
    # Call
            
    def __call__(self, x):
        
        # ---------------------------------------------------------------------------
        # A single value
        
        if np.array(x).size == 1:
            if x <= self.x0:
                return self.y0 + (x-self.x0)*self.easings[0].left_tangent
            if x >= self.x1:
                return self.y1 + (x-self.x1)*self.easings[-1].right_tangent
            
            for easing in self.easings:
                if easing.x0 + easing.Dx >= x:
                    return easing(x)
                
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
        ys[i_inf] = self.y0 + (xs[i_inf]-self.x0)*self.easings[0].left_tangent
        
        # ----- Points which are above the definition interval

        i_sup = np.where(xs >= self.x1)[0]
        ys[i_sup] = self.y1 + (xs[i_sup]-self.x1)*self.easings[-1].right_tangent

        # ----- Remaining points are within the definition interval
        
        idx = np.delete(np.arange(len(xs)), np.append(i_inf, i_sup))
        
        # Duplicaton of the xs for each of the bezier points
        axs = np.full((len(self), len(idx)), xs[idx]).transpose()

        # Deltas
        deltas = np.full((len(idx), len(self)), self.deltas)
        
        # Differences
        if False:
            diffs = (axs-self.bpoints[:len(self.bpoints)-1, 2]) / deltas
            ix, tx = np.where(np.logical_and(np.greater_equal(diffs, 0), np.less(diffs, 1)))
        else:
            diffs = (axs-self.bpoints[:len(self.bpoints)-1, 2])
            ix, tx = np.where(np.logical_and(np.greater_equal(diffs, 0), np.less(diffs, deltas)))

        # differences in a linear array (not useful here)
        # ts = diffs[ix, tx]
            
        # Array of the remaining x
        rem_x = xs[idx]
        
        # Let's loop on the easing to compute on the x which are in the interval
        # Note the this algorithm supposes that the number of easings is low
        # Compared to the number of x to compute
        
        easings = self.easings
        yints = np.full(len(idx), 8, np.float)
        for i in range(len(easings)):
            
            i_filter = np.where(tx==i)[0]
            vals = easings[i](rem_x[i_filter])
            
            yints[i_filter] = vals
            
        ys[idx] = yints
        
        return ys
    
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
               title=f"{self}")
        ax.grid()
        
        fig.savefig("test.png")
        plt.show()
    
def test_c():
    
    count = 10
    a = np.zeros(count*6, np.float).reshape(count, 6)
    for i in range(count):
        a[i, 2] = i
    a[:, 3] = np.random.random_sample(count)*3.
        
    count -= 1
    
    params = np.zeros(count*3).reshape(count, 3)
    
    valids = [k for k in Easing.CANONICS.keys()]
    interps = [valids[np.random.randint(0, len(valids))] for i in range(count)]
    
    modes   = [Easing.EASINGS[np.random.randint(0, 2)] for i in range(count)]
    
        
    c = WFCurve(a, params, interps, modes)
    c._plot(count=10)

#test_c()
        
    
            
    