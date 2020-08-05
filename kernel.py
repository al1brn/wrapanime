#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 15:08:42 2020

@author: alain
"""

import time
import inspect
import numpy as np
import random
import itertools

import bpy

from wrapanime.utils.errors import WrapException


# ******************************************************************************************************************************************************
# ******************************************************************************************************************************************************
# Time management
# ******************************************************************************************************************************************************
# ******************************************************************************************************************************************************
    
def get_frame(frame_or_str, scene=None):
    
    if type(frame_or_str) is str:

        if scene is None:
            scene = bpy.context.scene

        markers = scene.timeline_markers

        mrk = markers.get(frame_or_str)
        if mrk is None:
            raise WrapException("Time line marker unknown", f"The marker '{frame_or_str}' doesn't exist")
        return mrk.frame

    return frame_or_str


class Time():
    def __init__(self, frame=None, scene=None):
        self.scene = bpy.context.scene if scene is None else scene
        self.frame = self.scene.frame_current_final if frame is None else get_frame(frame, scene)
        self.fps   = self.scene.render.fps
        self.time  = self.frame/self.fps
        
        self.marker = None
        for m in self.scene.timeline_markers:
            if m.frame == self.frame:
                self.marker = m.name
                
    def __repr__(self):
        return f"Time(frame={self.frame}, time={self.time:.2f}, marker={self.marker})"
    
    @classmethod
    def Current(cls, scene):
        return cls(None, scene)
    
    @classmethod
    def Time(cls, t):
        scene = bpy.context.scene
        fps = scene.render.fps
        frame = int(round(t*fps))
        return cls(frame, scene)
    
    
class TimeInterval():
    def __init__(self, frame0, frame1):
        self.t0     = Time(frame0)
        self.t1     = Time(frame1)
        self.amp    = self.t1.frame - self.t0.frame
        self.length = self.amp + 1
        if self.amp < 1:
            raise WrapException(
                    "TimeInterval initialization error: interval can't be negative",
                    f"[{self.t0} to {self.t1}] is not valid"
                    )
            
        
    def __repr__(self):
        return f"TimeInterval [{self.t0} - {self.t1}]"
    
    @staticmethod
    def FullAnimation():
        scene = bpy.context.scene
        return TimeInterval(scene.frame_start, scene.frame_end)
    
    def contains(self, frame):
        frame = get_frame(frame)
        return (frame >= self.t0.frame) and (frame <= self.t1.frame)
    
    def progress(self, frame):
        frame = get_frame(frame)
        if frame <= self.t0.frame:
            return 0.
        if frame >= self.t1.frame:
            return 1.
        
        return (frame-self.t0.frame)/self.amp
    
    def milestones(self, count, sigma=0., min_space = 1):
        if count < 2:
            raise WrapException(
                    f"TimeInterval.milestones error: impossible to define less than 2 milestones (count={count} required)."
                    )
        if count*min_space > self.length:
            raise WrapException(
                    f"TimeInterval.milestones error: impossible to create {count} milestones with minimum space {min_space} in total length of {self.length}"
                    )

        # Generate random spaces
        spaces = np.array([random.gauss(1., sigma) for i in range(count-1)])
        
        # Normalize to int values. Sum is arounf total length
        spaces = np.round(spaces*self.length/sum(spaces))
        
        # Minimum space
        for i in range(len(spaces)):
            spaces[i] = max(min_space, spaces[i])
            
        # Add 1 randomly until total length is ok
        while sum(spaces) < self.length:
            i = random.randint(0, count-1)
            spaces[i] += 1
            
        # Substract randomly 1 until total length is ok
        while sum(spaces) > self.length:
            i = random.randint(0, count-1)
            if spaces[i] > min_space:
                spaces[i] -= 1
                
        # Let's return the milestones
        return [self.t0.frame] + [self.t0.frame + f for f in itertools.accumulate(spaces)]
    
    def split(self, count, sigma=0.):
        # Min length is 2
        if count > self.amp:
            raise WrapException(
                    "TimeInterval error: impossible to split in sub intervals",
                    f"Length of {self} is {self.lengh}. Impossible to split in {count} sub intervals"
                    )
            
        ms = self.milestones(count+1, sigma, min_space=1)
        return [TimeInterval(ms[i], ms[i+1]) for i in range(count)]
        
    
    def overlaps(self, count, length, sigma=0.):
        if length > self.length:
            raise WrapException(
                    f"TimeInterval.overlaps errors: impossible to create {count} overlapped sub intervals."
                    f"Length of {self} is {self.lengh}. Impossible to create overlapped sub intervals of length {length}."
                    )
            
        # Lengths of the sub intervals
        lengths = np.round(np.array([random.gauss(1., sigma)*length for i in range(count)]))
        # Make sure each one is ok
        for i in range(count):
            lengths[i] = max(2, min(self.length, lengths[i]))
        
        # Express the length in percentage of the total length
        pls = [l/self.length for l in lengths]
        
        # Progressive locations
        plocs = np.round(self.length * np.array([i/(count-1)*(1.-pls[i]) for i in range(count)]))
        
        return [TimeInterval(self.t0.frame + plocs[i], self.t0.frame + plocs[i] + lengths[i] - 1) for i in range(count)]
    
        
# -----------------------------------------------------------------------------------------------------------------------------
# Timers can trigger functions
        
class Trigger():
    
    TRIGGERS = []
    
    def __init__(self, objects, f):
        # Make sure the objects parameters is an array of objects
        if not hasattr(objects, '__len__'):
            objects = [objects]
            
        self.objects     = objects
        self.f           = f
        args = inspect.getfullargspec(f).args
        self.need_time   = 'time'   in args
        self.need_index  = 'index'  in args
        self.need_factor = 'factor' in args
        
        Trigger.TRIGGERS.append(self)
        
    def triggered(self, time):
        return False
    
    def get_call_args(self):
        return {}
        
    def call(self, time):
        
        kwargs = self.get_call_args()
        if self.need_time:
            kwargs["time"] = time
            
        for index, obj in enumerate(self.objects):
            if self.need_index:
                kwargs["index"] = index
            self.f(obj, **kwargs)
            
    
    def handle_time(self, time):
        if self.triggered(time):
            self.call(time)
            
    @classmethod
    def time_event(cls, time):
        for trig in cls.TRIGGERS:
            trig.handle_time(time)
    
        
TRIGGERS = ['==', '>', '>=', '<', '<=']

class TimeTrigger(Trigger):
    def __init__(self, objects, f, time, trigger='=='):
        
        if not trigger in TRIGGERS:
            raise WrapException(
                    "Event initialization error: event trigger '{trigger}' is not valid. Must be in {TRIGGERS}"
                    )
            
        super().__init__(objects, f)
        
        self.time    = time
        self.trigger = trigger
        self.need_factor = 'factor' in inspect.getfullargspec(f).args
        
    def triggered(self, time):
        return eval(f"{time.frame} {self.trigger} {self.time.frame}")
    
class TimesTrigger(Trigger):
    def __init__(self, objects, f, times):
        
        super().__init__(objects, f)
        
        if len(objects) != len(times):
            raise WrapException(
                    "timesTrigger initialization error: objects count different from times count",
                    f"Objects count={len(objects)}, Times count= {len(times)}."
                    )
        self.times = times
        
    def handle_time(self, time):

        for index, tm in enumerate(self.times):
            if time.frame == tm.frame:
                kwargs = {}
                if self.need_time:
                    kwargs['time'] = time
                if self.need_index:
                    kwargs['index'] = index
                    
                self.f(self.objects[index], **kwargs)
            
                
class IntervalTrigger(Trigger):
    def __init__(self, objects, f, interval):
        super().__init__(objects, f)
        self.interval = interval
        
    def triggered(self, time):
        return self.interval.contains(time.frame)
    
    def get_call_args(self):
        kwargs = super().get_call_args()
        if self.need_factor:
            kwargs['factor'] = self.interval.progress(time.frame)
        return kwargs

class IntervalsTrigger(Trigger):
    
    def __init__(self, objects, f, intervals):
        super().__init__(objects, f)
        if len(objects) != len(intervals):
            raise WrapException(
                    "IntervalsTrigger initialization error: objects count different from intervals count",
                    f"Objects count={len(objects)}, Intervals count= {len(intervals)}."
                    )
        self.intervals = intervals
        
        
    def handle_time(self, time):

        for index, interval in enumerate(self.intervals):
            if interval.contains(time.frame):
                kwargs = {}
                if self.need_time:
                    kwargs['time'] = time
                if self.need_index:
                    kwargs['index'] = index
                if self.need_factor:
                    kwargs['factor'] = interval.progress(time.frame)
                    
                self.f(self.objects[index], **kwargs)


# ******************************************************************************************************************************************************
# ******************************************************************************************************************************************************
# Kernel handler
# ******************************************************************************************************************************************************
# ******************************************************************************************************************************************************

update_functions = []

def update_clear():
    global update_functions
    update_functions = []

def update_register(f):
    update_functions.append(f)

def kernel_handler(scene):

    time = Time.Current(scene)
    print("frame:", time.frame, "Handle events")
    Trigger.time_event(time)

    # Loops
    for f in update_functions:
        f(time)


# Execute once

def execute():
    kernel_handler(bpy.context.scene)

# ******************************************************************************************************************************************************
# ******************************************************************************************************************************************************
# Registering
# ******************************************************************************************************************************************************
# ******************************************************************************************************************************************************

def menu_func(self, context):
    #self.layout.operator(AddMoebius.bl_idname, icon='MESH_ICOSPHERE')
    pass


def register():
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(kernel_handler)
    pass

    #bpy.utils.register_class(AddMoebius)
    #bpy.types.VIEW3D_MT_mesh_add.append(menu_func)


def unregister():
    bpy.app.handlers.frame_change_pre.remove(kernel_handler)

    #bpy.utils.unregister_class(AddArrow)

    #bpy.types.VIEW3D_MT_mesh_add.remove(menu_func)


if __name__ == "__main__":
    register()
