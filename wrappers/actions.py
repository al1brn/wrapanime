#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 09:42:35 2020

@author: alain
"""

import numpy as np

import bpy

import wrapanime.utils.blender as blender
from wrapanime.utils.errors import WrapException
import wrapanime.wrappers.root as root

import importlib
importlib.reload(root)
importlib.reload(blender)

class WCurve():
    def __init__(self, bcurve):
        self.bcurve  = bcurve
        length = len(bcurve)
        
        # Read the control points
        
        self.bpoints = np.empty(length*6, np.float).reshape(length, 6)
        vx = np.empty(length*2, np.float)
        
        bcurve.keyframe_points.foreach('co', vx)
        self.bpoints[2:4] = vx.reshape(length, 2)
        bcurve.keyframe_points.foreach('handle_left', vx)
        self.bpoints[0:2] = vx.reshape(length, 2)
        bcurve.keyframe_points.foreach('handle_right', vx)
        self.bpoints[4:6] = vx.reshape(length, 2)
        
        
        
        
        






class WAction():
    
    @property
    def curves():
        curves = []
        for action in bpy.data.actions:
            if action.name.startswith(self.prefix):
                actions.append(action)
                
        return actions
        
        








class WActions():
    prefix = "WA "
    """Manages the actions in Blender"""
    def __init__(self):
        pass
    
    @property
    def actions(self):
        actions = []
        for action in bpy.data.actions:
            if action.name.startswith(self.prefix):
                actions.append(action)
                
        return actions
    
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, index):
        if type(index) is int:
            return self.actions[index]
        elif type(index) is str:
            return bpy.data.actions.get(index)
        else:
            return bpy.data.actions.get(index.name)
    
    def new(self, name):
        named = []
        base = self.prefix + name
        for action in bpy.data.actions:
            if action.name.startswith(self.prefix + name):
                named.append(action.name)
                
        new_name = blender.get_free_name(base, named)
        action = bpy.data.actions.new(name=new_name)
        action.use_fake_user = True
        return action
    
    def delete(self, index):
        action = self[index]
        if action is not None:
            action.use_fake_user = False
            
        
    
    
                



# =============================================================================================================================
# Keyframe curves
        
class WKeyFrames(wgen.WKeyFrames):
    
    def set_length(self, length):
        
        # Remove the ones which are two numerous
        exceed = length-len(self)
        for i in range(exceed):
            self.obj.keyframes.remove(self.obj[-1], fast=True)
            
        # Complete if not enough
        need = length - len(self)
        if need > 0:
            self.obj.add(need)
    
    def set_bezier_func(self, bezier):
        self.set_length(len(bezier))
        self.bpoints2 = bezier.points
    
    
class KFAnimation():
    def __init__(self, obj):
        self.obj = blender.get_object(obj)
        
    @property
    def is_animated(self):
        return self.obj.animation_data is not None
        
    @property
    def animation(self):
        animation = self.obj.animation_data
        if animation is None:
            return self.obj.animation_data_create()
        else:
            return animation
        
    @property
    def action(self):
        action = self.animation.action
        if action is None:
            self.animation.action = bpy.data.actions.new(name="WA action")
            return self.animation.action
        
        return action
    
    @property
    def fcurves(self):
        return self.action.fcurves
    
    @classmethod
    def fcurve_of(cls, fcurve, name, index):
        if index < 0:
            return fcurve.data_path == name
        else:
            return (fcurve.data_path == name) and (fcurve.array_index == index)
    
    def fcurve(self, name, index=0):
        if self.fcurves is None:
            return None
        
        for curve in self.fcurves:
            if self.fcurve_of(curve, name, index):
                return curve
                
        return None
    
    def new(self, name, index=0):
        curve = self.fcurve(name, index)
        if curve is None:
            curve = self.fcurves.new(data_path=name, index=index)
        return curve
    
    # ===== Manage the keyframes and curves
    
    def keyframes(self, name, index=0, create=True):
        
        if create:
            curve = self.new(name, index)
        else:
            curve = self.fcurve(self, name, index)
        
        if curve is None:
            return None
        
        return WKeyFrames(curve.keyframe_points, wowner=None)
            

