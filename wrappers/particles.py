#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 07:55:13 2020

@author: alain
"""

import numpy as np

import bpy

from wrapanime.utils.errors import WrapException

from wrapanime.utils import blender as blender
from wrapanime.utils import geometry as geo
from wrapanime.wrappers import generated_wrappers as wgen
from wrapanime.wrappers import wrappers as wrappers

import importlib
importlib.reload(blender)
importlib.reload(geo)
importlib.reload(wgen)
importlib.reload(wrappers)


class Particles(wgen.WParticles):
    
    def __init__(self, obj_name):
        
        wo = wrappers.wrap(obj_name, 'CUBE', blender.wrap_collection("Particles"))
        
        #obj = blender.getcreate_object(obj_name, create='CUBE', collection=blender.wrap_collection("Particles"))
        #wo = wrappers.wrap(obj)
        parts = blender.getcreate_particles(wo.object)
        
        super().__init__(wo, parts.name)
        
    @property
    def particle_system(self):
        return self.top_object.particle_systems[self.psys_name]        
        
    @property
    def settings(self):
        return self.top_object.particle_systems[self.psys_name].settings      
        
    @classmethod
    def Duplicator(cls, model_name, length=1000):
        
        model = blender.get_object(model_name)
        if model is None:
            raise WrapException(f"Particles.Duplicator error: the object {model_name} doesn't exist")
            
        emitter_name  = "W Parts " + model.name
        parts = cls(emitter_name)
        
        settings = parts.settings
        settings.count = length
        
        settings.physics_type  = 'NEWTON'
        settings.particle_size = 1.
        
        settings.render_type = 'OBJECT'
        settings.instance_object = model
        #settings.show_unborn = True
        #settings.use_dead = True
        
        return parts
        
        
        
        
        
        
        
    
        
    

