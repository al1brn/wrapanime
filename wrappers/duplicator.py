import numpy as np

from wrapanime.utils.errors import WrapException

from wrapanime.wrappers import wrappers
from wrapanime.utils import blender as blend

import importlib
importlib.reload(wrappers)
importlib.reload(blend)

from wrapanime.wrappers.wrappers import WObjects

import wrapanime.utils.geometry as geo

# *****************************************************************************************************************************
# *****************************************************************************************************************************
# Gère une collection de copies d'un modèle
# Si le modèle n'existe pas, crée un mesh vide

class Duplicator(WObjects):

    DUPLIS = {}

    def __init__(self, model, linked=False, length=None):
        
        # The model to replicate must exist
        mdl = blend.get_object(model)
        if mdl is None:
            raise WrapException(f"Duplicator ERROR", "The object named '{model}' doesn't exist")
            
        model      = mdl
        model_name = mdl.name
            
        # Let's create the collection to host the duplicates
        
        coll_name  = model_name + "s"
        collection = blend.wrap_collection(coll_name)
        
        # The collection of WObjects is collection.objects
        super().__init__(collection.objects, self)
        
        self.collection    = collection
        
        self.model         = model
        self.model_name    = model_name
        self.base_name     = f"Z_{model_name}"

        #self.dupli_index   = Duplicator.get_dupli_index(self.model_name)
        #self.base_name     = "Z_{}_{}".format(self.model_name, self.dupli_index)

        self.linked        = linked
        
        if length is not None:
            self.set_length(length)
            
    # -----------------------------------------------------------------------------------------------------------------------------
    # Adjust the number of objects in the collection
    
    def set_length(self, length):
        
        count = length-len(self)
        
        if count > 0:
            for i in range(count):
                new_obj = blend.duplicate_object(self.model, self.collection, self.linked)
                if not self.linked:
                    new_obj.animation_data_clear()
                    
        elif count < 0:
            for i in range(-count):
                blend.delete_object(self.collection.objects[-1])
                
    # -----------------------------------------------------------------------------------------------------------------------------
    # The objects are supposed to all have the same parameters
    
    @property
    def rotation_mode(self):
        if len(self) > 0:
            return self[0].rotation_mode
        else:
            return 'XYZ'
        
    @rotation_mode.setter
    def rotation_mode(self, value):
        self.rotation_modes = value
        
    @property
    def euler_order(self):
        if len(self) > 0:
            return self[0].rotation_euler.order
        else:
            return 'XYZ'
        
    @euler_order.setter
    def euler_order(self, value):
        self.rotation_euler.order = value
        
    @property
    def track_axis(self):
        if len(self) > 0:
            return self[0].track_axis
        else:
            return 'POS_Y'
        
    @track_axis.setter
    def track_axis(self, value):
        self.track_axis_s = value
        
    @property
    def up_axis(self):
        if len(self) > 0:
            return self[0].up_axis
        else:
            return 'Z'
        
    @up_axis.setter
    def up_axis(self, value):
        self.up_axis_s = value
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # Orient with a quaternion
    
    def quat_orient(self, quat):
        if self.rotation_mode == 'QUATERNION':
            self.rotation_quaternions = quat
        elif self.rotation_mode == 'AXIS_ANGLE':
            self.rotation_axis_angles = geo.axis_angle(quat, True)
        else:
            self.rotation_eulers = geo.q_to_euler(quat, self.euler_order)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Orient with euler
    
    def euler_orient(self, euler):
        if self.rotation_mode == 'QUATERNION':
            self.rotation_quaternions = geo.e_to_quat(euler, self.euler_order)
        elif self.rotation_mode == 'AXIS_ANGLE':
            self.rotation_axis_angles = geo.axis_angle(geo.e_to_quat(euler, self.euler_order), True)
        else:
            self.rotation_eulers = euler
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # Orient with matrix
    
    def matrix_orient(self, matrix):
        if self.rotation_mode in ['QUATERNION', 'AXIS_ANGLE']:
            self.quat_orient(geo.m_to_quat(matrix))
        else:
            self.euler_orient(geo.m_to_euler(matrix, self.euler_order))
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # Track to a target location
    
    def track_to(self, location):
        locs = np.array(location) - self.locations
        q    = geo.q_tracker(self.track_axis, locs, up=self.up_axis)
        self.quat_orient(q)
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # Orient along a given axis
    
    def orient(self, axis):
        q    = geo.q_tracker(self.track_axis, axis, up=self.up_axis)
        self.quat_orient(q)
        
    
        
