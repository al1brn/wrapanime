from wrapanime.utils.errors import WrapException

from wrapanime.wrappers import wrappers
from wrapanime.utils import blender as blend

import importlib
importlib.reload(wrappers)
importlib.reload(blend)

from wrapanime.wrappers.wrappers import WObjects

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
        collection = blend.create_collection(coll_name)
        
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

            

