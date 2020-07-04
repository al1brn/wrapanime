import bpy

from wrapanime.wrappers.wrappers import WObject, ArrayOfWObject
from wrapanime.utils.errors import WrapException
import wrapanime.utils.blender as blend

# *****************************************************************************************************************************
# *****************************************************************************************************************************
# Gère une collection de copies d'un modèle
# Si le modèle n'existe pas, crée un mesh vide

class Duplicator(ArrayOfWObject):

    DUPLIS = {}

    def __init__(self, model, linked=False):
        super().__init__(None)

        if type(model) is str:
            model_name = model
            model = bpy.data.objects.get(model_name)

        if model is None:
            raise WrapException("Duplicator ERROR", "The object named '{}' doesn't exist".format(model_name))

        self.model         = model
        self.model_name    = model.name

        self.dupli_index   = Duplicator.get_dupli_index(self.model_name)
        self.base_name     = "Z_{}_{}".format(self.model_name, self.dupli_index)

        self.linked        = linked
        self.collection    = None

    @classmethod
    def get_dupli_index(cls, model_name):
        count = Duplicator.DUPLIS.get(model_name)
        if count is None:
            count = -1
        count +=1
        Duplicator.DUPLIS[model_name] = count
        return count

    # Get the collection containing the created objects
    def get_collection(self):

        if self.collection is None:

            cname = self.model_name + "s"
            self.collection = blend.create_collection(cname)

        return self.collection

        # ==== OLD
        try:
            coll = bpy.data.collections[cname]
        except:
            coll = bpy.data.collections.new(cname)
            bpy.context.scene.collection.children.link(coll)

        return coll

    # Name of a duplicate objects
    def duplicate_name(self, index):
        return "%s.%05i" % (self.base_name, index)

    @classmethod
    def index_from_name(cls, name):
        return int(name.split('.')[-1])

    # Get all the objects matching the collection name
    def matching_objects(self):
        objects = []
        base = self.base_name
        for o in bpy.data.objects:
            name = o.name.split('.')
            if name[0] == base:
                objects.append(o)
        return objects

    # Set the Length
    def set_length(self, length):

        # Set the array at the right size
        super().set_length(length)

        # List of the existing objects
        objects = self.matching_objects()
        existing = len(objects)

        # Create supplementary objects if necessary
        if existing < length:
            objects.extend(self.create_duplicate(length-existing, objects))

        # Into to self array
        for i in range(length):
            self.array[i] = self.wrap(objects[i])

        # Delete exceeding objects if any
        if existing > length:
            for i in range(length, len(objects)):
                blend.delete_object(objects[i])

    # Renum the objects
    def clean_names(self, matching=None):

        if matching is None:
            matching = self.matching_objects()

        key = "WA_RENAME"
        for o in self.objects:
            o.name = key
        for i in range(len(matching)):
            matching[i].name = self.duplicate_name(i)

    # Create duplicates
    def create_duplicate(self, count=1, matching=None):
        objects = []

        if count < 1:
            return objects

        if matching is None:
            matching = self.matching_objects()

        # Max index
        index = -1
        for obj in matching:
            index = max(index, self.index_from_name(obj.name))

        # Clean if too many holes
        if index > 10*(len(self) + count):
            self.clean_names(matching)
            index = len(matching) - 1

        # Creation loop
        for i in range(count):
            index += 1
            name = self.duplicate_name(index)

            # Create a copy
            new_obj = blend.duplicate_object(self.model, collection=self.get_collection(), link=self.linked)
            new_obj.name = name
            if not self.linked:
                new_obj.animation_data_clear()
            objects.append(new_obj)

        return objects
