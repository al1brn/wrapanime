import bpy

# ******************************************************************************************************************************************************
# ******************************************************************************************************************************************************
# Collections
# ******************************************************************************************************************************************************
# ******************************************************************************************************************************************************

def find_collection(context, item):
    collections = item.users_collection
    if len(collections) > 0:
        return collections[0]
    return context.scene.collection

def make_collection(collection_name, parent_collection):
    if collection_name in bpy.data.collections: # Does the collection already exist?
        return bpy.data.collections[collection_name]
    else:
        new_collection = bpy.data.collections.new(collection_name)
        parent_collection.children.link(new_collection) # Add the new collection under a parent
        return new_collection

def get_collection(coll):
    if type(coll) is str:
        return bpy.data.collections.get(coll)
    else:
        return coll

def get_object_collections(object):
    colls = []
    for coll in bpy.data.collections:
        if object.name in coll.objects:
            colls.append(coll)

    return colls

def create_collection(name, parent_name=None):

    new_coll = bpy.data.collections.get(name)

    if new_coll is None:

        new_coll = bpy.data.collections.new(name)

        parent = get_collection(parent_name)
        if parent is None:
            parent = bpy.context.scene.collection

        parent.children.link(new_coll)

    return new_coll

def link_object(obj, collection=None):

    colls = get_object_collections(obj)
    for coll in colls:
        coll.objects.unlink(obj)

    if collection is None:
        bpy.context.collection.objects.link(obj)
    else:
        collection.objects.link(obj)
    return obj

# *****************************************************************************************************************************
# *****************************************************************************************************************************
# Utilitaires de gestion des objets

# -----------------------------------------------------------------------------------------------------------------------------
# Duplication d'un objet et de sa hiérarchie

def duplicate_object(obj, collection=None, link=False):

    dupl = obj.copy()

    # Duplicate data
    if obj.data is not None:
        if link:
            dupl.data = obj.data
        else:
            dupl.data = obj.data.copy()

    # ----- Collection to place the duplicate into
    if collection is None:
        colls = get_object_collections(obj)
        for coll in colls:
            coll.objects.link(dupl)
    else:
        collection.objects.link(dupl)

    # ----- Children copy
    for child in obj.children:
        duplicate_object(child, collection=collection, link=link).parent = dupl

    # ----- Done !

    return dupl

# -----------------------------------------------------------------------------------------------------------------------------
# Récupère un objet par son nom, le crée s'il n'existe pas
# ATTENTION: Le mesh créé est vide

def create_object(name, type='MESH', parent=None, collection=None):

    bpy.ops.object.add(type=type)
    obj             = bpy.context.active_object
    obj.name        = name
    obj.parent      = parent
    obj.location    = bpy.context.scene.cursor.location
    link_object(obj, collection)

    return obj

# -----------------------------------------------------------------------------------------------------------------------------
# Supprime un objet et ses enfants

def delete_object(obj, children=True):

    def add_to_coll(o, coll):
        for child in o.children:
            add_to_coll(child, coll)
        coll.append(o)

    coll = []
    if children:
        add_to_coll(obj, coll)
    else:
        coll = [obj]

    for o in coll:
        bpy.data.objects.remove(o)


# -----------------------------------------------------------------------------------------------------------------------------
# Lisse l'object

def smooth_object(obj):

    mesh = Mesh(obj)
    for f in mesh.bm.faces:
        f.smooth = True
    mesh.done()

    return obj

# -----------------------------------------------------------------------------------------------------------------------------
# Cache l'objet

def hide_object(obj, value, frame=None, also_viewport=True):
    if also_viewport:
        print("DEBUG:", obj, value)
        obj.hide_viewport = value
    obj.hide_render = value
    if frame is not None:
        if also_viewport:
            obj.keyframe_insert(data_path="hide_viewport", frame=frame)
        obj.keyframe_insert(data_path="hide_render", frame=frame)

# -----------------------------------------------------------------------------------------------------------------------------
# Assigne une texture

def set_material(obj, material_name):

    # Get material
    mat = bpy.data.materials.get(material_name)
    if mat is None:
        return
        # mat = bpy.data.materials.new(name="Material")

    # Assign it to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    return mat
