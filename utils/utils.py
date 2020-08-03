# *****************************************************************************************************************************
# *****************************************************************************************************************************
# Utilitaires
# Crée le 12 novembre 2019
#
# Regroupe les petits utilitaires pratiques

import bpy
from math import (cos, sin, tan, atan2, pi, degrees, radians, sqrt)
from mathutils import (Vector, Matrix)

# *****************************************************************************************************************************
# *****************************************************************************************************************************
# Utilitaires d'affichage

# -----------------------------------------------------------------------------------------------------------------------------
# Message

def dump_object(obj, blanks):
    if obj is not None:
        if hasattr(obj, "dump"):
            obj.dump(blanks=blanks)
        else:
            print(obj)

def verbose(message, obj=None, blanks=0):
    print((" "*blanks)+message)
    print(obj, blanks)

def warning(title, message=None, obj=None, blanks=0):
    print((" "*blanks)+"WARNING:", title)
    if message is not None:
        print((" "*blanks)+message)
    dump_object(obj, blanks)

def error(title, message=None, obj=None):
    print()
    print("E"*100)
    print("ERROR:", title)
    if message is not None:
        print(message)
    dump_object(obj)
    print("E"*100)
    print()
    
    raise NameError("Fatal error, see above")
    
# -----------------------------------------------------------------------------------------------------------------------------
# Convertit un vecteur en chaîne de caractères

def vstr(V, title=None):
    sep = "["
    if title is not None:
        sep = title + ": ["
    s = ""
    for v in V:
        s += "%s%+6.2f" % (sep, v)
        sep = " "
    return s + "]"

# -----------------------------------------------------------------------------------------------------------------------------
# Convertit une matrice en chaîne de caractères

def mstr(M, title=None):
    if title is not None:
        s = "%s\n" % title
    else:
        s = "\n"
        
    for V in M:
        s += "    %s\n" % vstr(V)

    return s


# *****************************************************************************************************************************
# *****************************************************************************************************************************
# Boucle circulaire

def next_i(i, n):
    return 0 if i == n-1 else i+1

# *****************************************************************************************************************************
# *****************************************************************************************************************************
# Utilitaires de collections Blender

# -----------------------------------------------------------------------------------------------------------------------------
# Récupère la liste des collections d'un objet

def get_object_collections(object):
    colls = []
    for coll in bpy.data.collections:
        if object.name in coll.objects:
            colls.append(coll)
            
    return colls

# -----------------------------------------------------------------------------------------------------------------------------
# Récupère une collection par son nom

def get_collection(name):
    if name is None:
        return None
    
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    return None

# -----------------------------------------------------------------------------------------------------------------------------
# Crée une collection

def create_collection(name, parent_name=None):
    
    new_coll = bpy.data.collections.new(name)
    
    # Rattachement au parent
    
    parent = get_collection(parent_name)
    if parent is not None:
        parent.children.link(new_coll)
    
    return new_coll

# -----------------------------------------------------------------------------------------------------------------------------
# Link un objet nouvellement crée

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

# *****************************************************************************************************************************
# *****************************************************************************************************************************
# Géométrie

# -----------------------------------------------------------------------------------------------------------------------------
# Coordonnées sphériques

def to_spherical(V):
    
    Vxy = Vector((V[0], V[1]))
    
    theta = atan2(Vxy.y, Vxy.x)
    phi   = atan2(V[2], Vxy.length)
    
    return (Vector(V).length, theta, phi)

# -----------------------------------------------------------------------------------------------------------------------------
# Oriente un vecteur selon une direction donnée

def point_to(obj, V=(0., 0., 1.), adjustLength=False):
    if type(obj) is type("str"):
        obj = getVector(obj)
    S = Vecteur.to_spherical(V)
    obj.rotation_mode = 'XYZ'
    obj.rotation_euler = (0., pi/2-S[2], S[1])
    if adjustLength:
        obj.data.shape_keys.key_blocks["Longueur"].value = 0.1*S[0]
        
# -----------------------------------------------------------------------------------------------------------------------------
# Conversion d'un vecteur d'espace en vecteur 3D

def to_3Dvector(V):
    if len(V) == 1:
        return Vector((V[0], 0., 0.))
    elif len(V) == 2:
        return Vector((V[0], V[1], 0.))
    elif len(V) == 3:
        return Vector(V)
    else:
        M = bpy.context.scene.fourD_settings.M_proj()
        return M*Vector(V)





