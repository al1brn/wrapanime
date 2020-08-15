import bpy

from wrapanime.utils.errors import WrapException

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

def get_collection(coll, create=True):
    if type(coll) is str:
        c = bpy.data.collections.get(coll)
        if (c is None) and create:
            return create_collection(coll)
        return c
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
        get_collection(collection).objects.link(obj)
        
    return obj

def wrap_collection(name=None):
    if name is None:
        return get_collection("WrapAnime", create=True)
    
    cname = name if name[:2] == "W " else "W " + name
    
    return create_collection(cname, parent_name="WrapAnime")

def hidden_collection():
    return wrap_collection("W Hidden")

# *****************************************************************************************************************************
# *****************************************************************************************************************************
# Shape keys
    
def sk_name(name, step=None):
    return name if step is None else f"{name} {step:03d}"

def get_sk(obj, name, step=None, create=True):

    name = sk_name(name, step)
    
    if obj.data.shape_keys is None:
        if create:
            obj.shape_key_add(name=name)
            obj.data.shape_keys.use_relative = False
        else:
            return None
    
    # Does the shapekey exists?
    
    res = obj.data.shape_keys.key_blocks.get(name)
    
    # No !
    
    if (res is None) and create:
        
        eval_time = obj.data.shape_keys.eval_time 
        
        if step is not None:
            # Ensure the value is correct
            obj.data.shape_keys.eval_time = step*10
        
        res = obj.shape_key_add(name=name)
        
        # Less impact as possible :-)
        obj.data.shape_keys.eval_time = eval_time
        
    return res

def sk_exists(obj, name, step):
    return get_sk(obj, name, step, create=False) is not None

def set_on_sk(obj, name, step=None):
    
    if not sk_exists(obj, name, step):
        raise WrapException(f"The shape key '{sk_name(name, step)}' doesn't exist in object '{obj.name}'!")

    obj.data.shape_keys.eval_time = get_sk(obj, name, step).frame
    return obj.data.shape_keys.eval_time

def delete_sk(obj, name=None, step=None):
    
    if obj.data.shape_keys is None:
        return
    
    if name is None:
        obj.shape_key_clear()
    else:
        key = get_sk(obj, name, step)
        if key is not None:
            obj.shape_key_remove(key)


# *****************************************************************************************************************************
# *****************************************************************************************************************************
# Names management
            
def get_name_number(name):
    p = name.find(".") < 0
    if p:
        return None
    return int(name[len(name) - p + 1:])

def set_name_number(name, number):
    return f"{name}.{number:03d}"
    
            
def get_free_name(name, currents):
    
    if currents is None or len(currents) == 0:
        return name
    
    scur = currents.copy()
    scur.sort()
    if scur[0] != name:
        return name
    
    scur = scur[1:]
    
    for i in range(len(scur)):
        if get_name_number(scur[i]) > i:
            return set_name_number(name, i)
        
    return set_name_number(name, len(scur))
    

# *****************************************************************************************************************************
# *****************************************************************************************************************************
# Update the view port after objects transformations

def update_viewport():
    try:
        bpy.ops.object.editmode_toggle()
        bpy.ops.object.editmode_toggle()
    except:
        pass


# *****************************************************************************************************************************
# *****************************************************************************************************************************
# Utilitaires de gestion des objets
    
# -----------------------------------------------------------------------------------------------------------------------------
# Create an object

def create_object(name, what='CUBE', parent=None, collection=None, **kwargs):
    
    if what in ['MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'VOLUME', 'ARMATURE', 'LATTICE',
                'EMPTY', 'GPENCIL', 'CAMERA', 'LIGHT', 'SPEAKER', 'LIGHT_PROBE']:
        
        bpy.ops.object.add(type=what, **kwargs)
    
    elif what == 'CIRCLE':
        bpy.ops.mesh.primitive_circle_add(**kwargs)
    elif what == 'CONE':
        bpy.ops.mesh.primitive_cone_add(**kwargs)
    elif what == 'CUBE':
        bpy.ops.mesh.primitive_cube_add(**kwargs)
    elif what == 'GIZMO_CUBE':
        bpy.ops.mesh.primitive_cube_add_gizmo(**kwargs)
    elif what == 'CYLINDER':
        bpy.ops.mesh.primitive_cylinder_add(**kwargs)
    elif what == 'GRID':
        bpy.ops.mesh.primitive_grid_add(**kwargs)
    elif what in ['ICOSPHERE', 'ICO_SPHERE']:
        bpy.ops.mesh.primitive_ico_sphere_add(**kwargs)
    elif what == 'MONKEY':
        bpy.ops.mesh.primitive_monkey_add(**kwargs)
    elif what == 'PLANE':
        bpy.ops.mesh.primitive_plane_add(**kwargs)
    elif what == 'TORUS':
        bpy.ops.mesh.primitive_torus_add(**kwargs)
    elif what in ['UVSPHERE', 'UV_SPHERE']:
        bpy.ops.mesh.primitive_uv_sphere_add(**kwargs)
        
        
    elif what in ['BEZIERCIRCLE', 'BEZIER_CIRCLE']:
        bpy.ops.curve.primitive_bezier_circle_add(**kwargs)
    elif what in ['BEZIERCURVE', 'BEZIER_CURVE']:
        bpy.ops.curve.primitive_bezier_curve_add(**kwargs)
    elif what in ['NURBSCIRCLE', 'NURBS_CIRCLE']:
        bpy.ops.curve.primitive_nurbs_circle_add(**kwargs)
    elif what in ['NURBSCURVE', 'NURBS_CURVE']:
        bpy.ops.curve.primitive_nurbs_curve_add(**kwargs)
    elif what in ['NURBSPATH', 'NURBS_PATH']:
        bpy.ops.curve.primitive_nurbs_path_add(**kwargs)
        
    else:
        raise WrapException(f"Invalid object creation name: '{what}' is not valid")
        
    obj             = bpy.context.active_object
    obj.name        = name
    obj.parent      = parent
    obj.location    = bpy.context.scene.cursor.location
    
    if collection is not None:
        bpy.ops.collection.objects_remove_all()
        get_collection(collection).objects.link(obj)    

    return obj

# -----------------------------------------------------------------------------------------------------------------------------
# Get an object by name of object itself
# The object can also be a WObject
            
def get_object(obj_or_name, mandatory=True, otype=None):
    
    if type(obj_or_name) is str:
        obj = bpy.data.objects.get(obj_or_name)
    elif hasattr(obj_or_name, 'name'):
        obj = bpy.data.objects.get(obj_or_name.name)
    else:
        obj = obj_or_name
        
    if (obj is None) and mandatory:
        raise WrapException(f"Object '{obj_or_name}' doesn't exist")
        
    if hasattr(obj, "obj"):
        obj = obj.obj
        
    if (obj is not None) and (otype is not None):
        if obj.type != otype:
            raise WrapException(
                    f"Blender object type error: '{otype}' expected",
                    f"The type of the Blender object '{obj.name}' is '{obj.type}."
                    )
    return obj

# -----------------------------------------------------------------------------------------------------------------------------
# Get an object and create it if it doesn't exist
# if create is None -> no creation
# For creation, the create argument must contain a valid object creation name
    
def getcreate_object(obj_or_name, create=None, collection=None, **kwargs):
    
    obj = get_object(obj_or_name, mandatory = create is None)
    if obj is not None:
        return obj
    
    return create_object(obj_or_name, what=create, parent=None, collection=collection, **kwargs)


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

    mesh = obj.data
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
# Key frames mamangement
        
# ----------------------------------------------------------------------------------------------------
# Key frame animation
    
def is_animated(obj):
    return obj.animation_data is not None
    
def animation_data(obj, create=True):
    animation = obj.animation_data
    if create and (animation is None):
        return obj.animation_data_create()
    else:
        return animation
    
def animation_action(obj, create=True):
    animation = animation_data(obj, create)
    if animation is None:
        return None
    
    action = animation.action
    if create and (action is None):
        animation.action = bpy.data.actions.new(name="WA action")
    
    return animation.action

def get_fcurves(obj):
    return animation_action(obj, True).fcurves

def is_fcurve_of(fcurve, name, index=0):
    if fcurve.data_path == name:
        if fcurve.array_index < 0:
            return True
        return fcurve.array_index == index
    
    return False

def delete_fcurve(obj, fcurve):
    fcurves = get_fcurves(obj)
    try:
        fcurves.remove(fcurve)
    except:
        pass

def name_to_path_index(obj, name):
    """Transform a user name in blender (data_path, array_index) couple.
    
    Parameters
    ----------
    obj: Blender datablock
        A blender datablock
        
    name: str
        An extended data_path string.
        - 'y' and 'location.y' are interpreted as ('location', 1)
        - 'data.attr' is interpretated as ('attr', 0) for data property 
        
    Returns
    -------
    triplet (object, data_path, array_index)
    """
    
    indices = ['x', 'y', 'z', 'w']
    
    dic = {
        'x' : ('location',       0), 'y' : ('location',       1), 'z' : ('location',       2), 
        'sx': ('scale',          0), 'sy': ('scale',          1), 'sz': ('scale',          2), 
        'rx': ('rotation_euler', 0), 'ry': ('rotation_euler', 1), 'rz': ('rotation_euler', 2), 
        }
    
    index = 0
    parts = name.split('.')
    
    try:
        n, index = dic[parts[-1]]
        parts[-1] = n
        if len(parts) > 1:
            if parts[-1] == parts[-2]:
                parts = parts[:-1]
    except:
        pass
        
    ob = obj
        
    for i in range(len(parts)-1):
        try:
            ob = getattr(ob, parts[i])
        except:
            raise WrapException(
                f"Incorrect animation path: '{name}' is not valid for {obj}"
                )
            
    return ob, parts[-1], index

# ----------------------------------------------------------------------------------------------------
# Access to an animation curve
    
def get_fcurve(obj, name, index=None):
    
    if index is None:
        obj, name, index = name_to_path_index(obj, name)
    
    fcurves = get_fcurves(obj)
    if fcurves is None:
        return None
    
    for curve in fcurves:
        if is_fcurve_of(curve, name, index):
            return curve
            
    return None

# ----------------------------------------------------------------------------------------------------
# Get a keyframe at a fiven frame
    
def get_keyframe(obj, name, frame, index=None):
    
    if index is None:
        obj, name, index = name_to_path_index(obj, name)
        
    curve = get_fcurve(obj, name, index)
    for kf in curve.keyframe_points:
        if kf.co[0]==frame:
            return kf
        
    return None

# ----------------------------------------------------------------------------------------------------
# Create and animation curve

def new_curve(obj, name, index=None):
    
    if index is None:
        obj, name, index = name_to_path_index(obj, name)
    
    curve = get_fcurve(obj, name, index)

    if curve is None:
        fcurves = get_fcurves(obj)
        curve = fcurves.new(data_path=name, index=index)

    return curve

# ----------------------------------------------------------------------------------------------------
# Set an existing fcurve
    
def set_fcurve(obj, name, fcurve, index=None):
    
    kfp = fcurve.keyframe_points
    
    if len(kfp) == 0:
        return

    if index is None:
        obj, name, index = name_to_path_index(obj, name)
        
    fc = new_curve(obj, name, index)
    n = len(fc.keyframe_points)
    for i in range(n):
        fc.keyframe_points.remove(fc.keyframe_points[0], fast=True)
        
    fc.extrapolation = fcurve.extrapolation
    fc.keyframe_points.add(len(kfp))
    for kfs, kft in zip(kfp, fc.keyframe_points):
        kft.co            = kfs.co.copy()
        kft.interpolation = kfs.interpolation
        kft.amplitude     = kfs.amplitude
        kft.back          = kfs.back
        kft.easing        = kfs.easing
        kft.handle_left   = kfs.handle_left
        kft.handle_right  = kfs.handle_right
        kft.period        = kfs.period

# ----------------------------------------------------------------------------------------------------
# Delete keyframes

def kf_delete(obj, name, frame0=None, frame1=None, index=None):
    
    if index is None:
        obj, name, index = name_to_path_index(obj, name)
    
    curve = get_fcurve(obj, name, index)
    if curve is None:
        return
    
    kfs = []
    for kf in curve.keyframe_points:
        ok = True
        if frame0 is not None:
            ok = kf.co[0] >= frame0
        if frame1 is not None:
            if kf.co[0] > frame1:
                ok = False
        if ok:
            kfs.append(kf)
    
    for kf in kfs:
        try:
            curve.keyframe_points.remove(kf)
        except:
            pass
        
# ----------------------------------------------------------------------------------------------------
# Insert a key frame
        
def kf_insert(obj, name, frame, value, index=None):
    
    if index is None:
        obj, name, index = name_to_path_index(obj, name)
    
    curr = getattr(obj, name)
    try:
        v = curr.copy()
        v[index] = value
    except:
        v = value
    setattr(obj, name, v)
    obj.keyframe_insert(name, index=index, frame=frame)
    setattr(obj, name, curr)

# ----------------------------------------------------------------------------------------------------
# Animation on an interval
    
def kf_interval(obj, name, frame0, frame1, value0, value1, interpolation='LINEAR', index=None):
    
    if index is None:
        obj, name, index = name_to_path_index(obj, name)

    kf_delete(obj, name, frame0, frame1, index)
    
    curr = getattr(obj, name)
    
    try:
        v = curr.copy()
        v[index] = value0
    except:
        v = value0
    
    setattr(obj, name, v)
    obj.keyframe_insert(name, index=index, frame=frame0)
    
    
    try:
        v[index] = value1
    except:
        v = value1
    
    setattr(obj, name, v)
    obj.keyframe_insert(name, index=index, frame=frame1)
    
    setattr(obj, name, curr)
    
    kf = get_keyframe(obj, name, frame0, index)
    kf.interpolation = interpolation
    
    
# *****************************************************************************************************************************
# *****************************************************************************************************************************
# Particles
    
def getcreate_particles(obj):
    
    if len(obj.particle_systems) == 0:
        
        obj.modifiers.new("WParticles", type='PARTICLE_SYSTEM')
        part = obj.particle_systems[0]
    
        settings = part.settings
        settings.emit_from = 'VERT'
        settings.physics_type = 'NO'
        settings.particle_size = 0.1
        
        #settings.render_type = 'OBJECT'
        #settings.dupli_object = bpy.data.objects['Cube']
        #settings.show_unborn = True
        #settings.use_dead = True    
    
    return obj.particle_systems[0]

    
    

        

    

