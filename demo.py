import wrapanime as wrap

# In blender create 3 objects:
# An icosphere named icosphere
# A cylinder named "Tube model"
# An empty named "Empty"


# Creation

ico = wrap.WMeshObject("Icosphere")

cylinders = wrap.Duplicator("Tube model", linked=true)
cylinders.set_length(len(ico.faces))

tracker = wobject("Empty")

# Functions loop

def wobj_action(wobj, index, time, location, direction):
    wobj.location = location
    wobj.along(direction)
    wobj.scale = wrap.to_vector(wrap.distance(wobj, tracker))

# Time Animation

wrap.register_foreach(cylinders, wobj_action, locations=ico.t_faces, directions=ico.t_faces_normals)

wrap.frame_animation()
