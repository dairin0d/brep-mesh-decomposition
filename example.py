from mesh_decomposition import MeshDecomposer
from step_serialization import export_step

def convert_and_export(path, object_infos, fitting_tolerance=0.01):
    breps = []
    
    decomposer = MeshDecomposer()
    
    for info in object_infos:
        decomposer.set_vertices(info["verts"])
        decomposer.build_topology(info["tris"], info["seams"])
        decomposer.calc_triangle_infos()
        # decomposer.calc_vertex_infos() # currently not used
        decomposer.decompose(fitting_tolerance)
        decomposer.extract_boundaries()
        brep = decomposer.make_brep()
        decomposer.clear()
        
        brep["name"] = info["name"]
        brep["matrix"] = dict(
            origin=info.get("origin", (0, 0, 0)),
            axis_x=info.get("axis_x", (1, 0, 0)),
            axis_y=info.get("axis_y", (0, 1, 0)),
            axis_z=info.get("axis_z", (0, 0, 1)),
            scale=info.get("scale", 1.0),
        )
        breps.append(brep)
    
    export_step(path, breps)

def make_test_object(name, origin, height, segments, seams=False):
    import math
    verts = [(0.0, 0.0, height)]
    for i in range(segments+1):
        angle = (math.pi/2) * (i / (segments))
        verts.append((math.sin(angle), math.cos(angle), 0.0))
    seams = ([(0, i+1) for i in range(1, segments)] if seams else [])
    tris = [(0, i+1, i+2) for i in range(segments)]
    return dict(name=name, origin=origin, verts=verts, seams=seams, tris=tris)

object_infos = [
    make_test_object("plane", (0.0, 0.0, 0.0), 0.0, 2),
    make_test_object("seams", (2.0, 0.0, 0.0), 0.0, 2, True),
    make_test_object("cone", (-2.0, 0.0, 0.0), 1.0, 32),
]

convert_and_export("./example.step", object_infos, fitting_tolerance=0.01)
