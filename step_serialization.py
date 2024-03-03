# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 dairin0d https://github.com/dairin0d

import sys
import math
import datetime

import numpy as np

nonzero = math.ulp(0) or sys.float_info.min
epsilon = sys.float_info.epsilon

norm = np.linalg.norm

def discretize_bezier(result, bezier, tolerance, extra_condition=None):
    if len(result) == 0:
        result.append(bezier[0])
    
    queue = [bezier]
    
    while queue:
        p0, p1, p2, p3 = queue.pop()
        
        if extra_condition and extra_condition(p0, p1, p2, p3):
            result.append(p3)
            continue
        
        lenA = norm(p3 - p0)
        lenB = norm(p1 - p0) + norm(p2 - p1) + norm(p3 - p2)
        if (lenB - lenA) <= tolerance:
            result.append(p3)
            continue
        
        q0 = (p0 + p1) * 0.5
        q1 = (p1 + p2) * 0.5
        q2 = (p2 + p3) * 0.5
        r0 = (q0 + q1) * 0.5
        r1 = (q1 + q2) * 0.5
        s0 = (r0 + r1) * 0.5
        
        queue.append((p0, q0, r0, s0))
        queue.append((s0, r1, q2, p3))

# Winding number test for a point in a polygon
# Adapted from http://geomalgorithms.com/a03-_inclusion.html
def winding_number(point, polygon):
    wn = 0
    
    # Points may be 3D
    px = point[0]
    py = point[1]
    
    for index in range(len(polygon)):
        vertex0 = polygon[index-1]
        vertex1 = polygon[index]
        
        v0x = vertex0[0]
        v0y = vertex0[1]
        v1x = vertex1[0]
        v1y = vertex1[1]
        
        if v0y <= py:
            if v1y > py:
                if (v1x - v0x) * (py - v0y) - (px - v0x) * (v1y - v0y) > 0:
                    wn += 1
        else:
            if v1y <= py:
                if (v1x - v0x) * (py - v0y) - (px - v0x) * (v1y - v0y) < 0:
                    wn -= 1
    
    return wn

def contains_cone_apex(f, brep_points, brep_edges, origin, matrix_inv):
    def extra_condition(p0, p1, p2, p3):
        xmin = min(p0[0], p1[0], p2[0], p3[0])
        xmax = max(p0[0], p1[0], p2[0], p3[0])
        ymin = min(p0[1], p1[1], p2[1], p3[1])
        ymax = max(p0[1], p1[1], p2[1], p3[1])
        return (xmin <= 0) and (xmax >= 0) and (ymin <= 0) and (ymax >= 0)
    
    tolerance = 1e-5
    
    center = (0, 0)
    total_winding_number = 0
    
    for bound in f.bounds:
        polyline = []
        for ei, same_orientation in bound:
            control_point_ids = (brep_edges[ei][-1] if same_orientation else reversed(brep_edges[ei][-1]))
            control_points = [matrix_inv @ (brep_points[p_i] - origin) for p_i in control_point_ids]
            for p_i in range(3, len(control_points), 3):
                bezier = control_points[p_i-3:p_i+1]
                discretize_bezier(polyline, bezier, tolerance, extra_condition)
        
        total_winding_number += winding_number(center, polyline)
    
    return total_winding_number < 0

class Entry:
    def __init__(self, container, name, data=None):
        container.entries.append(self)
        self.container = container
        self.id = "#" + str(len(container.entries))
        self.name = name
        self.data = data
    
    @classmethod
    def _format_float(cls, value):
        value_str = str(value)
        if "." in value_str: return value_str.upper()
        return f"{value:.1E}"
    
    @classmethod
    def _serialize(cls, value, container):
        if isinstance(value, str): return "'" + value.replace("'", "''") + "'"
        if isinstance(value, bytes): return value.decode()
        if isinstance(value, bool): return (".T." if value else ".F.")
        if isinstance(value, int): return str(value)
        if isinstance(value, float): return cls._format_float(value)
        if isinstance(value, Entry):
            return (value.id if value.container == container else value._to_str(container))
        if isinstance(value, (list, tuple)):
            return "(" + ",".join(cls._serialize(item, container) for item in value) + ")"
        raise Exception(f"Unsupported value type {type(value)}")
    
    def _to_str(self, container):
        content = [self._serialize(item, container) for item in self.data]
        if not self.name: return "( " + " ".join(content) + " )"
        return self.name + "(" + ",".join(content) + ")"
    
    def __str__(self):
        return self._to_str(self.container)

class Entity:
    def __init__(self, container, name):
        self.container = container
        self.name = name
    
    def __call__(self, *args):
        return Entry(self.container, self.name, args)

class STEP:
    def __init__(self):
        self.cache = {}
        self.entries = []
    
    def __call__(self, data):
        return Entry(self, '', data)
    
    def __getattr__(self, name):
        entity = self.cache.get(name, None)
        if not entity:
            entity = Entity(self, name.upper())
            self.cache[entity.name] = entity
        return entity
    
    def __str__(self):
        return "\n".join(entry.id + " = " + str(entry) + ";" for entry in self.entries)

def get_geometry(brep, step):
    brep_points = brep["points"]
    brep_verts = brep["verts"]
    brep_edges = brep["edges"]
    brep_shells = brep["shells"]
    
    points = []
    for pos in brep_points:
        p = step.CARTESIAN_POINT('', pos)
        points.append(p)
    
    verts = []
    for p_i in brep_verts:
        v = step.VERTEX_POINT('', points[p_i])
        verts.append(v)
    
    def step_edge(v0i, v1i, control_points):
        control_points = [points[i] for i in control_points]
        
        knot_count = ((len(control_points) - 1) // 3) + 1
        multiplicities = [3] * knot_count
        multiplicities[0] = 4
        multiplicities[-1] = 4
        knots = [float(i) for i in range(knot_count)]
        
        # It seems that FreeCAD does not recognize BEZIER_CURVE
        bezier = step.B_SPLINE_CURVE_WITH_KNOTS('', 3, control_points, b".UNSPECIFIED.",
            False, False, multiplicities, knots, b".PIECEWISE_BEZIER_KNOTS.")
        
        return step.EDGE_CURVE('', verts[v0i], verts[v1i], bezier, True)
    
    edges = []
    for v0i, v1i, control_points in brep_edges:
        e = step_edge(v0i, v1i, control_points)
        edges.append(e)
    
    def insert_cone_apex(f, bounds):
        bsurf = f.surface
        if bsurf["type"] != 'CONE': return
        
        # FreeCAD doesn't seem to have a problem when cone has only one boundary
        if len(f.bounds) < 2: return
        
        pos_info = bsurf["position"]
        origin = np.asarray(pos_info["location"])
        axis_z = np.asarray(pos_info["axis"])
        axis_x = np.asarray(pos_info["ref_direction"])
        axis_y = np.cross(axis_z, axis_x)
        matrix_inv = np.array((axis_x, axis_y, axis_z))
        
        if not contains_cone_apex(f, brep_points, brep_edges, origin, matrix_inv): return
        
        radius = bsurf["radius"]
        tan_a = math.tan(bsurf["semi_angle"])
        
        # Somehow, cylinders can still result in a negative winding number
        # (possibly due to errors/problems elsewhere); avoid them explicitly
        if tan_a == 0: return
        
        apex = origin - axis_z * (radius / tan_a)
        
        # circle_radius = 0.0
        # apex += axis_z * (circle_radius / tan_a)
        
        location = step.CARTESIAN_POINT('', tuple(apex))
        # axis = step.DIRECTION('', tuple(axis_z))
        # ref_direction = step.DIRECTION('', tuple(axis_x))
        # placement = step.AXIS2_PLACEMENT_3D('', location, axis, ref_direction)
        
        # circle_point = step.CARTESIAN_POINT('', tuple(apex - axis_x*circle_radius))
        # vertex = step.VERTEX_POINT('', circle_point)
        
        # circle = step.CIRCLE('', placement, circle_radius)
        # edge_curve = step.EDGE_CURVE('', vertex, vertex, circle, True)
        
        # A degenerate spline seems to be sufficient
        vertex = step.VERTEX_POINT('', location)
        control_points = [location, location, location, location]
        bezier = step.B_SPLINE_CURVE_WITH_KNOTS('', 3, control_points, b".UNSPECIFIED.",
            False, False, [4, 4], [0.0, 1.0], b".PIECEWISE_BEZIER_KNOTS.")
        edge_curve = step.EDGE_CURVE('', vertex, vertex, bezier, True)
        
        oriented_edge = step.ORIENTED_EDGE('', b"*", b"*", edge_curve, True)
        edges_loop = [oriented_edge]
        bounds.append(step.FACE_BOUND('', step.EDGE_LOOP('', edges_loop), True))
    
    def face_bounds(f):
        bounds = []
        
        for bound in f.bounds:
            # The 2nd and 3rd arguments are supposed to be start vertex and end vertex,
            # but they are implicit (redefined as derived) in the ORIENTED_EDGE
            edges_loop = [step.ORIENTED_EDGE('', b"*", b"*", edges[ei], same_orientation)
                for ei, same_orientation in bound]
            bounds.append(step.FACE_BOUND('', step.EDGE_LOOP('', edges_loop), True))
        
        insert_cone_apex(f, bounds)
        
        return bounds
    
    def face_surface(f):
        bsurf = f.surface
        surf_type = bsurf["type"]
        
        pos_info = bsurf["position"]
        location = step.CARTESIAN_POINT('', pos_info["location"])
        axis = step.DIRECTION('', pos_info["axis"])
        ref_direction = step.DIRECTION('', pos_info["ref_direction"])
        placement = step.AXIS2_PLACEMENT_3D('', location, axis, ref_direction)
        
        if surf_type == 'PLANE':
            return step.PLANE('', placement)
        
        if surf_type == 'SPHERE':
            radius = bsurf["radius"]
            return step.SPHERICAL_SURFACE('', placement, radius)
        
        if surf_type == 'CONE':
            radius = bsurf["radius"]
            semi_angle = bsurf["semi_angle"]
            if abs(semi_angle) <= epsilon:
                return step.CYLINDRICAL_SURFACE('', placement, radius)
            else:
                return step.CONICAL_SURFACE('', placement, radius, semi_angle)
        
        if surf_type == 'CYCLIDE':
            r_maj = bsurf["major_radius"]
            r_min = bsurf["minor_radius"]
            skew = bsurf["skewness"]
            if skew <= epsilon * r_min:
                return step.TOROIDAL_SURFACE('', placement, r_maj, r_min)
            else:
                return step.DUPIN_CYCLIDE_SURFACE('', placement, r_maj, r_min, skew)
        
        raise Exception(f"Unrecognized surface type {surf_type}")
    
    def step_face(f):
        return step.ADVANCED_FACE('', face_bounds(f), face_surface(f), f.outward)
    
    result = []
    
    for bshell in brep_shells:
        faces = [step_face(f) for f in bshell.faces]
        
        if bshell.is_closed:
            # Note: for a solid with voids inside, brep_with_voids might
            # be preferable? (though MANIFOLD_SOLID_BREP puts no limit
            # on internal voids, so perhaps it's not required)
            shell = step.CLOSED_SHELL('', faces)
            geometry = step.MANIFOLD_SOLID_BREP('', shell)
            result.append(geometry)
        else:
            shell = step.OPEN_SHELL('', faces)
            shells = [shell]
            geometry = step.SHELL_BASED_SURFACE_MODEL('', shells)
            result.append(geometry)
    
    return result

def get_placement(brep, step):
    matrix = brep["matrix"]
    # At least FreeCAD seems to ignore CARTESIAN_TRANSFORMATION_OPERATOR_3D
    origin = step.CARTESIAN_POINT('', matrix["origin"])
    axis_x = step.DIRECTION('', matrix["axis_x"])
    #axis_y = step.DIRECTION('', matrix["axis_y"])
    axis_z = step.DIRECTION('', matrix["axis_z"])
    #scale = matrix["scale"]
    #return step.CARTESIAN_TRANSFORMATION_OPERATOR_3D('', axis_x, axis_y, origin, scale, axis_z)
    return step.AXIS2_PLACEMENT_3D('', origin, axis_z, axis_x)

def make_step_graph(breps):
    # Most of the STEP boilerplate here is just mimicking the results
    # of some FreeCAD exports. I didn't actually bother to study the
    # STEP/EXPRESS format specification in much detail.
    
    # Questions:
    # * How to deal with scale? Even in the specification itself, only uniform scale
    #   is supported (and e.g. FreeCAD seem to ignore the 3D transformation info).
    
    s = STEP() # inline
    step = STEP() # referenced
    
    app_context = step.APPLICATION_CONTEXT(
        'core data for automotive mechanical design processes')
    
    protocol_def = step.APPLICATION_PROTOCOL_DEFINITION('international standard',
      'automotive_design', 2000, app_context)
    
    prod_def_context = step.PRODUCT_DEFINITION_CONTEXT(
        'part definition', app_context, 'design')
    prod_context = step.PRODUCT_CONTEXT('', app_context, 'mechanical')
    
    # Note: Blender's length units only affect how the values are displayed,
    # but the values themselves are supposed to always be in meters
    unit_length = step(( s.NAMED_UNIT(b"*"), s.LENGTH_UNIT(), s.SI_UNIT(b"$", b".METRE.") ))
    unit_plane_angle = step(( s.NAMED_UNIT(b"*"), s.PLANE_ANGLE_UNIT(), s.SI_UNIT(b"$", b".RADIAN.") ))
    unit_solid_angle = step(( s.NAMED_UNIT(b"*"), s.SOLID_ANGLE_UNIT(), s.SI_UNIT(b"$", b".STERADIAN.") ))
    uncertainty = step.UNCERTAINTY_MEASURE_WITH_UNIT(
        s.LENGTH_MEASURE(1.0e-7), unit_length,
        'distance_accuracy_value', 'confusion accuracy')
    context_of_items = step((
        s.GEOMETRIC_REPRESENTATION_CONTEXT(3),
        s.GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT([uncertainty]),
        s.GLOBAL_UNIT_ASSIGNED_CONTEXT([unit_length, unit_plane_angle, unit_solid_angle]),
        s.REPRESENTATION_CONTEXT('Context #1', '3D Context with UNIT and UNCERTAINTY'),
    ))
    
    curve_font = step.DRAUGHTING_PRE_DEFINED_CURVE_FONT('continuous')
    curve_color = step.COLOUR_RGB('', 0.1, 0.1, 0.1)
    curve_style = step.CURVE_STYLE('', curve_font, s.POSITIVE_LENGTH_MEASURE(0.1), curve_color)
    
    fill_color = step.COLOUR_RGB('', 0.8, 0.8, 0.8)
    fill_area_style_color = step.FILL_AREA_STYLE_COLOUR('', fill_color)
    fill_area_style = step.FILL_AREA_STYLE('', [fill_area_style_color])
    surf_style_fill_area = step.SURFACE_STYLE_FILL_AREA(fill_area_style)
    surf_side_style = step.SURFACE_SIDE_STYLE('', [surf_style_fill_area])
    surf_style_usage = step.SURFACE_STYLE_USAGE(b".BOTH.", surf_side_style)
    
    presentation_style_assignment = step.PRESENTATION_STYLE_ASSIGNMENT([surf_style_usage, curve_style])
    
    for brep in breps:
        identifier = brep["name"]
        name = identifier
        
        prod = step.PRODUCT(identifier, name, '', [prod_context])
        prod_def_formation = step.PRODUCT_DEFINITION_FORMATION('', '', prod)
        prod_def = step.PRODUCT_DEFINITION('design', '', prod_def_formation, prod_def_context)
        prod_def_shape = step.PRODUCT_DEFINITION_SHAPE('', '', prod_def)
        prod_category = step.PRODUCT_RELATED_PRODUCT_CATEGORY('part', b"$", [prod])
        
        # Note: FreeCAD/OpenCASCADE also recognizes geometrically_bounded_wireframe_shape_representation
        # https://dev.opencascade.org/doc/overview/html/occt_user_guides__step.html
        
        geometries = get_geometry(brep, step)
        geometric_representations = [get_placement(brep, step)] + geometries
        shape_repr = step.ADVANCED_BREP_SHAPE_REPRESENTATION(
            '', geometric_representations, context_of_items)
        shape_def = step.SHAPE_DEFINITION_REPRESENTATION(prod_def_shape, shape_repr)
        
        for geometry in geometries:
            styled_item = step.STYLED_ITEM('color', [presentation_style_assignment], geometry)
            mech_des_geom_repr = step.MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION(
                '', [styled_item], context_of_items)
    
    return step

def serialize_step(step, **kwargs):
    current_datetime = datetime.datetime.now()
    
    version = kwargs.get("version") or (1, 0, 0)
    if not isinstance(version, str): version = ".".join(str(v) for v in version)
    
    file_descr = kwargs.get("description", 'Blender Model')
    model_type = 'Open CASCADE Shape Model'
    current_datetime_str = current_datetime.strftime("%Y-%m-%dT%H-%M-%S")
    author = kwargs.get("author", 'Author')
    exporter_name = kwargs.get("exporter", f'STEP exporter {version}')
    app_name = kwargs.get("app", 'Blender')
    
    file_schema = 'AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'
    
    entries_str = str(step)
    
    return f"""ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('{file_descr}'),'2;1');
FILE_NAME('{model_type}','{current_datetime_str}',('{author}'),(
    ''),'{exporter_name}','{app_name}','Unknown');
FILE_SCHEMA(('{file_schema}'));
ENDSEC;
DATA;
{entries_str}
ENDSEC;
END-ISO-10303-21;
"""

def export_step(file_path, breps, **kwargs):
    step = make_step_graph(breps)
    result = serialize_step(step, **kwargs)
    
    with open(file_path, "w") as file:
        file.write(result)
