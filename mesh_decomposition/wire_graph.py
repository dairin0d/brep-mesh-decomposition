# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 dairin0d https://github.com/dairin0d

import math
import bisect

import numpy as np

from .utils import *
from .optimization import nelder_mead
from .bezier_fitting import BesierFitter

def calc_tangent(pL, pM, pR, hard_angle_cos):
    nL = pL - pM
    magL = norm(nL)
    if magL > 0.0: nL /= magL
    
    nR = pR - pM
    magR = norm(nR)
    if magR > 0.0: nR /= magR
    
    cos_angle = np.dot(nL, -nR)
    if cos_angle <= hard_angle_cos: return np.zeros(3)
    
    tangent = pR - pL
    mag_t_limit = min(magL, magR)
    mag_t = norm(tangent)
    scale = (mag_t_limit/mag_t if mag_t > mag_t_limit else 1.0)
    tangent *= scale / 3.0
    
    return tangent

def get_2_normals(normals):
    best_ij = (0, -1)
    best_dot = 1.0
    
    count = len(normals)
    for i in range(count-1):
        for j in range(i+1, count):
            ij_dot = abs(np.dot(normals[i], normals[j]))
            if ij_dot < best_dot:
                best_dot = ij_dot
                best_ij = (i, j)
    
    return normals[best_ij[0]], normals[best_ij[1]], best_dot

def align_tangent(surfaces, p, tangent, ref_direction, post_normalize, smin=0.8, smax=0.9):
    surf_count = len(surfaces)
    normals = [surf.normal(p) for surf in surfaces]
    
    if surf_count == 1:
        tangent = tangent - normals[0] * np.dot(normals[0], tangent)
    else:
        normalA, normalB, similarity = get_2_normals(normals)
        
        # if np.dot(normalA, normalB) < 0: normalB = -normalB
        # normalAB = np.asarray(normalize(*(normalA + normalB)))
        # tangentP = tangent - normalAB * np.dot(normalAB, tangent)
        # tangent = tangentP
        
        factor = (similarity - smin) / (smax - smin)
        factor = min(max(factor, 0.0), 1.0)
        
        tangentAB = np.asarray(normalize(*np.cross(normalA, normalB)))
        if not post_normalize: tangentAB *= np.dot(tangentAB, tangent)
        if np.dot(tangentAB, ref_direction) < 0: tangentAB = -tangentAB
        
        normalAB = np.asarray(normalize(*(normalA + normalB)))
        tangentP = tangent - normalAB * np.dot(normalAB, tangent)
        if post_normalize: tangentP = np.asarray(normalize(*tangentP))
        if np.dot(tangentP, ref_direction) < 0: tangentP = -tangentP
        
        tangent = tangentAB * (1-factor) + tangentP * factor
    
    if post_normalize: tangent = np.asarray(normalize(*tangent))
    
    if np.dot(tangent, ref_direction) < 0: tangent = -tangent
    
    return tangent

def make_bezier_error_calc():
    u_values = (0.25, 0.5, 0.75)
    # u_values = (0.125, 10/6.0, 0.25, 2.0/6.0, 0.375, 0.5, 0.625, 4.0/6.0, 0.75, 5.0/6.0, 0.875)
    coefs = [((1-u)**3, 3*(1-u)**2*u, 3*(1-u)*u**2, u**3) for u in u_values]
    
    # Note: max error results in a much more accurate fit than mean or RMS errors
    def calc_error_base(surfaces, p0, p1, p2, p3):
        error = 0.0
        for u0, u1, u2, u3 in coefs:
            p_src = p0*u0 + p1*u1 + p2*u2 + p3*u3
            for surf in surfaces:
                p_dst = surf.nearest_point(p_src)
                delta_mag = norm(p_dst - p_src)
                error = max(error, delta_mag)
        return error
    
    return calc_error_base

bezier_error = make_bezier_error_calc()

def optimize_tangents(surfaces, p0, p3, t1, t2):
    delta_mag = norm(p3 - p0)
    if delta_mag == 0: return np.zeros(3), np.zeros(3)
    
    t1_mag = max(norm(t1), epsilon)
    t2_mag = max(norm(t2), epsilon)
    # delta_mag_scale = 1
    delta_mag_scale = 0.5
    params_max = (delta_mag_scale*divide(delta_mag, t1_mag), delta_mag_scale*divide(delta_mag, t2_mag))
    
    def calc_error(params):
        p1 = p0 + t1 * min(abs(params[0]), params_max[0])
        p2 = p3 + t2 * min(abs(params[1]), params_max[1])
        return bezier_error(surfaces, p0, p1, p2, p3)
    
    # params = ((1.0, 1.0) if len(surfaces) == 1 else (0.0, 0.0))
    params = ((1.0, 1.0) if len(surfaces) == 1 else (params_max[0]/3.0, params_max[1]/3.0))
    params, error = nelder_mead(calc_error, params, step=0.1)
    
    t1 = t1 * min(abs(params[0]), params_max[0])
    t2 = t2 * min(abs(params[1]), params_max[1])
    
    # if len(surfaces) > 1:
    #     def calc_error(params):
    #         p1 = p0 + params[:3]
    #         p2 = p3 + params[3:]
    #         return bezier_error(surfaces, p0, p1, p2, p3)
        
    #     params = [*t1, *t2]
    #     params, error = nelder_mead(calc_error, params, step=delta_mag*0.01)
        
    #     t1 = params[:3]
    #     t2 = params[3:]
    
    return t1, t2

def optimize_tangents_continuous(surfaces, pL, tL, pM, tM, pR, tR):
    mag1 = norm(pM-pL)
    mag2 = norm(pR-pM)
    delta_mag_scale = 1
    # delta_mag_scale = 0.5
    params_max = delta_mag_scale*divide(min(mag1, mag2), norm(tM))
    
    ptL = pL+tL
    ptR = pR+tR
    
    def calc_error(params):
        scaleL = min(abs(params[0]), params_max)
        scaleR = min(abs(params[1]), params_max)
        errorL = bezier_error(surfaces, pL, ptL, pM-tM*scaleL, pM)
        errorR = bezier_error(surfaces, pM, pM+tM*scaleR, ptR, pR)
        return max(errorL, errorR)
    
    params = [1.0, 1.0]
    params, error = nelder_mead(calc_error, params, step=0.01)
    
    scaleL = min(abs(params[0]), params_max)
    scaleR = min(abs(params[1]), params_max)
    return -tM * scaleL, tM * scaleR

def closest_average(pos, surfaces):
    if len(surfaces) == 0: return pos
    new_pos = np.zeros(3)
    weight = 1.0 / len(surfaces)
    for surf in surfaces:
        new_pos += surf.nearest_point(pos) * weight
    return new_pos

def mean_distance(pos, surfaces):
    d = 0.0
    for surf in surfaces:
        delta = surf.nearest_point(pos) - pos
        d += norm(delta)
    return d / len(surfaces)

def snap_to_surfaces(pos, surfaces, snap_iters=8, mean_iters=8, use_optimizer=False):
    if use_optimizer:
        if len(surfaces) == 1:
            pos[:] = closest_average(pos, surfaces)
        else:
            def calc_error(pos):
                d = 0.0
                for surf in surfaces:
                    delta = surf.nearest_point(pos) - pos
                    d = max(d, norm(delta))
                return d
            
            pos[:] = closest_average(pos, surfaces)
            r = mean_distance(pos, surfaces)
            new_pos, error = nelder_mead(calc_error, pos, step=r*0.1)
            pos[:] = new_pos
    else:
        if len(surfaces) <= 1:
            snap_iters = 1
            mean_iters = 0
        
        for i in range(snap_iters):
            new_pos = pos
            for surf in surfaces:
                new_pos = surf.nearest_point(new_pos)
            pos[:] = new_pos
        
        for i in range(mean_iters):
            new_pos = np.zeros(3)
            weight = 1.0 / len(surfaces)
            for surf in surfaces:
                new_pos += surf.nearest_point(pos) * weight
            pos[:] = new_pos
    
    return pos

def discretize_bezier(result, bezier, resolution):
    if len(result) == 0:
        result.append(bezier[0])
    
    queue = [bezier]
    
    while queue:
        p0, p1, p2, p3 = queue.pop()
        
        lenA = norm(p3 - p0)
        if lenA <= resolution:
            lenB = norm(p1 - p0) + norm(p2 - p1) + norm(p3 - p2)
            if (lenB - lenA) <= resolution:
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

def calc_path_lengths(points):
    path_lengths = [0.0]
    for p_i in range(1, len(points)):
        p0 = points[p_i-1]
        p1 = points[p_i]
        path_lengths.append(path_lengths[p_i-1] + norm(p1 - p0))
    return path_lengths

def calc_variance_dimensions(points):
    try:
        centered = points - np.mean(points, axis=0)
        U, S, Vh = np.linalg.svd(centered, full_matrices=False)
        n = len(S)
        d0 = (S[0] if n > 0 else 0.0)
        d1 = (S[1] if n > 1 else 0.0)
        d2 = (S[2] if n > 2 else 0.0)
        return ((1.0, d1/d0, d2/d0) if d0 > 0 else (0.0, 0.0, 0.0))
    except np.linalg.LinAlgError:
        return (0.0, 0.0, 0.0)

def smooth_polyline(points, factor=0.5):
    result = [None] * len(points)
    result[0] = points[0]
    result[-1] = points[-1]
    
    for p_i in range(1, len(points)-1):
        pL = points[p_i-1]
        pM = points[p_i]
        pR = points[p_i+1]
        result[p_i] = pM + ((pL+pR)*0.5 - pM)*factor
    
    return result

def resample_polyline(points, path_lengths, count):
    result = [points[0]]
    
    count = max(count, 2)
    length_scale = path_lengths[-1] / count
    for i in range(1, count):
        length = length_scale * i
        p_i = bisect.bisect_left(path_lengths, length)
        p0 = points[p_i-1]
        p1 = points[p_i]
        l0 = path_lengths[p_i-1]
        l1 = path_lengths[p_i]
        t = (length - l0) / (l1 - l0)
        result.append(p0*(1 - t) + p1*t)
    
    result.append(points[-1])
    
    return result

def upsample_line(result, p0, p1, resolution, adjuster):
    if len(result) == 0:
        result.append(p0)
    
    queue = [(p0, p1)]
    
    while queue:
        p0, p1 = queue.pop()
        
        lenA = norm(p1 - p0)
        if lenA <= resolution:
            result.append(p1)
            continue
        
        pM = adjuster((p0 + p1) * 0.5)
        queue.append((p0, pM))
        queue.append((pM, p1))

class WireVertex:
    def __init__(self, pos):
        self.pos = np.copy(pos)
        self.edges = set()
    
    def get_surfaces(self):
        return {loop.face.surf for edge in self.edges for loop in edge.loops}

class WireEdge:
    def __init__(self, v0, v1, points):
        self.v0 = v0
        self.v1 = v1
        self.is_closed = (v0 is None)
        self.loops = set()
        
        self.points = points
        
        if self.is_closed:
            self.points.append(self.points[0])
        else:
            self.points.insert(0, v0.pos)
            self.points.append(v1.pos)
        
        self.tangents = [[np.zeros(3), np.zeros(3)] for i in range(len(self.points) - 1)]
        
        if v0: v0.edges.add(self)
        if v1: v1.edges.add(self)
    
    def get_beziers(self):
        for seg_i in range(len(self.tangents)):
            p0 = self.points[seg_i]
            p3 = self.points[seg_i+1]
            t1, t2 = self.tangents[seg_i]
            p1, p2 = p0+t1, p3+t2
            yield (p0, p1, p2, p3)
    
    def set_beziers(self, beziers):
        new_points = []
        new_tangents = []
        new_points.append(self.points[0])
        for i, bezier in enumerate(beziers):
            p0, p1, p2, p3 = bezier
            if i > 0: new_points.append(p0)
            new_tangents.append((p1 - p0, p2 - p3))
        new_points.append(self.points[-1])
        
        self.points = new_points
        self.tangents = new_tangents
    
    def get_surfaces(self):
        return {loop.face.surf for loop in self.loops}
    
    def calc_tangents(self, hard_angle_cos=-1):
        if self.is_closed:
            pL, pM = self.points[-2], self.points[0]
            i_start, i_stop = 1, len(self.points)
        else:
            pL, pM = self.points[0], self.points[1]
            i_start, i_stop = 2, len(self.points) - 1
        
        for i in range(i_start, i_stop):
            pR = self.points[i]
            tangent = calc_tangent(pL, pM, pR, hard_angle_cos)
            self.tangents[i-1][0] = tangent
            self.tangents[i-2][1] = -tangent
            pL, pM = pM, pR

class WireLoop:
    def __init__(self, face, edges):
        self.face = face
        self.edges = edges # each item is (edge, aligned)
        
        face.loops.add(self)
        
        for edge, aligned in edges:
            edge.loops.add(self)

class WireFace:
    def __init__(self, surf, loops, is_manifold):
        self.surf = surf
        self.is_manifold = is_manifold
        self.loops = set()
        
        for loop in loops:
            if isinstance(loop, WireLoop): continue
            WireLoop(self, loop) # adds to self.loops

class WireGraph:
    def __init__(self, begin=None, advance=None):
        self.verts = set()
        self.edges = set()
        self.faces = set()
        
        self.begin = begin or (lambda text, count: None)
        self.advance = advance or (lambda: None)
    
    def add_vert(self, pos):
        vert = WireVertex(pos)
        self.verts.add(vert)
        return vert
    
    def add_edge(self, v0, v1, points):
        edge = WireEdge(v0, v1, points)
        self.edges.add(edge)
        return edge
    
    def add_face(self, surf, loops, is_manifold):
        face = WireFace(surf, loops, is_manifold)
        self.faces.add(face)
        return face
    
    def calculate_tangents(self, hard_angle):
        hard_angle_cos = cos(hard_angle)
        
        self.begin("Wire Tangents", len(self.edges))
        for e in self.edges:
            self.advance()
            
            # The source edge spline is only relevant for open-shell boundary edges
            if len(e.loops) > 1: continue
            
            e.calc_tangents(hard_angle_cos)
            
            # if e.is_closed:
            #     pL, pM = e.points[-2], e.points[0]
            #     i_start, i_stop = 1, len(e.points)
            # else:
            #     pL, pM = e.points[0], e.points[1]
            #     i_start, i_stop = 2, len(e.points) - 1
            
            # for i in range(i_start, i_stop):
            #     pR = e.points[i]
            #     tangent = calc_tangent(pL, pM, pR, hard_angle_cos)
            #     e.tangents[i-1][0] = tangent
            #     e.tangents[i-2][1] = -tangent
            #     pL, pM = pM, pR
    
    def fit(self, **kwargs):
        def get_option(name, default):
            result = kwargs.get(name)
            return (default if result is None else result)
        
        options = dict(
            resample_iterations=get_option("resample_iterations", 3),
            smooth_iterations=get_option("smooth_iterations", 5),
            smooth_factor=min(max(get_option("smooth_factor", 0.25), 0.0), 0.5),
            min_samples=max(get_option("min_samples", 0), 0),
            dimension_samples=max(get_option("dimension_samples", 7), 0),
            align_tangents=get_option("align_tangents", True),
        )
        
        self.begin("Fit Wire Verts", len(self.verts))
        for v in self.verts:
            snap_to_surfaces(v.pos, v.get_surfaces())
            self.advance()
        
        self.begin("Fit Wire Edges", len(self.edges))
        for e in self.edges:
            self.fit_edge(e, options)
            self.advance()
        
        max_error = 0.0
        avg_error = 0.0
        error_count = 0
        for e in self.edges:
            surfaces = e.get_surfaces()
            for seg_i in range(len(e.tangents)):
                t1, t2 = e.tangents[seg_i]
                p0 = e.points[seg_i]
                p3 = e.points[seg_i+1]
                error = bezier_error(surfaces, p0, p0+t1, p3+t2, p3)
                max_error = max(max_error, error)
                avg_error += error
                error_count += 1
        
        print(f"Edge fit error: max={max_error}, mean={avg_error/max(error_count, 1)}")
    
    def fit_edge(self, e, options):
        surfaces = e.get_surfaces()
        
        def snap_points(points, *args, **kwargs):
            for p_i in range(1, len(points) - 1):
                snap_to_surfaces(points[p_i], surfaces, *args, **kwargs)
        
        snap_points(e.points)
        
        is_internal_edge = (len(e.loops) > 1)
        
        # Polyline smoothing, then fitting beziers to it?
        # (uniformly distributed number of splines, since
        # dynamic bezier fitting doesn't work very well)
        
        # Finding edge self-intersections, and doing a "boolean" on them?
        
        if is_internal_edge:
            resample_iterations = options["resample_iterations"]
            smooth_iterations = options["smooth_iterations"]
            smooth_factor = options["smooth_factor"]
            min_samples = options["min_samples"]
            dimension_samples = options["dimension_samples"]
            
            for smooth_iteration in range(smooth_iterations):
                e.points = smooth_polyline(e.points, factor=smooth_factor)
                snap_points(e.points, snap_iters=0, mean_iters=1)
            # snap_points(e.points)
            
            for resample_iteration in range(resample_iterations):
                # for smooth_iteration in range(smooth_iterations):
                #     e.points = smooth_polyline(e.points, factor=smooth_factor)
                #     snap_points(e.points)
                
                # Do a coarse resampling to eliminate potential irregularities
                path_lengths = calc_path_lengths(e.points)
                if path_lengths[-1] > 0: # apparently can happen
                    var_dim = calc_variance_dimensions(e.points)
                    count = 2 + round(max(min_samples, (var_dim[1] + var_dim[2]) * dimension_samples))
                    e.points = resample_polyline(e.points, path_lengths, count)
                    e.tangents = [[np.zeros(3), np.zeros(3)] for i in range(len(e.points) - 1)]
                
                snap_points(e.points)
                
                if path_lengths[-1] == 0: # apparently can happen
                    break
            
            for smooth_iteration in range(smooth_iterations):
                e.points = smooth_polyline(e.points, factor=smooth_factor)
                snap_points(e.points, snap_iters=0, mean_iters=1)
            
            snap_points(e.points)
            
            e.calc_tangents()
        
        if is_internal_edge and not options["align_tangents"]: return
        
        for seg_i in range(len(e.tangents)):
            t1, t2 = e.tangents[seg_i]
            p0 = e.points[seg_i]
            p3 = e.points[seg_i+1]
            # if is_internal_edge:
            #     t1 = (p3-p0) / 3.0
            #     t2 = (p0-p3) / 3.0
            # t1 = align_tangent(surfaces, p0, t1, p3-p0, is_internal_edge)
            # t2 = align_tangent(surfaces, p3, t2, p0-p3, is_internal_edge)
            # t1 = align_tangent(surfaces, p0, t1, p3-p0, False)
            # t2 = align_tangent(surfaces, p3, t2, p0-p3, False)
            t1, t2 = optimize_tangents(surfaces, p0, p3, t1, t2)
            # t1 = align_tangent(surfaces, p0, t1, p3-p0, False)
            # t2 = align_tangent(surfaces, p3, t2, p0-p3, False)
            e.tangents[seg_i] = [t1, t2]
        
        if is_internal_edge:
            is_looped = (tuple(e.points[0]) == tuple(e.points[-1]))
            p_i_max = len(e.points) - 1
            seg_i_max = len(e.tangents) - 1
            
            def get_point_segments(p_i):
                if is_looped and (p_i == 0): return seg_i_max, 0
                if p_i == 0: return None, 0
                if p_i == p_i_max: return None, seg_i_max
                return p_i - 1, p_i
            
            for p_i in range(len(e.points) - int(is_looped)):
                seg_l, seg_r = get_point_segments(p_i)
                if (seg_l is None) or (seg_r is None): continue
                tL1, tL2 = e.tangents[seg_l]
                tR1, tR2 = e.tangents[seg_r]
                pL = e.points[seg_l]
                pM = e.points[seg_r]
                pR = e.points[seg_r+1]
                tL2, tR1 = optimize_tangents_continuous(surfaces, pL, tL1, pM, (tR1-tL2)*0.5, pR, tR2)
                e.tangents[seg_l] = (tL1, tL2)
                e.tangents[seg_r] = (tR1, tR2)

"""
1. Fit edges to surfaces
    * For open-shell boundary edges (single loop):
        * Extract splines (smooth where angle < limit, sharp otherwise)
    * Snap/fit vertices to the corresponding surfaces
    * For open-shell boundary edges (single loop):
        * Snap/fit the splines to the (single) surface
    * For multi-surface edges (multiple loops):
        * Ideally, we would ignore the source polyline and use primitive intersections
        * However, since primitive fits can be inaccurate, sensible intersections are not guaranteed
        * Fit polyline points to the surfaces
        * Resample to avoid potentially criss-crossing edges (8 samples should be enough?)
        * Fit the resampled polyline; refine until deviation is sufficiently small (or fit to surfaces as splines)
2. Split by parametric boundaries [NOT RELEVANT: the issue was due to negative cone angles]
3. Where necessary, "bevel" edges and vertices
    * Needs information about sharpness/smoothness
    * Needs to take care not to extend beyond other boundaries
"""
