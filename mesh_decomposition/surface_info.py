# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 dairin0d https://github.com/dairin0d

import math

import numpy as np

from .utils import *
from .optimization import nelder_mead
from .plane_fitter import PlaneFitter

class BrepPrimitive:
    type = None
    origin = None
    axis_x = None
    axis_y = None
    axis_z = None
    _matrix_inv = None
    _matrix = None
    
    def get_matrix(self, inverse):
        if self._matrix_inv is None:
            axis_z = ((0.0, 0.0, 1.0) if self.axis_z is None else self.axis_z)
            axis_x = normalize(*orthogonal_3d(*axis_z))
            axis_y = np.cross(axis_z, axis_x)
            self._matrix_inv = np.array((axis_x, axis_y, axis_z))
            self._matrix = self._matrix_inv.T
        return (self._matrix_inv if inverse else self._matrix)
    
    def _fill_brep(self, brep):
        matrix_inv = self.get_matrix(True)
        brep["type"] = self.type
        brep["position"] = dict(
            location=tuple(self.origin),
            axis=tuple(matrix_inv[2]),
            ref_direction=tuple(matrix_inv[0]),
        )
        return brep
    
    def abs_distance(self, points):
        return np.abs(self.distance(points))
    
    def max_distance(self, points):
        return np.max(np.abs(self.distance(points)))
    
    def mean_distance(self, points):
        return np.mean(np.abs(self.distance(points)))
    
    def rms_distance(self, points):
        distances = self.distance(points)
        distances *= distances
        return np.sqrt(np.mean(distances))

class PlanePrimitive(BrepPrimitive):
    type = 'PLANE'
    
    def __init__(self, origin, axis_z):
        self.origin = origin
        self.axis_z = axis_z
    
    def normal(self, point):
        return self.axis_z
    
    def nearest(self, point):
        p = point - self.origin
        return self.origin + (p - self.axis_z * np.dot(p, self.axis_z))
    
    def distance(self, points):
        "Returns signed distances (for each point)"
        p = points - self.origin
        return np.dot(p, self.axis_z)
    
    def to_brep(self):
        return self._fill_brep({})

class SpherePrimitive(BrepPrimitive):
    type = 'SPHERE'
    
    def __init__(self, origin, radius, axis_z=None):
        self.origin = origin
        self.axis_z = (np.array((0.0, 0.0, 1.0)) if axis_z is None else axis_z)
        self.radius = radius
    
    def normal(self, point):
        return np.asarray(normalize(*(point - self.origin)))
    
    def nearest(self, point):
        p = point - self.origin
        mag = norm(p)
        return self.origin + ((p/mag)*self.radius if mag > nonzero else (0, 0, self.radius))
    
    def distance(self, points):
        "Returns signed distances (for each point)"
        p = points - self.origin
        return np.sqrt(np.sum(p*p, axis=1)) - self.radius
    
    def to_brep(self):
        return self._fill_brep({"radius":self.radius})

class ConePrimitive(BrepPrimitive):
    type = 'CONE'
    
    def __init__(self, origin, axis_z, radius, tan_a):
        self.origin = origin
        self.axis_z = axis_z
        self.radius = radius
        self.tan_a = tan_a
    
    def normal(self, point):
        p = point - self.origin
        pz = np.dot(p, self.axis_z)
        prj = pz * self.axis_z
        ort = p - prj
        n = normalize(*ort) - self.axis_z * self.tan_a
        return np.asarray(normalize(*n))
    
    def nearest(self, point):
        p = point - self.origin
        pz = np.dot(p, self.axis_z)
        prj = pz * self.axis_z
        ort = p - prj
        pxy = norm(ort)
        if pxy > nonzero:
            ort /= pxy
        else:
            ort = orthogonal_3d(*self.axis_z)
            mag = norm(ort)
            ort /= mag
        
        o_2d = np.array((self.radius, 0.0))
        n_2d = np.array((1, -self.tan_a))
        n_2d /= norm(n_2d)
        p_2d = np.array((pxy, pz)) - o_2d
        p_2d = o_2d + (p_2d - np.dot(p_2d, n_2d) * n_2d)
        
        return self.origin + ort*p_2d[0] + self.axis_z*p_2d[1]
    
    def distance(self, points):
        "Returns signed distances (for each point)"
        
        p = points - self.origin
        pz = np.dot(p, self.axis_z)
        prj = pz[:, np.newaxis] @ self.axis_z[np.newaxis, :]
        ort = p - prj
        pxy = np.sqrt(np.sum(ort*ort, axis=1))
        # Note: for a cone, geometric distance = algebraic distance / sqrt(1 + tan_a^2)
        return (pxy - (self.radius + self.tan_a*pz)) / sqrt(1.0 + self.tan_a*self.tan_a)
    
    def to_brep(self):
        # FreeCAD doesn't import correctly the cones with negative angle
        if copysign(1, self.tan_a) < 0: # use copysign to hangle the -0.0 case
            self.tan_a = -self.tan_a
            self.axis_z = -self.axis_z
            self._matrix_inv = None
            self._matrix = None
        return self._fill_brep({"radius":self.radius, "semi_angle":atan(self.tan_a)})

class CyclidePrimitive(BrepPrimitive):
    type = 'CYCLIDE'
    
    def __init__(self, origin, axis_z, axis_x, r_maj, r_min, skew):
        self.origin = origin
        self.axis_z = axis_z
        self.axis_x = axis_x
        self.r_maj = r_maj
        self.r_min = r_min
        self.skew = skew
    
    def _nearest_ball(self, px, py):
        # https://en.wikipedia.org/wiki/Dupin_cyclide#Cyclide_as_channel_surface
        r_ort = sqrt(self.r_maj**2 - self.skew**2)
        phi = np.arctan2(py, px)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cx = self.r_maj * cos_phi
        cy = r_ort * sin_phi
        r_phi = self.r_min - self.skew * cos_phi
        return cx, cy, r_phi
    
    def normal(self, point):
        axis_x, axis_y, axis_z = self.get_matrix(True)
        p = point - self.origin
        px = np.dot(p, axis_x)
        py = np.dot(p, axis_y)
        pz = np.dot(p, axis_z)
        
        cx, cy, r_phi = self._nearest_ball(px, py)
        
        ball_center = axis_x*cx + axis_y*cy
        return np.asarray(normalize(*(p - ball_center)))
    
    def nearest(self, point):
        axis_x, axis_y, axis_z = self.get_matrix(True)
        p = point - self.origin
        px = np.dot(p, axis_x)
        py = np.dot(p, axis_y)
        pz = np.dot(p, axis_z)
        
        cx, cy, r_phi = self._nearest_ball(px, py)
        
        ball_center = axis_x*cx + axis_y*cy
        d = p - ball_center
        
        mag = norm(d)
        return self.origin + ball_center + r_phi * (d/mag if mag > nonzero else axis_z)
    
    def distance(self, points):
        "Returns signed distances (for each point)"
        
        axis_x, axis_y, axis_z = self.get_matrix(True)
        p = points - self.origin
        px = np.dot(p, axis_x)
        py = np.dot(p, axis_y)
        pz = np.dot(p, axis_z)
        
        cx, cy, r_phi = self._nearest_ball(px, py)
        
        px -= cx
        py -= cy
        px *= px
        py *= py
        pz *= pz
        return np.sqrt(px+py+pz) - r_phi
    
    def to_brep(self):
        return self._fill_brep({"major_radius":self.r_maj, "minor_radius":self.r_min, "skewness":self.skew})

class SurfaceInfo:
    # Normal estimation fallback values:
    zero_stats = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    default_axis = (0.0, 0.0, 1.0)
    
    # Plane detection settings:
    plane_axis_distance = 1.0 - 1e-3
    plane_tan_a = 1e2
    plane_radius = 1e2
    
    cyclide_chord_deviation = 0.01
    torus_ring_coaxiality = 1.0 - 0.01
    torus_ring_coincidence = 0.01
    
    # Optmimization settings:
    # Note: max distance seems to result in a more accurate optimization
    use_max_distance = True
    optimizer_config = dict(step=0.01)
    
    @classmethod
    def FromTriangle(cls, t, vs_tid, vs_pos):
        vi0, vi1, vi2 = vs_tid[t]
        x0, y0, z0 = vs_pos[vi0]
        x1, y1, z1 = vs_pos[vi1]
        x2, y2, z2 = vs_pos[vi2]
        ax = x1 - x0; ay = y1 - y0; az = z1 - z0
        bx = x2 - x0; by = y2 - y0; bz = z2 - z0
        nx = ay*bz - az*by
        ny = az*bx - ax*bz
        nz = ax*by - ay*bx
        weight = sqrt(nx*nx + ny*ny + nz*nz)
        mag = weight or nonzero
        nx /= mag; ny /= mag; nz /= mag
        
        surf = SurfaceInfo()
        surf.tris = [t]
        surf.normal_stats = (weight * 0.5, nx, ny, nz, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        origin = np.array((x0, y0, z0))
        axis_z = np.array((nx, ny, nz))
        
        surf.primitive = PlanePrimitive(origin, axis_z)
        surf.fit_error = 0.0
        
        return surf
    
    @classmethod
    def merge_normals(cls, stats0, stats1):
        w0, x0, y0, z0, xx0, xy0, xz0, yy0, yz0, zz0 = stats0
        w1, x1, y1, z1, xx1, xy1, xz1, yy1, yz1, zz1 = stats1
        
        # Based on (DOI: 10.1145/3221269.3223036)
        # "Numerically Stable Parallel Computation of (Co-)Variance" (2018)
        w = w0 + w1
        if not w: return cls.zero_stats, np.array(cls.default_axis), 0.0
        x = (x0*w0 + x1*w1) / w
        y = (y0*w0 + y1*w1) / w
        z = (z0*w0 + z1*w1) / w
        dx = x0 - x1
        dy = y0 - y1
        dz = z0 - z1
        ww = w0*w1 / w
        xx = xx0 + xx1 + ww*dx*dx
        xy = xy0 + xy1 + ww*dx*dy
        xz = xz0 + xz1 + ww*dx*dz
        yy = yy0 + yy1 + ww*dy*dy
        yz = yz0 + yz1 + ww*dy*dz
        zz = zz0 + zz1 + ww*dz*dz
        stats = (w, x, y, z, xx, xy, xz, yy, yz, zz)
        
        # To estimate the axis, we can treat normals as points on the
        # Gaussian sphere, and try to fit a plane to them
        
        nx, ny, nz = PlaneFitter.covariance_to_normal(xx, xy, xz, yy, yz, zz, w)
        
        mag2 = nx*nx + ny*ny + nz*nz
        if not mag2: # Can't fit a plane; attempt a cross-product instead
            nx, ny, nz = y0*z1 - z0*y1, z0*x1 - x0*z1, x0*y1 - y0*x1
            mag2 = nx*nx + ny*ny + nz*nz
            if not mag2: # Fall back to the mean
                nx, ny, nz = x, y, z
                mag2 = nx*nx + ny*ny + nz*nz
                if not mag2: return stats, np.array(cls.default_axis), 0.0
        
        mag = sqrt(mag2)
        nx /= mag; ny /= mag; nz /= mag
        distance = nx*x + ny*y + nz*z
        
        if distance < 0:
            nx = -nx; ny = -ny; nz = -nz; distance = -distance
        
        return stats, np.array((nx, ny, nz)), distance
    
    @classmethod
    def gather_points(cls, surf0, surf1, v_data, e_data, tag, points):
        vs_tid, vs_pos, vs_tag = v_data
        es_tid, es_pos, es_tag = e_data
        
        points_count = 0
        
        def add_point(positions, tags, id):
            nonlocal points_count
            if tags[id] == tag: return
            tags[id] = tag
            points[points_count] = positions[id]
            points_count = points_count + 1
        
        for tris in (surf0.tris, surf1.tris):
            for t in tris:
                vi0, vi1, vi2 = vs_tid[t]
                add_point(vs_pos, vs_tag, vi0)
                add_point(vs_pos, vs_tag, vi1)
                add_point(vs_pos, vs_tag, vi2)
        
        verts_count = points_count
        
        for tris in (surf0.tris, surf1.tris):
            for t in tris:
                ei0, ei1, ei2 = es_tid[t]
                add_point(es_pos, es_tag, ei0)
                add_point(es_pos, es_tag, ei1)
                add_point(es_pos, es_tag, ei2)
        
        return verts_count, points_count
    
    @classmethod
    def calc_A_matrix(cls, points, bufs):
        return cls.calc_A_matrix_xyz(points[:, 0], points[:, 1], points[:, 2], bufs)
    
    @classmethod
    def calc_A_matrix_xyz(cls, x, y, z, bufs):
        A = bufs[:len(x), :6]
        A[:, 0] = 1.0
        A[:, 1] = x
        A[:, 2] = y
        A[:, 3] = z
        # x^2 + y^2
        A[:, 4] = A[:, 1]; A[:, 4] *= A[:, 4]
        A[:, 5] = A[:, 2]; A[:, 5] *= A[:, 5]
        A[:, 5] += A[:, 4]
        # z^2
        A[:, 4] = A[:, 3]; A[:, 4] *= A[:, 4]
        return A
    
    @classmethod
    def calc_A_matrix_2d(cls, points, bufs):
        return cls.calc_A_matrix_xy(points[:, 0], points[:, 1], bufs)
    
    @classmethod
    def calc_A_matrix_xy(cls, x, y, bufs):
        A = bufs[:len(x), :6]
        A[:, 0] = 1.0
        A[:, 1] = x
        A[:, 2] = y
        # x^2 + y^2
        A[:, 4] = A[:, 1]; A[:, 4] *= A[:, 4]
        A[:, 3] = A[:, 2]; A[:, 3] *= A[:, 3]
        A[:, 3] += A[:, 4]
        return A[:, :4]
    
    @classmethod
    def direct_fit(cls, A):
        # Based on (DOI: 10.1145/37401.37420)
        # "Direct least-squares fitting of algebraic surfaces" (1987)
        
        n = A.shape[1]
        
        P = A.T @ A
        U = np.zeros(P.shape)
        
        for i in range(0, n):
            p_ii = P[i, i]
            # If matrix is degenerate, we're probably dealing with a plane?
            if not (p_ii > 0): return [0.0] * n
            
            U[i, i] = 1
            for j in range(i+1, n):
                u_ij = P[i, j] / p_ii
                U[i, j] = u_ij
                for k in range(j, n):
                    P[j, k] -= u_ij*P[i, k]
        
        # Last factor is always 1 (by construction), so no need to calculate it
        i = n-1
        U_1 = np.delete(U, i, axis=0)
        det = np.linalg.det
        return [((-1)**(i+j)) * det(np.delete(U_1, j, axis=1)) for j in range(n-1)]
    
    @classmethod
    def extract_params(cls, q):
        # squishy sphere (actual sphere if k == 1):
        # (x^2 + y^2) +   k*(z^2) - 2*x0*x - 2*y0*y - 2*z0*k*z + x0^2 + y0^2 + k*z0^2 - r^2 = 0

        # conical frustum (can represent both cones and cylinders):
        # (also can represent plane as a special case, since we use centered points?)
        # (x^2 + y^2) - t^2*(z^2) - 2*x0*x - 2*y0*y -  2*r*t*z + x0^2 + y0^2 + 0*z0^2 - r^2 = 0
        
        x = -q[1] / 2.0
        y = -q[2] / 2.0
        k = q[4]
        sphere_threshold = 0.25
        if k > sphere_threshold: # sphere
            z = (-q[3] / 2.0) / k
            r = sqrt(max(0.0, x*x + y*y + k*z*z - q[0]))
            return (x, y, z), r, nan, k
        else: # conical frustum
            r = sqrt(max(0.0, x*x + y*y - q[0]))
            tan_a = copysign(sqrt(max(0.0, -k)), -q[3])
            return (x, y, 0), r, tan_a, None
    
    @classmethod
    def optimize(cls, A, offset, radius, tan_a, is_sphere):
        basis = [np.sum(A[:, i]) for i in range(A.shape[1])]
        
        # Note: it appears that optimizing JUST the radius and tan_a for cones/cylinders
        # leads to more accurate results than optimizing all parameters. Optimizing
        # the offset afterwards does not seem to lead to any noticeable improvement.
        # On the contrary, optimizing parameters separately for spheres leads to worse
        # results than optimizing all parameters jointly.
        
        count = A.shape[0]
        
        if is_sphere:
            if cls.use_max_distance:
                axyz2 = A[:, 4] + A[:, 5]
                ax = -2*A[:, 1]
                ay = -2*A[:, 2]
                az = -2*A[:, 3]
                ac = A[:, 0]
                def calc_error(params):
                    x, y, z, r = params
                    d = np.abs(axyz2 + x*ax + y*ay + z*az + (x*x + y*y + z*z - r*r)*ac)
                    return sqrt(np.max(d))
            else:
                bxyz2 = basis[4] + basis[5]
                bx = -2*basis[1]
                by = -2*basis[2]
                bz = -2*basis[3]
                bc = basis[0]
                def calc_error(params):
                    x, y, z, r = params
                    d = abs(bxyz2 + x*bx + y*by + z*bz + (x*x + y*y + z*z - r*r)*bc)
                    return sqrt(d / count)
            
            params = [offset[0], offset[1], offset[2], radius]
            params, error = nelder_mead(calc_error, params, **cls.optimizer_config)
            x, y, z, radius = params
            offset = (x, y, z)
        else:
            # Note: for a cone, geometric distance = algebraic distance / sqrt(1 + tan_a^2)
            if cls.use_max_distance:
                axy2 = A[:, 5]
                az2 = -A[:, 4]
                ax = -2*A[:, 1]
                ay = -2*A[:, 2]
                az = -2*A[:, 3]
                ac = A[:, 0]
                def calc_error(params):
                    x, y, t, r = params
                    t2 = t*t
                    d = np.abs(axy2 + t2*az2 + x*ax + y*ay + (r*t)*az + (x*x + y*y - r*r)*ac)
                    return sqrt(np.max(d) / (1.0 + t2))
            else:
                bxy2 = basis[5]
                bz2 = -basis[4]
                bx = -2*basis[1]
                by = -2*basis[2]
                bz = -2*basis[3]
                bc = basis[0]
                def calc_error(params):
                    x, y, t, r = params
                    t2 = t*t
                    d = abs(bxy2 + t2*bz2 + x*bx + y*by + (r*t)*bz + (x*x + y*y - r*r)*bc)
                    return sqrt((d / count) / (1.0 + t2))
            
            params = [offset[0], offset[1], tan_a, radius]
            params, error = nelder_mead(calc_error, params, mask=[2,3], **cls.optimizer_config)
            x, y, tan_a, radius = params
            offset = (x, y, 0)
        
        return offset, radius, tan_a, error
    
    @classmethod
    def merge_fit(cls, surf0, surf1, v_data, e_data, tag, points, bufs, curv_data, face_error_weight=1.0):
        verts_count, points_count = cls.gather_points(surf0, surf1, v_data, e_data, tag, points)
        src_points = points[:points_count]
        
        normal_stats, primitive = cls._merge_fit(surf0, surf1, src_points, verts_count, bufs)
        
        _points = src_points[:verts_count]
        error = primitive.max_distance(_points)
        
        def fit_cyclide(cyclide):
            nonlocal primitive, error
            if (not cyclide) or (cyclide.type != 'CYCLIDE'): return
            cyclide, cyclide_error = cls.fit_cyclide(cyclide, _points)
            if (cyclide_error < error) or math.isnan(error):
                primitive, error = cyclide, cyclide_error
        
        fit_cyclide(surf0.primitive)
        fit_cyclide(surf1.primitive)
        fit_cyclide(cls.init_cyclide_2_tubes(surf0.primitive, surf1.primitive))
        fit_cyclide(cls.init_cyclide_2_rings(surf0.primitive, surf1.primitive, _points, bufs))
        if len(surf0.tris) + len(surf1.tris) >= 4:
            fit_cyclide(cls.init_cyclide_cone(primitive, _points, bufs))
        
        verts_dist = primitive.max_distance(src_points[:verts_count])
        faces_dist = primitive.max_distance(src_points[verts_count:])
        final_error = max(verts_dist, faces_dist*face_error_weight)
        
        return normal_stats, primitive, final_error
    
    @classmethod
    def _merge_fit(cls, surf0, surf1, src_points, verts_count, bufs):
        center = np.mean(src_points, axis=0)
        points = src_points - center
        
        normal_stats, axis, axis_distance = cls.merge_normals(surf0.normal_stats, surf1.normal_stats)
        
        def fit_plane():
            fit_axis = PlaneFitter.fit(points, None, centered=True)[1]
            if fit_axis is not None: axis = fit_axis
            primitive = PlanePrimitive(center, axis)
            return normal_stats, primitive
        
        if axis_distance >= cls.plane_axis_distance:
            return fit_plane()
        
        axis_matrix, axis_matrix_inv = make_axis_matrix(axis)
        
        # Creating a new array is probably still faster than multiplying each row
        # by matrix in-place (via a python loop)
        aligned = (axis_matrix_inv @ points.T).T
        
        A = cls.calc_A_matrix(aligned, bufs)
        offset, radius, tan_a, sphere_stretch = cls.extract_params(cls.direct_fit(A))
        
        def extent(points):
            dx, dy, dz = np.max(points, axis=0) - np.min(points, axis=0)
            return sqrt(dx*dx + dy*dy + dz*dz) * 0.5
        aligned_extent = extent(aligned)
        
        if (abs(tan_a) >= cls.plane_tan_a) or (radius == 0.0) or (radius > cls.plane_radius * aligned_extent):
            return fit_plane()
        
        #offset, radius, tan_a, error = cls.optimize(A[:verts_count], offset, radius, tan_a, sphere_stretch)
        #origin = center + (axis_matrix @ offset)
        
        if sphere_stretch:
            offset, radius, tan_a, error = cls.optimize(A[:verts_count], offset, radius, tan_a, sphere_stretch)
            
            if radius > cls.plane_radius * aligned_extent:
                return fit_plane()
            
            origin = center + (axis_matrix @ offset)
            primitive = SpherePrimitive(origin, radius, axis)
            return normal_stats, primitive
        
        def _calc_error(A, x, y, t, r):
            axy2 = A[:, 5]
            az2 = -A[:, 4]
            ax = -2*A[:, 1]
            ay = -2*A[:, 2]
            az = -2*A[:, 3]
            ac = A[:, 0]
            
            t2 = t*t
            d = np.abs(axy2 + t2*az2 + x*ax + y*ay + (r*t)*az + (x*x + y*y - r*r)*ac)
            return sqrt(np.max(d) / (1.0 + t2))
        
        def fit_axis(axis):
            axis_matrix, axis_matrix_inv = make_axis_matrix(axis)
            
            # Creating a new array is probably still faster than multiplying each row
            # by matrix in-place (via a python loop)
            aligned = (axis_matrix_inv @ points.T).T
            
            A = cls.calc_A_matrix(aligned, bufs)
            offset, radius, tan_a, sphere_stretch = cls.extract_params(cls.direct_fit(A))
            
            offset, radius, tan_a, error = cls.optimize(A[:verts_count], offset, radius, tan_a, sphere_stretch)
            origin = center + (axis_matrix @ offset)
            
            return origin, radius, tan_a, sphere_stretch, error
            
            #error = _calc_error(A[:verts_count], offset[0], offset[1], tan_a, radius)
            #return A, axis_matrix, offset, radius, tan_a, sphere_stretch, error
        
        def alt_axis():
            is_plane_0 = (surf0.primitive.type == 'PLANE')
            is_plane_1 = (surf1.primitive.type == 'PLANE')
            if is_plane_0 and is_plane_1:
                axis = np.cross(surf0.primitive.axis_z, surf1.primitive.axis_z)
                mag = norm(axis)
                if mag < 1e-3: return None
                axis /= mag
            elif is_plane_0:
                axis = surf1.primitive.axis_z
            elif is_plane_1:
                axis = surf0.primitive.axis_z
            else:
                axis0 = surf0.primitive.axis_z
                axis1 = surf1.primitive.axis_z
                area0 = surf0.normal_stats[0]
                area1 = surf1.normal_stats[0]
                sign = copysign(1.0, np.dot(axis0, axis1))
                axis = axis0*area0 + sign*axis1*area1
                mag = norm(axis)
                if mag < 1e-3: return None
                axis /= mag
            return axis
        
        _points = points[:verts_count]
        
        def parse_params(params):
            ox, oy, oz, ax, ay, az, t, r = params
            a_mag = sqrt(ax*ax + ay*ay + az*az)
            if a_mag > 0:
                ax /= a_mag; ay /= a_mag; az /= a_mag
            else:
                ax, ay, az = 0, 0, 1
            r = abs(r)
            o = np.array((ox, oy, oz))
            a = np.array((ax, ay, az))
            return o, a, t, r
        
        def calc_error(params):
            o, a, t, r = parse_params(params)
            return ConePrimitive(o, a, r, t).max_distance(_points)
        
        def fit_axis_abs(_axis, before):
            nonlocal origin, axis, radius, tan_a, error
            
            _origin, _radius, _tan_a, _sphere_stretch, _error = fit_axis(_axis)
            
            if (not before) or ((_error < error) or math.isnan(error)):
                params = [*(_origin - center), *_axis, _tan_a, _radius]
            else:
                params = [*(origin - center), *axis, tan_a, radius]
            
            params, _error = nelder_mead(calc_error, params, **cls.optimizer_config)
            _offset, _axis, _tan_a, _radius = parse_params(params)
            _origin = center + _offset
            
            if (_error < error) or math.isnan(error):
                origin, axis, radius, tan_a, error = _origin, _axis, _radius, _tan_a, _error
        
        #"""
        offset, radius, tan_a, error = cls.optimize(A[:verts_count], offset, radius, tan_a, sphere_stretch)
        origin = center + (axis_matrix @ offset)
        
        #error = _calc_error(A[:verts_count], offset[0], offset[1], tan_a, radius)
        
        _axis = alt_axis()
        _axis2 = None
        _axis3 = None
        if _axis is not None:
            _axis2 = np.cross(_axis, axis)
            _axis2_mag = norm(_axis2)
            if _axis2_mag > 0:
                _axis2 /= _axis2_mag
                _axis3 = np.cross(_axis2, axis)
            else:
                _axis2 = None
        
        if _axis2 is not None:
            fit_axis_abs(_axis, True)
            #fit_axis_abs(_axis2, False)
            #fit_axis_abs(_axis3, False)
            
            """
            _origin, _radius, _tan_a, _sphere_stretch, _error = fit_axis(_axis)
            if _error < error:
                origin, axis, radius, tan_a, error = _origin, _axis, _radius, _tan_a, _error
            
            _axis = _axis2
            _origin, _radius, _tan_a, _sphere_stretch, _error = fit_axis(_axis)
            if _error < error:
                origin, axis, radius, tan_a, error = _origin, _axis, _radius, _tan_a, _error
            #"""
            
            '''
            params = [*(origin - center), *axis, tan_a, radius]
            params, error = nelder_mead(calc_error, params, **cls.optimizer_config)
            offset, axis, tan_a, radius = parse_params(params)
            origin = center + offset
            '''
        else:
            #offset, radius, tan_a, error = cls.optimize(A[:verts_count], offset, radius, tan_a, sphere_stretch)
            #origin = center + (axis_matrix @ offset)
            #"""
            
            params = [*(origin - center), *axis, tan_a, radius]
            params, error = nelder_mead(calc_error, params, **cls.optimizer_config)
            offset, axis, tan_a, radius = parse_params(params)
            origin = center + offset
        
        if abs(tan_a) >= cls.plane_tan_a:
            return fit_plane()
        
        origin = np.asarray(origin)
        axis = np.asarray(axis)
        primitive = ConePrimitive(origin, axis, radius, tan_a)
        
        return normal_stats, primitive
    
    @classmethod
    def init_cyclide_cone(cls, cone, points, bufs):
        if cone.type != 'CONE': return None
        
        points = points - cone.origin
        
        axis_matrix, axis_matrix_inv = make_axis_matrix(cone.axis_z)
        aligned = (axis_matrix_inv @ points.T).T
        xy = aligned[:, :2]
        z = aligned[:, 2]
        min_z = np.min(z)
        max_z = np.max(z)
        
        xy_mag = np.linalg.norm(xy, axis=1)
        r = cone.radius + z * cone.tan_a
        rel_chord_deviation = np.max(np.abs(xy_mag - r)) / max(cone.radius, max_z - min_z)
        if rel_chord_deviation <= cls.cyclide_chord_deviation: return None
        
        # Convert points to deviation vectors
        xy *= (1.0 - np.nan_to_num(r / xy_mag, copy=False))[:, np.newaxis]
        
        plane_axis = PlaneFitter.fit(aligned, None)[1]
        plane_matrix, plane_matrix_inv = make_axis_matrix(plane_axis)
        plane_points = (plane_matrix_inv @ aligned.T).T
        plane_points[:, 2] = 0.0 # make sure they lie in plane
        
        q = cls.direct_fit(cls.calc_A_matrix_2d(plane_points, bufs))
        center = np.array((-q[1] / 2.0, -q[2] / 2.0, 0.0))
        r_maj = sqrt(max(0.0, np.dot(center, center) - q[0]))
        
        r_min = cone.radius
        s = 0.0 # consider only torus for now
        
        start_i = (np.argmax(z) if cone.tan_a < 0 else np.argmin(z))
        axis_x = np.array(normalize(*(plane_points[start_i] - center)))
        
        origin = plane_matrix @ center
        axis_z = plane_axis
        axis_x = plane_matrix @ axis_x
        
        origin = (axis_matrix @ origin) + cone.origin
        axis_z = axis_matrix @ axis_z
        axis_x = axis_matrix @ axis_x
        
        return CyclidePrimitive(origin, axis_z, axis_x, r_maj, r_min, s)
    
    @classmethod
    def init_cyclide_2_rings(cls, cone0, cone1, points, bufs):
        if not ((cone0.type == 'CONE') and (cone1.type == 'CONE')): return None
        
        if abs(cone1.tan_a) < abs(cone0.tan_a):
            cone0, cone1 = cone1, cone0
        
        if abs(np.dot(cone0.axis_z, cone1.axis_z)) < cls.torus_ring_coaxiality: return None
        
        origin_delta = cone1.origin - cone0.origin
        origin_offset = np.dot(origin_delta, cone0.axis_z)
        origin_deviation = origin_delta - cone0.axis_z * origin_offset
        relative_origin_deviation = norm(origin_deviation) / max(cone0.radius, cone1.radius)
        if relative_origin_deviation > cls.torus_ring_coincidence: return None
        
        return cls.init_cyclide_ring(cone0, points, bufs)
    
    @classmethod
    def init_cyclide_ring(cls, cone, points, bufs):
        points = points - cone.origin
        
        axis_matrix, axis_matrix_inv = make_axis_matrix(cone.axis_z)
        aligned = (axis_matrix_inv @ points.T).T
        r = np.linalg.norm(aligned[:, :2], axis=1)
        z = aligned[:, 2]
        
        q = cls.direct_fit(cls.calc_A_matrix_xy(r, z, bufs))
        cr = -q[1] / 2.0
        cz = -q[2] / 2.0
        r_min = sqrt(max(0.0, cr*cr + cz*cz - q[0]))
        
        if r_min <= 0.0: return None
        
        r_maj = cr
        s = 0.0
        
        axis_z = cone.axis_z
        axis_x = axis_matrix_inv[0]
        origin = cone.origin + axis_z * cz
        
        return CyclidePrimitive(origin, axis_z, axis_x, r_maj, r_min, s)
    
    @classmethod
    def shortest_line_between(cls, a0, an, b0, bn):
        # Adapted from https://math.stackexchange.com/questions/1993953/
        # an and bn are expected to be normalized
        cn = np.cross(an, bn)
        mag = norm(cn)
        if mag > epsilon:
            cn /= mag
            rhs = b0 - a0
            lhs = np.array([an, -bn, cn]).T
            at, bt, d = np.linalg.solve(lhs, rhs)
        else:
            at = np.dot(an, b0 - a0) * 0.5
            bt = np.dot(bn, a0 - b0) * 0.5
            cn = None
        return at, bt, cn
    
    @classmethod
    def init_cyclide_2_tubes(cls, cone0, cone1):
        if not ((cone0.type == 'CONE') and (cone1.type == 'CONE')): return None
        
        o0, a0 = cone0.origin, cone0.axis_z
        o1, a1 = cone1.origin, cone1.axis_z
        t0, t1, axis_z = cls.shortest_line_between(o0, a0, o1, a1)
        if axis_z is None: return None
        
        pM = (o0+a0*t0 + o1+a1*t1) * 0.5
        dA, dB = ((-a1*t1, a0*t1) if abs(t1) > abs(t0) else (-a0*t0, a1*t0))
        if np.dot(dA, dB) > 0: dB = -dB
        
        cm = (dA+dB) * 0.5 # chord midpoint
        h = norm(cm) # deviation from chord
        c = norm(cm - dA) # chord half-length
        if not (c < h * cls.plane_tan_a): return None
        
        r_maj = (h*h + c*c) / (2*h) # cyclide major radius
        
        # Assuming that origins tend to be near the middle of the surface patches,
        # actual radius is more likely to be twice as big.
        r_maj *= 2.0
        
        cn = cm / norm(cm)
        cc = pM + cn * r_maj # cyclide center
        
        def calc_r(cone, p):
            z = np.dot(p - cone.origin, cone.axis_z)
            return abs(cone.radius + cone.tan_a * z)
        
        pA = pM + dA
        rA = calc_r(cone0, pA)
        pB = pM + dB
        rB = calc_r(cone1, pB)
        rM = (calc_r(cone0, pM) + calc_r(cone1, pM)) * 0.5
        
        if copysign(1, rM-rA) == copysign(1, rM-rB):
            p0, r0, p1, r1 = pM, rM, pA, (rA + rB) * 0.5
        else:
            p0, r0, p1, r1 = ((pA, rA, pB, rB) if rA < rB else (pB, rB, pA, rA))
        
        # Technically, Dupin cyclides can be nonmanifold, but that case is not relevant here
        n0 = p0 - cc; n0 /= norm(n0)
        n1 = p1 - cc; n1 /= norm(n1)
        angle = acos(np.dot(n0, n1))
        u = angle / math.pi
        r1 = (max(r0 + (r1 - r0) / u, 0) if u > 0 else r0)
        
        r_min = (r0 + r1) * 0.5 # minor radius
        s = (max(r0, r1) - min(r0, r1)) * 0.5 # skewness
        axis_x = (p0 - cc if r0 < r1 else cc - p0)
        r_maj = norm(axis_x) # just to make sure it's consistent
        axis_x /= r_maj
        
        return CyclidePrimitive(cc, axis_z, axis_x, r_maj, r_min, s)
    
    @classmethod
    def fit_cyclide(cls, cyclide, points):
        def parse_params(params):
            origin = params[0:3]
            axis_z = params[3:6]
            axis_x = params[6:9]
            r_maj = max(params[9], 0.0)
            r_min = max(params[10], 0.0)
            skew = min(max(params[11], 0.0), min(r_maj, r_min))
            axis_z = axis_z / norm(axis_z)
            axis_x = axis_x - (np.dot(axis_x, axis_z) * axis_z)
            axis_x = axis_x / norm(axis_x)
            #"""
            # Constrain the solution coordinates to orthonormal values
            params[3:6] = axis_z
            params[6:9] = axis_x
            params[9] = r_maj
            params[10] = r_min
            params[11] = skew
            #"""
            return origin, axis_z, axis_x, r_maj, r_min, skew
        
        def calc_error(params):
            origin, axis_z, axis_x, r_maj, r_min, skew = parse_params(params)
            return CyclidePrimitive(origin, axis_z, axis_x, r_maj, r_min, skew).max_distance(points)
        
        params = np.array([*cyclide.origin, *cyclide.axis_z, *cyclide.axis_x, cyclide.r_maj, cyclide.r_min, cyclide.skew])
        #print("params before:", list(params))
        #print("cyclide error before:", calc_error(params))
        
        # Allow only toroidal surfaces for now
        params[-1] = 0
        mask = list(range(len(params)))[:-1]
        
        params, error = nelder_mead(calc_error, params, mask=mask, **cls.optimizer_config)
        #print("params after:", list(params))
        #print("cyclide error after:", error)
        
        origin, axis_z, axis_x, r_maj, r_min, skew = parse_params(params)
        
        return CyclidePrimitive(origin, axis_z, axis_x, r_maj, r_min, skew), error
    
    def nearest_point(self, p):
        return self.primitive.nearest(p)
    
    def normal(self, p):
        return self.primitive.normal(p)
    
    def to_brep(self):
        return self.primitive.to_brep()
