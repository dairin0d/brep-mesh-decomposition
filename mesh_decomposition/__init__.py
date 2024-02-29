# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 dairin0d https://github.com/dairin0d

import math

import numpy as np

from .utils import *
from .priority_heap import PriorityHeap
from .surface_info import SurfaceInfo
from .wire_graph import WireGraph

# Note: it seems that for operations on individual vectors, using
# plain local variables is the fastest option, much faster than numpy
# (and for storage, (un)packing local vars from/to tuples is fastest)
# Built-in python functions like sqrt or sum are also faster for scalars

# It appears that numpy operations have a significant overhead for small
# arrays. An equivalent calculation in pure Python seems to be faster
# when the number of elements is <= 16 (more or less).

# "Struct of arrays" seems to be somewhat faster than "array of structs"

# Tuple/list unpacking seems noticeably faster than any method of
# accessing part(s) of complex data, such as:
# * object fields (both with and without __slots__)
# * named tuples (by name, by index, unpacking)
# * getting from dictionary by key
# * getting from tuple/list by index
# In fact, only starting from 8 elements or higher the index access
# becomes more efficient than tuple/list unpacking

def MeshDecomposer():
    progress = 0
    progress_max = 1
    progress_text = ""
    progress_info = ""
    
    vertices = None
    triangles = None
    triangle_edges = None
    opposites = None
    edge_midpoints = None
    boundaries = None
    tri_infos = None
    corner_infos = None
    clusters = None
    graph_verts = None
    graph_edges = None
    wire_graph = None
    
    outs_container = {}
    
    viz_lines = []
    def viz_line(a, b):
        viz_lines.append((a, b))
    def viz_dir(a, b, s=1.0):
        b = (a[0]+b[0]*s, a[1]+b[1]*s, a[2]+b[2]*s)
        viz_lines.append((a, b))
    def get_vert(t, c):
        return vertices[triangles[t][c]]
    
    viz_colors = None
    def viz_color(t, c, color):
        nonlocal viz_colors
        if viz_colors is None:
            viz_colors = [[(1,1,1)] * 3 for t1 in triangles]
        viz_colors[t][c] = color
    
    # Note: it probably makes more sense to gather all boundaries after the surface
    # segments are determined, since we need to track their intersections too
    
    # Note: when indexing corners, here I rely on python's negative index behavior
    # to avoid an extra modulo 3 operation
    
    # Note: there may be more than one shared boundary between clusters
    # (e.g. when a cone/cylinder is subtracted from a cone/cylinder/sphere)
    
    # Returns corners, because normals and other info may be split on boundaries
    def vertex_skirt(t, c):
        t_adj = opposites[t]
        
        inner = [(t, c)]
        outer = [(t, (c-1) % 3), (t, (c-2) % 3)]
        
        tR = -1
        adj = t_adj[c-1]
        while adj:
            tR, cR = adj
            if tR == t: break
            inner.append((tR, (cR-1) % 3))
            outer.append(adj)
            adj = opposites[tR][cR-2]
        
        if tR != t:
            adj = t_adj[c-2]
            while adj:
                tL, cL = adj
                inner.insert(0, (tL, (cL-2) % 3))
                outer.insert(0, adj)
                adj = opposites[tL][cL-1]
        
        return inner, outer
    
    def clear():
        nonlocal vertices, triangles, triangle_edges, opposites, edge_midpoints
        nonlocal boundaries, tri_infos, corner_infos
        nonlocal clusters, graph_verts, graph_edges, wire_graph
        nonlocal viz_lines, viz_colors
        vertices = None
        triangles = None
        triangle_edges = None
        opposites = None
        edge_midpoints = None
        boundaries = None
        tri_infos = None
        corner_infos = None
        clusters = None
        graph_verts = None
        graph_edges = None
        wire_graph = None
        
        viz_lines.clear()
        viz_colors = None
    
    def begin(text, count=1):
        nonlocal progress, progress_max, progress_text, progress_info
        progress = 0
        progress_max = count
        progress_text = text
        progress_info = ""
    
    def get_task_name():
        return progress_text
    
    def get_progress_info():
        return progress_info
    
    def set_progress_info(info):
        nonlocal progress_info
        progress_info = info
    
    def advance():
        nonlocal progress
        # This appears to be slightly faster than the += operator
        progress = progress + 1
    
    def get_progress():
        return progress
    
    def set_progress(value):
        nonlocal progress
        progress = value
    
    def get_progress_relative():
        return (progress / progress_max if progress_max > 0 else 1.0)
    
    def set_progress_relative(value):
        nonlocal progress
        progress = value * progress_max
    
    def set_vertices(verts):
        """
        verts: collection of 3D positions for each vertex
        """
        
        nonlocal vertices
        vertices = verts
    
    def set_edge_midpoints(midpoints):
        """
        midpoints: collection of 3D midpoint positions for each edge
        """
        
        nonlocal edge_midpoints
        edge_midpoints = midpoints
    
    def set_topology(tris, adjacent_corners, tris_edges):
        """
        tris: collection of triangles (vertex index tuples); it's expected
            that all indices are valid and there are no duplicate triangles
        adjacent_corners: collection of adjacent triangle corners. E.g., if
            the triangle ABC is adjacent to triangles CBD and EAB, the entry
            for ABC in the adjacent_corners would look like this:
            ((index of CBD, 2), None, (index of EAB, 0))
            2 is the index of corner D in the triangle CBD
            0 is the index of corner E in the triangle EAB
            None indicates that the corresponding corner has no adjacent
            counterpart (due to a non-manifold edge)
        tris_edges: collection of edge index tuples for each triangle
        """
        
        nonlocal triangles, triangle_edges, opposites
        triangles = tris
        triangle_edges = tris_edges
        opposites = adjacent_corners
    
    def build_topology(tris, seams=None):
        """
        tris: collection of triangles (vertex index tuples); it's expected
            that all indices are valid and there are no duplicate triangles
        seams: an optional collection of edge keys (start vertex, end vertex)
            for each edge that must always be treated as a topological boundary
        """
        
        nonlocal triangles, triangle_edges, opposites, edge_midpoints, progress
        triangles = tris
        opposites = [None] * len(triangles)
        triangle_edges = [None] * len(triangles)
        edge_midpoints = []
        
        edges = {}
        edge_indices = {}
        
        edges_get = edges.get
        edge_indices_get = edge_indices.get
        
        if seams:
            begin("Topology Init", len(seams))
            for edge in seams:
                edges[(edge[0], edge[1])] = None
                edges[(edge[1], edge[0])] = None
                progress = progress + 1
        
        def add_edge(t, tri, c):
            viA = tri[c-1]
            viB = tri[c-2]
            edge = (viA, viB)
            edges[edge] = (None if edge in edges else (t, c))
            
            if viA > viB: edge = (viB, viA)
            index = edge_indices_get(edge)
            if index is None:
                index = len(edge_indices)
                edge_indices[edge] = index
                ax, ay, az = vertices[viA]
                bx, by, bz = vertices[viB]
                edge_midpoints.append(((ax+bx)*0.5, (ay+by)*0.5, (az+bz)*0.5))
            return index
        
        def get_opposite(t, tri, c):
            edge = (tri[c-1], tri[c-2])
            cd = edges_get(edge, None) # direct
            cr = edges_get((edge[1], edge[0]), None) # reverse
            return (cr if cd and cr else None)
        
        begin("Topology Edges", len(triangles))
        for t, tri in enumerate(triangles):
            ei0 = add_edge(t, tri, 0)
            ei1 = add_edge(t, tri, 1)
            ei2 = add_edge(t, tri, 2)
            triangle_edges[t] = (ei0, ei1, ei2)
            progress = progress + 1
        
        begin("Topology Links", len(triangles))
        for t, tri in enumerate(triangles):
            opposites[t] = (
                get_opposite(t, tri, 0),
                get_opposite(t, tri, 1),
                get_opposite(t, tri, 2),
            )
            progress = progress + 1
    
    def set_triangle_infos(infos):
        """
        infos: collection of (area, normal) records for each triangle
        """
        
        nonlocal tri_infos
        tri_infos = infos
    
    def calc_triangle_infos():
        nonlocal tri_infos, progress
        tri_infos = [None] * len(triangles)
        
        begin("Triangle Normals", len(triangles))
        for t, tri in enumerate(triangles):
            vi0, vi1, vi2 = tri
            
            x0, y0, z0 = vertices[vi0]
            x1, y1, z1 = vertices[vi1]
            x2, y2, z2 = vertices[vi2]
            
            ax = x1 - x0; ay = y1 - y0; az = z1 - z0
            bx = x2 - x0; by = y2 - y0; bz = z2 - z0
            
            nx = ay*bz - az*by
            ny = az*bx - ax*bz
            nz = ax*by - ay*bx
            
            magnitude = sqrt(nx*nx + ny*ny + nz*nz) or nonzero
            normal = (nx/magnitude, ny/magnitude, nz/magnitude)
            area = magnitude * 0.5
            
            tri_infos[t] = (area, normal)
            
            progress = progress + 1
    
    def calc_vertex_infos():
        nonlocal corner_infos, progress
        corner_infos = [None] * len(triangles)
        
        # Curvature estimation here is based on (DOI: 10.1631/jzus.2005.AS0128)
        # "Curvatures estimation on triangular mesh" (2005)
        # Though here I weigh triangle normals by the area to avoid disbalance
        
        curvature_queue = []
        
        def calc_vertex_normal(t, c):
            tri_corner_info = corner_infos[t]
            if tri_corner_info and tri_corner_info[c]: return # skip if already processed
            
            # Normal, kmin, kmax, dmin, dmax
            # kG = kmin * kmax, kH = (kmin + kmax)/2
            vert_info = [None, None, None, None, None]
            
            # Consider only 1-ring around the vertex for now
            inner, outer = vertex_skirt(t, c)
            
            nx = 0.0
            ny = 0.0
            nz = 0.0
            
            for t1, c1 in inner:
                # Assign vert_info reference to all shared corners
                tri_corner_info = corner_infos[t1]
                if not tri_corner_info:
                    tri_corner_info = [None, None, None]
                    corner_infos[t1] = tri_corner_info
                tri_corner_info[c1] = vert_info
                
                area, normal = tri_infos[t1]
                tnx, tny, tnz = normal
                nx = nx + tnx * area
                ny = ny + tny * area
                nz = nz + tnz * area
            
            magnitude = sqrt(nx*nx + ny*ny + nz*nz) or nonzero
            vert_info[0] = (nx/magnitude, ny/magnitude, nz/magnitude)
            
            #viz_dir(get_vert(t, c), vert_info[0], 0.1) ###########################
            
            curvature_queue.append((t, c, vert_info, outer, inner))
        
        def calc_vertex_curvature(t, c, vert_info, outer):
            nx, ny, nz = vert_info[0]
            
            vi = triangles[t][c]
            vx, vy, vz = vertices[vi]
            
            tangents = []
            
            t0, c0 = t, c
            
            k_a = nan
            e1x, e1y, e1z = None, None, None
            
            for t, c in outer:
                vi = triangles[t][c]
                v1x, v1y, v1z = vertices[vi]
                
                vert1_info = corner_infos[t][c]
                n1x, n1y, n1z = vert1_info[0]
                
                dx = v1x - vx; dy = v1y - vy; dz = v1z - vz # delta
                d_mag2 = dx*dx + dy*dy + dz*dz
                if not d_mag2: continue # degenerate edge
                
                dn = dx*nx + dy*ny + dz*nz # projection on normal
                tx = dx - dn*nx; ty = dy - dn*ny; tz = dz - dn*nz # tangent
                magnitude = sqrt(tx*tx + ty*ty + tz*tz) or nonzero
                tx, ty, tz = (tx/magnitude, ty/magnitude, tz/magnitude)
                
                #viz_dir(get_vert(t0, c0), (tx, ty, tz), 0.05) ###########################
                
                dnx = n1x - nx; dny = n1y - ny; dnz = n1z - nz
                k = -(dx*dnx + dy*dny + dz*dnz) / d_mag2
                tangents.append((tx, ty, tz, k))
                
                if abs(k) <= abs(k_a): continue
                #if k <= k_a: continue
                k_a = k
                e1x, e1y, e1z = tx, ty, tz
            
            if e1x is None:
                vert_info[1] = 0.0
                vert_info[2] = 0.0
                vert_info[3] = (0.0, 0.0, 0.0)
                vert_info[4] = (0.0, 0.0, 0.0)
                return
            
            e2x = e1y*nz - e1z*ny
            e2y = e1z*nx - e1x*nz
            e2z = e1x*ny - e1y*nx
            
            a11 = 0.0
            a12 = 0.0
            a22 = 0.0
            a13 = 0.0
            a23 = 0.0
            
            for tx, ty, tz, k in tangents:
                cos_theta = tx*e1x + ty*e1y + tz*e1z
                if cos_theta > 1.0:
                    cos_theta = 1.0
                elif cos_theta < -1.0:
                    cos_theta = -1.0
                sin_theta = sqrt(1 - cos_theta*cos_theta)
                cos2_theta = cos_theta * cos_theta
                sin2_theta = sin_theta * sin_theta
                cossin_theta = cos_theta * sin_theta
                k_a_cos2 = k - k_a * cos2_theta
                a11 = a11 + cos2_theta * sin2_theta
                a12 = a12 + cossin_theta * sin2_theta
                a22 = a22 + sin2_theta * sin2_theta
                a13 = a13 + k_a_cos2 * cossin_theta
                a23 = a23 + k_a_cos2 * sin2_theta
            
            denominator = a11*a22 - a12*a12
            
            if abs(denominator) <= 1e-3:
                i = int(k_a > 0.0)
                j = 1 - i
                vert_info[1+i] = k_a
                vert_info[1+j] = 0.0
                vert_info[3+i] = (e1x, e1y, e1z)
                vert_info[3+j] = (e2x, e2y, e2z)
                #viz_dir(get_vert(t0, c0), vert_info[3], 0.05) ###########################
                #viz_dir(get_vert(t0, c0), vert_info[4], 0.05) ###########################
                return
            
            k_b = (a13*a22 - a23*a12) / denominator
            k_c = (a11*a23 - a12*a13) / denominator
            
            kG = k_a*k_c - k_b*k_b/4.0
            kH = (k_a + k_c)/2.0
            hk_sqrt = sqrt(kH*kH - kG)
            k_max = kH + hk_sqrt
            k_min = kH - hk_sqrt
            
            k_delta = k_min - k_max
            if abs(k_delta) > epsilon:
                theta = 0.5 * asin(k_b / k_delta)
                cos_theta = cos(theta)
                sin_theta = sin(theta)
                d_max = (
                    cos_theta * e1x + sin_theta * e2x,
                    cos_theta * e1y + sin_theta * e2y,
                    cos_theta * e1z + sin_theta * e2z,
                )
                d_min = (
                    cos_theta * e2x - sin_theta * e1x,
                    cos_theta * e2y - sin_theta * e1y,
                    cos_theta * e2z - sin_theta * e1z,
                )
            else:
                d_max = (e1x, e1y, e1z)
                d_min = (e2x, e2y, e2z)
            
            #viz_dir(get_vert(t0, c0), d_max, 0.05) ###########################
            #viz_dir(get_vert(t0, c0), d_min, 0.05) ###########################
            
            vert_info[1] = k_min
            vert_info[2] = k_max
            vert_info[3] = d_min
            vert_info[4] = d_max
        
        begin("Vertex Normals", len(triangles))
        for t in range(len(triangles)):
            calc_vertex_normal(t, 0)
            calc_vertex_normal(t, 1)
            calc_vertex_normal(t, 2)
            progress = progress + 1
        
        #for item in curvature_queue:
        #    print(item)
        
        begin("Vertex Curvature", len(curvature_queue))
        for t, c, vert_info, outer, inner in curvature_queue:
            calc_vertex_curvature(t, c, vert_info, outer)
            
            #print(vert_info)
            normal, kmin, kmax, dmin, dmax = vert_info
            #print([kmin, kmax])
            k_abs_min = min(abs(kmin), abs(kmax))
            k_abs_max = max(abs(kmin), abs(kmax))
            k_delta = max(kmin, kmax) - min(kmin, kmax)
            # 0: plane/sphere, 0.5: cylinder, 1: saddle
            k_ratio = k_delta / (2 * (k_abs_max or nonzero))
            k_hyper = 1 - 1/(1 + k_abs_max)
            r = k_hyper
            g = k_ratio
            b = 0 #1 - k_hyper*k_ratio
            
            #print([k_abs_min, k_abs_max, k_delta, k_hyper, k_ratio])
            
            k_limit = 0.01
            k_limit2 = 0.1
            is_curved = (k_abs_max > k_limit)
            is_cylindrical = (abs(k_ratio - 0.5) < 0.1) and (k_delta > k_limit2)
            is_saddle = (k_ratio - 0.5) > 0.1
            
            r = is_curved
            g = is_cylindrical and is_curved
            b = is_saddle and is_curved
            
            color = (r, g, b)
            
            for t, c in inner:
                viz_color(t, c, color)
            
            progress = progress + 1
        
        '''
        for tri_colors in viz_colors:
            r0, g0, b0 = tri_colors[0]
            r1, g1, b1 = tri_colors[1]
            r2, g2, b2 = tri_colors[2]
            r = ((r0+r1+r2)/3) > 0.5
            g = ((g0+g1+g2)/3) > 0.5
            b = ((b0+b1+b2)/3) > 0.5
            color = (r, g, b)
            tri_colors[0] = color
            tri_colors[1] = color
            tri_colors[2] = color
        '''
    
    class Cluster:
        def __init__(self, surf):
            self.surf = surf
            self.links = {}
    
    class ClusterLink:
        def __init__(self, cluster0, cluster1, normal_stats, primitive, cost, scale=1.0):
            self.cluster0 = cluster0
            self.cluster1 = cluster1
            self.normal_stats = normal_stats
            self.primitive = primitive
            self.cost = cost
            self.scale = scale
    
    # Hierarchical clustering based on (DOI: 10.1007/s41095-020-0192-6)
    # "Simple primitive recognition via hierarchical face clustering" (2020)
    
    def decompose(max_error=0.01, link_penalty=1.0, face_error_weight=1.0):
        nonlocal clusters, progress
        
        face_error_weight = min(max(face_error_weight, 0.0), 1.0)
        
        clusters = [None] * len(triangles)
        
        outs_container["clusters"] = clusters # for test
        
        links = PriorityHeap()
        
        vs_tid = triangles
        vs_pos = vertices
        vs_tag = [0] * len(vs_pos)
        
        es_tid = triangle_edges
        es_pos = edge_midpoints
        es_tag = [0] * len(es_pos)
        
        v_data = vs_tid, vs_pos, vs_tag
        e_data = es_tid, es_pos, es_tag
        
        max_points = len(vs_pos) + len(es_pos)
        points = np.zeros((max_points, 3))
        bufs = np.ones((max_points, 6))
        
        tag = 0
        
        def cost_scale(cluster0, cluster1):
            if link_penalty <= 0: return 1.0
            # Idea adapted from (DOI: 10.1007/s00371-006-0375-x)
            # "Hierarchical mesh segmentation based on fitting primitives" (2006)
            # The premise is to penalize the creation of unbalanced trees
            links = set()
            links.update(cluster0.links.keys())
            links.update(cluster1.links.keys())
            links.discard(cluster0)
            links.discard(cluster1)
            return 1.0 + link_penalty * len(links)
        
        def fit_primitive(cluster0, cluster1):
            nonlocal tag
            tag += 1
            curv = corner_infos
            surf0, surf1, face_err_w = cluster0.surf, cluster1.surf, face_error_weight
            return SurfaceInfo.merge_fit(surf0, surf1, v_data, e_data, tag, points, bufs, curv, face_err_w)
        
        def link_clusters(cluster0, cluster1):
            if cluster1 in cluster0.links: return
            normal_stats, primitive, cost = fit_primitive(cluster0, cluster1)
            link = ClusterLink(cluster0, cluster1, normal_stats, primitive, cost)
            cluster0.links[cluster1] = link
            cluster1.links[cluster0] = link
            links.add(link, cost)
        
        def merge(link):
            cluster0 = link.cluster0
            cluster1 = link.cluster1
            
            if len(cluster1.surf.tris) > len(cluster0.surf.tris):
                cluster0, cluster1 = cluster1, cluster0
            
            surf = cluster0.surf
            surf.tris.extend(cluster1.surf.tris)
            surf.normal_stats = link.normal_stats
            surf.primitive = link.primitive
            surf.fit_error = link.cost
            
            for t in cluster1.surf.tris:
                clusters[t] = cluster0
            
            cluster0.links.pop(cluster1)
            if link in links: links.remove(link)
            
            def update_link(link2, cluster2):
                normal_stats, primitive, cost = fit_primitive(cluster0, cluster2)
                scale = cost_scale(cluster0, cluster2)
                link2.normal_stats = normal_stats
                link2.primitive = primitive
                link2.cost = cost * scale
                link2.scale = scale
                links.add(link2, link2.cost) # update the priority
            
            for cluster2, link2 in cluster0.links.items():
                update_link(link2, cluster2)
            
            for cluster2, link2 in cluster1.links.items():
                if link2 is link: continue
                
                cluster2.links.pop(cluster1)
                
                if cluster2 in cluster0.links:
                    if link2 in links: links.remove(link2)
                    continue
                
                cluster0.links[cluster2] = link2
                cluster2.links[cluster0] = link2
                
                if link2.cluster1 is cluster1:
                    link2.cluster1 = cluster0
                else:
                    link2.cluster0 = cluster0
                
                update_link(link2, cluster2)
        
        begin("Clusters Setup", len(triangles))
        for t in range(len(triangles)):
            clusters[t] = Cluster(SurfaceInfo.FromTriangle(t, vs_tid, vs_pos))
            progress = progress + 1
        
        begin("Clusters Linking", len(triangles))
        for t, t_adj in enumerate(opposites):
            cluster0 = clusters[t]
            adj0, adj1, adj2 = t_adj
            if adj0: link_clusters(cluster0, clusters[adj0[0]])
            if adj1: link_clusters(cluster0, clusters[adj1[0]])
            if adj2: link_clusters(cluster0, clusters[adj2[0]])
            progress = progress + 1
        
        links_max = len(links)
        begin("Clustering", 1.0)
        while links:
            link = links.pop()
            cost = link.cost / link.scale
            set_progress_info(f"Fitting error: {cost:.6}")
            # print(cost, "/", max_error)
            if cost <= max_error: merge(link)
            progress = 1.0 - (len(links) / links_max)
    
    class GVertex:
        def __init__(self, pos):
            self.pos = np.copy(pos)
            self.edges = []
            self.tangent = None
            self.next = {}
        
        def add_edge(self, ge):
            if ge not in self.edges: self.edges.append(ge)
        
        def replace_edge(self, ge_old, ge_new):
            i = self.edges.index(ge_old)
            self.edges[i] = ge_new
            
            keys = []
            for key, ge in self.next.items():
                if ge != ge_old: continue
                self.next[key] = ge_new
                keys.append(key)
            
            return keys
        
        def get_surfaces(self):
            return {surf for ge in self.edges for surf in ge.get_surfaces()}
    
    class GEdge:
        def __init__(self, gv0, gv1, clusters):
            self.gv0 = gv0
            self.gv1 = gv1
            self.clusters = clusters
            self.points = []
            self.clusters0 = clusters
            self.clusters1 = clusters
        
        def opposite(self, gv):
            if gv == self.gv0: return self.gv1
            if gv == self.gv1: return self.gv0
            return None
        
        def add_cluster(self, cluster, t, c, aligned):
            if not cluster: return
            # Duplicates can happen
            entry = (cluster, t, c, aligned)
            if entry not in self.clusters: self.clusters.append(entry)
        
        def get_surfaces(self):
            return {cluster.surf for cluster, t, c, aligned in self.clusters}
    
    def extract_boundaries(**options):
        nonlocal graph_verts, graph_edges, wire_graph, progress
        
        verts_map = {}
        edges_map = {}
        graph_verts = set()
        graph_edges = set()
        
        outs_container["graph_verts"] = graph_verts
        outs_container["graph_edges"] = graph_edges
        
        def add_vert(vi):
            gv = verts_map.get(vi)
            if not gv:
                gv = GVertex(vertices[vi])
                verts_map[vi] = gv
            return gv
        
        def add_edge(t, c, vi0, vi1, adj):
            cluster0 = clusters[t]
            cluster1 = (clusters[adj[0]] if adj else None)
            if cluster0 == cluster1: return
            
            gv0 = add_vert(vi0)
            gv1 = add_vert(vi1)
            
            ge = edges_map.get((vi0, vi1))
            if not ge:
                ge = GEdge(gv0, gv1, [])
                gv0.add_edge(ge)
                gv1.add_edge(ge)
                edges_map[(vi0, vi1)] = ge
                edges_map[(vi1, vi0)] = ge
            
            aligned0 = (ge.gv0 == gv0)
            aligned1 = not aligned0
            
            ge.add_cluster(cluster0, t, c, aligned0)
            gv0.next[(cluster0, t)] = ge
            
            if adj:
                _t, _c = adj
                ge.add_cluster(cluster1, _t, _c, aligned1)
                gv1.next[(cluster1, _t)] = ge
        
        def merge_points(points0, points1, aligned0, aligned1):
            if not points0: return points1
            if not points1: return points0
            if not aligned0: points0.reverse()
            if not aligned1: points1.reverse()
            points0.extend(points1)
            return points0
        
        def flip_alignment(ge_clusters):
            return [(cluster, t, c, not aligned) for cluster, t, c, aligned in ge_clusters]
        
        def dissolve_vert(gv):
            if len(gv.edges) != 2: return
            
            ge0, ge1 = gv.edges
            
            if ge0 == ge1: # standalone boundary loop
                ge0.gv0 = None
                ge0.gv1 = None
                ge0.points.append(gv.pos)
                graph_verts.discard(gv)
                return
            
            gv0 = ge0.opposite(gv)
            gv1 = ge1.opposite(gv)
            
            if gv == ge0.gv1:
                aligned1 = (gv == ge1.gv0)
                ge0.gv1 = gv1
                ge0.points.append(gv.pos)
                ge0.points = merge_points(ge0.points, ge1.points, True, aligned1)
                ge0.clusters1 = (ge1.clusters1 if aligned1 else flip_alignment(ge1.clusters0))
            else:
                aligned1 = (gv == ge1.gv1)
                ge0.gv0 = gv1
                ge0.points.insert(0, gv.pos)
                ge0.points = merge_points(ge1.points, ge0.points, aligned1, True)
                ge0.clusters0 = (ge1.clusters0 if aligned1 else flip_alignment(ge1.clusters1))
            
            gv1.replace_edge(ge1, ge0)
            
            graph_edges.discard(ge1)
            graph_verts.discard(gv)
        
        def get_corner_info(ge, cluster, t):
            if t is None:
                for _cluster, _t, c, aligned in ge.clusters1:
                    if _cluster != cluster: continue
                    if aligned: return _t, c, aligned, ge.gv1
                
                for _cluster, _t, c, aligned in ge.clusters0:
                    if _cluster != cluster: continue
                    if not aligned: return _t, c, aligned, ge.gv0
                
                return
            
            at_start = None
            for _cluster, _t, c, aligned in ge.clusters0:
                if (_cluster != cluster) or (_t != t): continue
                at_start = aligned
                break
            
            at_end = None
            for _cluster, _t, c, aligned in ge.clusters1:
                if (_cluster != cluster) or (_t != t): continue
                at_end = aligned
                break
            
            if (at_end is True) or (at_start is True):
                for _cluster, _t, c, aligned in ge.clusters1:
                    if (_cluster != cluster) or not aligned: continue
                    return _t, c, aligned, ge.gv1
            elif (at_start is False) or (at_end is False):
                for _cluster, _t, c, aligned in ge.clusters0:
                    if (_cluster != cluster) or aligned: continue
                    return _t, c, aligned, ge.gv0
        
        def find_next_edge(gv, cluster, t, c):
            while True:
                ge = gv.next.get((cluster, t))
                if ge: return ge, t, c
                t, c = opposites[t][c-2]
        
        def walk_edge_loop(cluster, ge_start, processed_edges):
            ge = ge_start
            t = None
            loop = []
            is_manifold = True
            
            while True:
                t, c, aligned, tail = get_corner_info(ge, cluster, t)
                loop.append((ge, aligned))
                processed_edges.add(ge)
                
                if len(ge.clusters) != 2: is_manifold = False
                
                if tail is None: break
                
                ge, t, c = find_next_edge(tail, cluster, t, c)
                if ge == ge_start: break
            
            return loop, is_manifold
        
        def extract_loops(cluster, cluster_edges):
            loops = []
            processed_edges = set()
            is_manifold = True
            
            for ge in cluster_edges:
                if ge in processed_edges: continue
                loop, _is_manifold = walk_edge_loop(cluster, ge, processed_edges)
                loops.append(loop)
                if not _is_manifold: is_manifold = False
            
            return loops, is_manifold
        
        begin("Boundary Edges", len(triangles))
        for t, t_adj in enumerate(opposites):
            vi0, vi1, vi2 = triangles[t]
            adj0, adj1, adj2 = t_adj
            add_edge(t, 0, vi1, vi2, adj0)
            add_edge(t, 1, vi2, vi0, adj1)
            add_edge(t, 2, vi0, vi1, adj2)
            progress = progress + 1
        
        graph_verts.update(verts_map.values())
        graph_edges.update(edges_map.values())
        
        for gv in verts_map.values():
            dissolve_vert(gv)
        
        # Note: if cluster has no edges, it's a standalone closed primitive (sphere or cyclide)
        # Initialize with all clusters, so that even standalone primitives are present
        clusters_edges = {cluster: set() for cluster in clusters}
        for ge in graph_edges:
            for cluster, t, c, aligned in ge.clusters:
                clusters_edges[cluster].add(ge)
        
        wire_graph = WireGraph(begin=begin, advance=advance)
        
        verts_map = {None: None}
        for gv in graph_verts:
            verts_map[gv] = wire_graph.add_vert(gv.pos)
        
        edges_map = {}
        for ge in graph_edges:
            edges_map[ge] = wire_graph.add_edge(verts_map[ge.gv0], verts_map[ge.gv1], ge.points)
        
        for cluster, cluster_edges in clusters_edges.items():
            loops, is_manifold = extract_loops(cluster, cluster_edges)
            wire_loops = ([(edges_map[edge], aligned) for edge, aligned in loop] for loop in loops)
            wire_graph.add_face(cluster.surf, wire_loops, is_manifold)
        
        wire_graph.calculate_tangents(options.get("hard_angle", 0.0))
        wire_graph.fit(**options)
    
    class Shell:
        def __init__(self):
            self.faces = []
            self.is_closed = True
    
    class Face:
        def __init__(self):
            self.bounds = []
            self.surface = None
            self.outward = True
    
    def make_brep():
        nonlocal progress
        
        points = []
        verts = []
        edges = []
        shells = []
        processed_faces = set()
        
        points_map = {}
        verts_map = {}
        edges_map = {}
        
        def map_point(p):
            p = tuple(p)
            i = points_map.get(p)
            if i is None:
                i = len(points)
                points_map[p] = i
                points.append(p)
            return i
        
        # Note: we can't use wire_graph.verts, since for
        # standalone loops there are no vertices
        def map_vert(i):
            j = verts_map.get(i)
            if j is None:
                j = len(verts)
                verts_map[i] = j
                verts.append(i)
            return j
        
        def get_control_points(wire_edge):
            p0 = wire_edge.points[0]
            yield p0
            for seg_i in range(len(wire_edge.tangents)):
                p0 = wire_edge.points[seg_i]
                p3 = wire_edge.points[seg_i+1]
                t1, t2 = wire_edge.tangents[seg_i]
                p1, p2 = p0+t1, p3+t2
                yield p1
                yield p2
                yield p3
        
        for wire_edge in wire_graph.edges:
            control_points = [map_point(p) for p in get_control_points(wire_edge)]
            v0i = map_vert(control_points[0])
            v1i = map_vert(control_points[-1])
            edges_map[wire_edge] = len(edges)
            edges.append((v0i, v1i, control_points))
        
        def is_outward_oriented(surf):
            normal_agreement = 0.0
            for t in surf.tris:
                area, normal = tri_infos[t]
                tri = triangles[t]
                vi0, vi1, vi2 = tri
                x0, y0, z0 = vertices[vi0]
                x1, y1, z1 = vertices[vi1]
                x2, y2, z2 = vertices[vi2]
                p = ((x0+x1+x2)/3.0, (y0+y1+y2)/3.0, (z0+z1+z2)/3.0)
                normal_agreement += np.dot(normal, surf.normal(p)) * area
            return bool(normal_agreement >= 0.0) # Convert numpy.bool_ to Python bool
        
        def make_face(wire_face):
            face = Face()
            
            for wire_loop in wire_face.loops:
                loop = [(edges_map[wire_edge], aligned) for wire_edge, aligned in wire_loop.edges]
                face.bounds.append(loop)
            
            face.outward = is_outward_oriented(wire_face.surf)
            face.surface = wire_face.surf.to_brep()
            
            return face
        
        def add_connected_faces(wire_face):
            shell = Shell()
            shells.append(shell)
            
            queue = []
            
            def enqueue(wire_face):
                if wire_face in processed_faces: return
                # Must be added immediately, to not process faces multiple times
                processed_faces.add(wire_face)
                queue.append(wire_face)
            
            enqueue(wire_face)
            
            while queue:
                wire_face = queue.pop()
                advance()
                
                shell.faces.append(make_face(wire_face))
                if not wire_face.is_manifold: shell.is_closed = False
                
                for wire_loop in wire_face.loops:
                    for wire_edge, aligned in wire_loop.edges:
                        for wire_loop2 in wire_edge.loops:
                            enqueue(wire_loop2.face)
        
        begin("Processing Parts", len(wire_graph.faces))
        for wire_face in wire_graph.faces:
            if wire_face in processed_faces: continue
            add_connected_faces(wire_face)
        
        return dict(points=points, verts=verts, edges=edges, shells=shells)
    
    decomposer_type = type("MeshDecomposer", (), {
        "task_name": property(lambda self: get_task_name()),
        "progress": property((lambda self: get_progress()),
                             (lambda self, value: set_progress(value))),
        "progress_relative": property((lambda self: get_progress_relative()),
                                      (lambda self, value: set_progress_relative(value))),
        "progress_info": property((lambda self: get_progress_info()),
                                  (lambda self, value: set_progress_info(value))),
        "clear": staticmethod(clear),
        "set_vertices": staticmethod(set_vertices),
        "set_edge_midpoints": staticmethod(set_edge_midpoints),
        "set_topology": staticmethod(set_topology),
        "build_topology": staticmethod(build_topology),
        "set_triangle_infos": staticmethod(set_triangle_infos),
        "calc_triangle_infos": staticmethod(calc_triangle_infos),
        "calc_vertex_infos": staticmethod(calc_vertex_infos),
        "decompose": staticmethod(decompose),
        "extract_boundaries": staticmethod(extract_boundaries),
        "make_brep": staticmethod(make_brep),
        
        "viz_lines": property(lambda self: viz_lines),
        "viz_colors": property(lambda self: viz_colors),
        "clusters": property(lambda self: outs_container.get("clusters")),
        "graph_verts": property(lambda self: outs_container.get("graph_verts")),
        "graph_edges": property(lambda self: outs_container.get("graph_edges")),
    })
    
    decomposer = decomposer_type()
    # Assign these directly to instance, to avoid staticmethod indirection
    decomposer.begin = begin
    decomposer.advance = advance
    
    return decomposer
