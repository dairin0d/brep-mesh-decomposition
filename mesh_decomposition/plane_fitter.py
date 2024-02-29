# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 dairin0d https://github.com/dairin0d

import math

import numpy as np

from .utils import *

class PlaneFitter:
    @classmethod
    def fit(cls, points, weights, centered=False):
        if weights is None: return cls.fit_svd(points, centered)
        return cls.fit_lsq(points, weights, centered)
    
    @classmethod
    def fit_svd(cls, points, centered=False):
        if centered:
            center = np.zeros(3)
            centered = points
        else:
            center = np.mean(points, axis=0)
            centered = points - center
        
        try:
            U, S, Vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None, None
        
        normal = Vh[-1] # already normalized
        
        return center, normal
    
    @classmethod
    def fit_lsq(cls, points, weights, centered=False):
        if centered:
            center = np.zeros(3)
            centered = points
        else:
            try:
                center = np.average(points, weights=weights, axis=0)
            except ZeroDivisionError: # all weights are zero
                center = np.mean(points, axis=0)
            centered = points - center
        
        xx = np.dot(centered[:,0] * centered[:,0], weights)
        xy = np.dot(centered[:,0] * centered[:,1], weights)
        xz = np.dot(centered[:,0] * centered[:,2], weights)
        yy = np.dot(centered[:,1] * centered[:,1], weights)
        yz = np.dot(centered[:,1] * centered[:,2], weights)
        zz = np.dot(centered[:,2] * centered[:,2], weights)
        
        w_sum = np.sum(weights)
        nx, ny, nz = cls.covariance_to_normal(xx, xy, xz, yy, yz, zz, w_sum)
        n_mag2 = nx*nx + ny*ny + nz*nz
        if n_mag2 == 0: return None, None
        
        n_mag = sqrt(n_mag2)
        return center, np.array((nx/n_mag, ny/n_mag, nz/n_mag))
    
    @classmethod
    def covariance_to_normal(cls, xx, xy, xz, yy, yz, zz, w):
        # https://www.ilikebigbits.com/2015_03_04_plane_from_points.html
        # https://www.ilikebigbits.com/2017_09_25_plane_from_points_2.html
        
        xx /= w; xy /= w; xz /= w; yy /= w; yz /= w; zz /= w
        
        Xx, Xy, Xz = yy*zz - yz*yz, xz*yz - xy*zz, xy*yz - xz*yy # Xx is determinant
        Yx, Yy, Yz = xz*yz - xy*zz, xx*zz - xz*xz, xy*xz - yz*xx # Yy is determinant
        Zx, Zy, Zz = xy*yz - xz*yy, xy*xz - yz*xx, xx*yy - xy*xy # Zz is determinant
        
        weight = Xx*Xx
        nx = Xx * weight
        ny = Xy * weight
        nz = Xz * weight
        weight = Yy*Yy * copysign(1.0, (nx*Yx + ny*Yy + nz*Yz))
        nx += Yx * weight
        ny += Yy * weight
        nz += Yz * weight
        weight = Zz*Zz * copysign(1.0, (nx*Zx + ny*Zy + nz*Zz))
        nx += Zx * weight
        ny += Zy * weight
        nz += Zz * weight
        
        return nx, ny, nz
    
    @classmethod
    def calc_center(cls, points, weights=None):
        if weights is not None:
            w_sum = 0.0
            for (x, y, z), w in zip(points, weights):
                cx = cx+x*w
                cy = cy+y*w
                cz = cz+z*w
                w_sum = w_sum+w
            
            if (not w_sum) or (w_sum == math.nan):
                return cls.calc_center(points)
        else:
            for x, y, z in points:
                cx = cx+x
                cy = cy+y
                cz = cz+z
            w_sum = len(points)
        
        return cx/w_sum, cy/w_sum, cz/w_sum, w_sum
    
    @classmethod
    def fit_lsq_py(cls, points, weights, centered=False):
        if centered:
            cx, cy, cz = 0.0, 0.0, 0.0
            w_sum = (len(points) if weights is None else sum(weights))
        else:
            cx, cy, cz, w_sum = cls.calc_center(points, weights)
        
        # Adapted from https://www.ilikebigbits.com/2015_03_04_plane_from_points.html
        
        xx = 0.0
        xy = 0.0
        xz = 0.0
        yy = 0.0
        yz = 0.0
        zz = 0.0
        
        if weights:
            for (x, y, z), w in zip(points, weights):
                x = x-cx
                y = y-cy
                z = z-cz
                xx = xx + x*x*w
                xy = xy + x*y*w
                xz = xz + x*z*w
                yy = yy + y*y*w
                yz = yz + y*z*w
                zz = zz + z*z*w
        else:
            for x, y, z in points:
                x = x-cx
                y = y-cy
                z = z-cz
                xx = xx + x*x
                xy = xy + x*y
                xz = xz + x*z
                yy = yy + y*y
                yz = yz + y*z
                zz = zz + z*z
        
        nx, ny, nz = cls.covariance_to_normal(xx, xy, xz, yy, yz, zz, w_sum)
        n_mag2 = nx*nx + ny*ny + nz*nz
        if n_mag2 == 0: return None, None
        
        n_mag = sqrt(n_mag2)
        return np.array((cx, cy, cz)), np.array((nx/n_mag, ny/n_mag, nz/n_mag))
