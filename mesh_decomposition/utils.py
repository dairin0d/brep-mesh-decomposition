# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 dairin0d https://github.com/dairin0d

import sys
import math

import numpy as np

# math.ulp(0) returns the minimal denormalized number, but on platforms
# (or if compiled) without their support, it will be treated as 0
nonzero = math.ulp(0) or sys.float_info.min
epsilon = sys.float_info.epsilon
isfinite = math.isfinite
isnan = math.isnan
inf = math.inf
nan = math.nan
pi = math.pi
sqrt = math.sqrt
copysign = math.copysign
acos = math.acos
asin = math.asin
atan = math.atan
cos = math.cos
sin = math.sin
tan = math.tan
norm = np.linalg.norm

def orthogonal_3d(x, y, z):
    # https://math.stackexchange.com/questions/137362/
    return copysign(z, x), copysign(z, y), -(copysign(x, z) + copysign(y, z))

def dot_product(ax, ay, az, bx, by, bz):
    return ax*bx + ay*by + az*bz

def cross_product(ax, ay, az, bx, by, bz):
    return (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx)

def normalize(x, y, z):
    mag = sqrt(x*x + y*y + z*z)
    return ((x/mag, y/mag, z/mag) if mag > 0.0 else (0.0, 0.0, 0.0))

def make_axis_matrix(z_axis):
    # z axis is expected to be normalized
    zx, zy, zz = z_axis
    xx, xy, xz = orthogonal_3d(zx, zy, zz)
    x_mag = sqrt(xx*xx + xy*xy + xz*xz)
    x_axis = xx/x_mag, xy/x_mag, xz/x_mag
    xx, xy, xz = x_axis
    y_axis = zy*xz - zz*xy, zz*xx - zx*xz, zx*xy - zy*xx
    # rotation matrices are orthogonal, so we can use transpose instead of inverse
    matrix_inv = np.array((x_axis, y_axis, z_axis))
    matrix = matrix_inv.T
    return matrix, matrix_inv

def divide():
    from math import copysign
    inf = math.inf
    nan = math.nan
    isnan = math.isnan
    def divide(x, y):
        # Make sure these are python floats, because e.g. numpy floats
        # dont raise the ZeroDivisionError, and just print a warning
        x = float(x)
        y = float(y)
        try:
            return x / y
        except ZeroDivisionError:
            if isnan(x) or isnan(y): return nan
            return copysign(inf, x) * copysign(1.0, y)
    return divide
divide = divide()

def linear_roots(a, b):
    if a == 0: return []
    B = -b/a
    return ([B] if isfinite(B) else [])

def quadratic_roots(a, b, c):
    if a == 0: return linear_roots(b, c)
    B = b/(2*a)
    C = c/a
    if not (isfinite(B) and isfinite(C)): return linear_roots(b, c)
    
    D = B*B - C
    
    if D < 0: return []
    if D == 0: return [-B, -B]
    sqrt_D = sqrt(D)
    return [-B - sqrt_D, -B + sqrt_D]

# Adapted from https://www.particleincell.com/2013/cubic-line-intersection/
def cubic_roots(a, b, c, d, imaginary_threshold=1e-16):
    if a == 0: return quadratic_roots(b, c, d)
    A = b/a
    B = c/a
    C = d/a
    if not (isfinite(A) and isfinite(B) and isfinite(C)): return quadratic_roots(b, c, d)
    
    Q = (3*B - A**2)/9
    R = (9*A*B - 27*C - 2*A**3)/54
    D = Q**3 + R**2 # polynomial discriminant
    
    t = []
    
    if D >= 0: # complex or duplicate roots
        sqrt_D = sqrt(D)
        S = copysign(1.0, R + sqrt_D) * abs(R + sqrt_D)**(1/3)
        T = copysign(1.0, R - sqrt_D) * abs(R - sqrt_D)**(1/3)
        
        t.append(-A/3 + (S + T)) # real root
        # imaginary part of complex root: abs(sqrt(3)*(S - T)/2)
        if D <= imaginary_threshold: # if imaginary part is ~= 0
            t.append(-A/3 - (S + T)/2) # real part of complex root
            t.append(t[-1]) # real part of complex root (same value)
    else: # distinct real roots
        th = acos(R/sqrt(-(Q**3)))
        
        sqrt_Q = sqrt(-Q)
        t.append(2 * sqrt_Q * cos(th/3) - A/3)
        t.append(2 * sqrt_Q * cos((th + 2*pi)/3) - A/3)
        t.append(2 * sqrt_Q * cos((th + 4*pi)/3) - A/3)
    
    t.sort()
    
    return t

def bezier_roots(p0, p1, p2, p3, duplicates=False, threshold=1e-16):
    a = -p0 + 3*p1 - 3*p2 + p3
    b = 3*p0 - 6*p1 + 3*p2
    c = -3*p0 + 3*p1
    d = p0
    roots = cubic_roots(a, b, c, d)
    
    t_min = -threshold
    t_max = 1+threshold
    
    result = []
    for t in roots:
        if (t >= t_min) and (t <= t_max):
            t = min(max(t, 0.0), 1.0)
            if duplicates or (t not in result): result.append(t)
    return result
