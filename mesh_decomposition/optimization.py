# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 dairin0d https://github.com/dairin0d

import math

import numpy as np

# This was developed with Blender in mind, which has a built-in numpy but not scipy

# Based on (DOI: 10.1007/s10589-010-9329-3)
# "Implementing the Nelder-Mead simplex algorithm with adaptive parameters" (2012)
# Also, taking some inspiration from scipy and https://github.com/fchollet/nelder-mead
def nelder_mead(f, x0, step=None, mask=None, maxiter=None, plateau=(10, 1e-5), adaptive=False):
    """
    f: function f(x) to minimize
    x0: initial guess (N-dimensional vector)
    step: initial size of the simplex
    mask: an optional index array specifying which axes to optimize
    maxiter: max iteration count
    plateau: stagnation stopping criteria (iterations, threshold)
    adaptive: whether to adjust parameters for high-dimensional cases
    
    To constrain allowed solutions, f(x) can return infinity in forbidden regions.
    
    Note: it's generally preferable to specify the initial simplex size, since
    with the default one the algorithm might take many more steps to converge
    """
    
    x0 = np.asarray(x0)
    
    if (mask is not None) and (len(mask) > 0):
        mask = np.asarray(mask) # must be an array or a list, not a tuple
        x_full = x0.copy()
        x0 = x0[mask]
        f_orig = f
        
        def f(x):
            x_full[mask] = x
            return f_orig(x_full)
    else:
        mask = None
    
    n = len(x0)
    
    simplex = [(x0, f(x0))]
    for i in range(n):
        x = x0.copy()
        if step:
            x[i] += step
        else:
            x[i] = (1.00025 * x[i]) or 0.05
        simplex.append((x, f(x)))
    
    n_factor = (n if adaptive and (n > 2) else 2)
    reflection = 1.0
    expansion = 1.0 + 2.0/n_factor
    contraction = 0.75 - 0.5/n_factor
    shrink = 1.0 - 1.0/n_factor
    
    sort_key = (lambda item: item[1])
    
    centroid = np.zeros(n)
    centroid_indices = list(range(1, n))
    
    shrink_indices = list(range(1, n+1))
    
    if maxiter is None: maxiter = math.inf
    
    plateau_count, plateau_limit = plateau
    plateau_limit = max(plateau_limit, 0.0)
    
    plateau_last = math.inf
    plateau_iter = 0
    
    while maxiter > 0:
        maxiter -= 1
        
        simplex.sort(key=sort_key)
        
        x_best, f_best = simplex[0]
        
        if f_best < plateau_last:
            plateau_last = f_best - plateau_limit
            plateau_iter = 0
        else:
            plateau_iter += 1
            if plateau_iter >= plateau_count: break
        
        centroid[:] = x_best
        for i in centroid_indices:
            centroid += simplex[i][0]
        centroid /= n
        
        x_worst, f_worst = simplex[-1]
        
        delta = reflection * (centroid - x_worst)
        x_r = centroid + delta
        f_r = f(x_r)
        
        if f_r < f_best: # expansion
            x_e = centroid + expansion * delta
            f_e = f(x_e)
            simplex[-1] = ((x_e, f_e) if f_e < f_r else (x_r, f_r))
            continue
        
        if f_r < simplex[-2][1]: # reflection
            simplex[-1] = (x_r, f_r)
            continue
        
        if f_r < f_worst: # outside contraction
            x_oc = centroid + contraction * delta
            f_oc = f(x_oc)
            if f_oc <= f_r:
                simplex[-1] = (x_oc, f_oc)
                continue
        else: # inside contraction
            x_ic = centroid - contraction * delta
            f_ic = f(x_ic)
            if f_ic < f_worst:
                simplex[-1] = (x_ic, f_ic)
                continue
        
        # shrink
        for i in shrink_indices:
            x = x_best + shrink * (simplex[i][0] - x_best)
            simplex[i] = (x, f(x))
    
    simplex.sort(key=sort_key)
    
    if mask is None: return simplex[0]
    
    x_best, f_best = simplex[0]
    x_full[mask] = x_best
    return x_full, f_best
