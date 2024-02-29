# An adaptation of https://github.com/ynakajima/polyline2bezier
# into python (I didn't thoroughly test my port, so may contain bugs)

# original: fitCurves.c
# http://tog.acm.org/resources/GraphicsGems/gems/fitCurves.c

# THIS SOURCE CODE IS PUBLIC DOMAIN, and
# is freely available to the entire computer graphics community
# for study, use, and modification.  We do request that the
# comment at the top of each file, identifying the original
# author and its original publication in the book Graphics
# Gems, be retained in all programs that use these files.

# An Algorithm for Automatically Fitting Digitized Curves
# by Philip J. Schneider
# from "Graphics Gems", Academic Press, 1990

import math

import numpy as np

def BesierFitter():
    bezCurve = [None] * 4 # Control points of fitted Bezier curve
    
    Q1 = [None] * 3 # Q'
    Q2 = [None] * 2 # Q''
    
    Vtemp = [None] * 4
    
    magnitude = np.linalg.norm
    
    def normalized(v):
        mag = magnitude(v)
        return (v / mag if mag > 0.0 else np.zeros(len(v)))
    
    # "An Algorithm for Automatically Fitting Digitized Curves"
    # Philip J. Schneider ("Graphics Gems", Academic Press, 1990)
    # Original: http://tog.acm.org/resources/GraphicsGems/gems/fitCurves.c (Public domain)
    # JavaScript port: https://github.com/ynakajima/polyline2bezier (MIT)
    # See also: https://github.com/burningmime/curves (ZLib)
    def Fit(polyline, error, iterationError=2.0):
        nPts = len(polyline)
        
        # Unit tangent vectors at endpoints
        tHat1 = ComputeLeftTangent(polyline, 0)
        tHat2 = ComputeRightTangent(polyline, nPts - 1)
        
        u = np.zeros(nPts) # Parameter values for points
        
        bezierSegments = []
        
        iterationError *= error
        FitCubic(polyline, 0, nPts - 1, u, tHat1, tHat2, error, iterationError, bezierSegments)
        
        return bezierSegments
    
    def FitCubic(d, first, last, u, tHat1, tHat2, error, iterationError, bezierSegments):
        nPts = last - first + 1 # Number of points in subset
        
        if nPts < 2: return
        
        # Use heuristic if region only has two points in it
        if nPts == 2:
            dist = magnitude(d[last] - d[first]) / 3.0
            bezCurve[0] = d[first]
            bezCurve[3] = d[last]
            bezCurve[1] = bezCurve[0] + (tHat1 * dist)
            bezCurve[2] = bezCurve[3] + (tHat2 * dist)
            bezierSegments.append(np.copy(bezCurve))
            return
        
        # Parameterize points, and attempt to fit curve
        ChordLengthParameterize(d, first, last, u) # Parameter values for point
        GenerateBezier(d, first, last, u, bezCurve, tHat1, tHat2)
        
        maxError = 0.0 # Maximum fitting error
        splitPoint = 0 # Point to split point set at
        
        # Find max deviation of points to fitted curve
        splitPoint, maxError = ComputeMaxError(d, first, last, u, bezCurve)
        
        if maxError < error:
            bezierSegments.append(np.copy(bezCurve))
            return
        
        # If error not too large, try some reparameterization and iteration
        if maxError < iterationError:
            maxIterations = 4 # Max times to try iterating
            
            for iteration in range(maxIterations):
                Reparameterize(d, first, last, u, bezCurve) # Improved parameter values
                GenerateBezier(d, first, last, u, bezCurve, tHat1, tHat2)
                splitPoint, maxError = ComputeMaxError(d, first, last, u, bezCurve)
                
                if maxError < error:
                    bezierSegments.append(np.copy(bezCurve))
                    return
        
        # Fitting failed -- split at max error point and fit recursively
        tHatCenter = ComputeCenterTangent(d, splitPoint) # Unit tangent vector at splitPoint
        FitCubic(d, first, splitPoint, u, tHat1, tHatCenter, error, iterationError, bezierSegments)
        FitCubic(d, splitPoint, last, u, -tHatCenter, tHat2, error, iterationError, bezierSegments)
    
    def GenerateBezier(d, first, last, u, bezCurve, tHat1, tHat2):
        # C and X matrices
        C00 = 0.0
        C01 = 0.0
        C10 = 0.0
        C11 = 0.0
        X0 = 0.0
        X1 = 0.0
        
        # Since we care only about ratio of determinants,
        # normalize values to avoid infinities/NaNs
        scale = 1.0 / math.sqrt(last - first + 1) # compensate for summation
        _tHat1 = tHat1 * scale
        _tHat2 = tHat2 * scale
        
        for i in range(first, last+1):
            # Bezier multipliers
            u1 = u[i]
            u0 = 1.0 - u1
            u1u1 = u1 * u1
            u0u0 = u0 * u0
            b0 = u0 * u0u0
            b1 = 3.0 * u1 * u0u0
            b2 = 3.0 * u1u1 * u0
            b3 = u1 * u1u1
            
            # A is the right-hand side for the equation
            A0 = _tHat1 * b1
            A1 = _tHat2 * b2
            
            C00 += np.dot(A0, A0)
            C01 += np.dot(A0, A1)
            C10 = C01 # C matrix is symmetric
            C11 += np.dot(A1, A1)
            
            tmp = (d[i] - (d[first] * (b0 + b1) + d[last] * (b2 + b3))) * scale
            X0 += np.dot(A0, tmp)
            X1 += np.dot(A1, tmp)
        
        # compensate for large numbers
        scale = np.max(np.abs([C00, C01, C11, X0, X1]))
        
        if scale > 1.0:
            scale = 1.0 / scale
            C00 *= scale
            C01 *= scale
            C10 *= scale
            C11 *= scale
            X0 *= scale
            X1 *= scale
        
        # Compute the determinants of C and X
        det_C0_C1 = C00 * C11 - C10 * C01
        det_C0_X = C00 * X1 - C10 * X0
        det_X_C1 = X0 * C11 - X1 * C01
        
        # Finally, derive alpha values (left and right)
        alpha_l = (0.0 if det_C0_C1 == 0 else det_X_C1 / det_C0_C1)
        alpha_r = (0.0 if det_C0_C1 == 0 else det_C0_X / det_C0_C1)
        
        # If alpha negative, use the Wu/Barsky heuristic (see text)
        # (if alpha is 0, you get coincident control points that lead to
        # divide by zero in any subsequent newtonRaphsonRootFind() call.
        segLength = magnitude(d[last] - d[first])
        epsilon = 1e-6 * segLength
        
        if (alpha_l < epsilon) or (alpha_r < epsilon):
            # fall back on standard (probably inaccurate) formula,
            # and subdivide further if needed.
            dist = segLength / 3.0
            alpha_l = dist
            alpha_r = dist
        
        # First and last control points of the Bezier curve are
        # positioned exactly at the first and last data points
        # Control points 1 and 2 are positioned an alpha distance
        # out on the tangent vectors, left and right, respectively
        bezCurve[0] = d[first]
        bezCurve[3] = d[last]
        bezCurve[1] = bezCurve[0] + (tHat1 * alpha_l)
        bezCurve[2] = bezCurve[3] + (tHat2 * alpha_r)
    
    # Assign parameter values to digitized points using relative distances between points.
    def ChordLengthParameterize(d, first, last, u):
        u[first] = 0.0
        
        for i in range(first+1, last+1):
            u[i] = u[i - 1] + magnitude(d[i] - d[i - 1])
        
        scale = 1.0 / u[last]
        
        for i in range(first+1, last+1):
            u[i] *= scale
    
    def Reparameterize(d, first, last, u, bezCurve):
        for i in range(first, last+1):
            u[i] = NewtonRaphsonRootFind(bezCurve, d[i], u[i])
    
    # Q: Current fitted curve
    # P: Digitized point
    # u: Parameter value for "P"
    def NewtonRaphsonRootFind(Q, P, u):
        # Generate control vertices for Q'
        for i in range(3):
            Q1[i] = (Q[i + 1] - Q[i]) * 3.0
        
        # Generate control vertices for Q''
        for i in range(2):
            Q2[i] = (Q1[i + 1] - Q1[i]) * 2.0
        
        # Compute Q(u), Q'(u), Q''(u)
        Q_u = BezierII(3, Q, u)
        Q1_u = BezierII(2, Q1, u)
        Q2_u = BezierII(1, Q2, u)
        
        # Compute f(u)/f'(u)
        numerator = np.dot(Q_u - P, Q1_u)
        denominator = np.dot(Q1_u, Q1_u) + np.dot(Q_u - P, Q2_u)
        
        # u = u - f(u)/f'(u) # improved u
        if denominator != 0.0: u -= numerator / denominator
        
        return u
    
    # Find the maximum squared distance of digitized points to fitted curve.
    # splitPoint: Point of maximum error
    def ComputeMaxError(d, first, last, u, bezCurve):
        maxError = 0.0
        splitPoint = (last - first + 1) / 2
        
        for i in range(first+1, last):
            P = BezierII(3, bezCurve, u[i]) # Point on curve
            v = P - d[i] # Vector from point to curve
            dist = magnitude(v) # Current error
            
            if dist >= maxError:
                maxError = dist
                splitPoint = i
        
        return (splitPoint, maxError)
    
    # Evaluate a Bezier curve at a particular parameter value
    def BezierII(degree, points, t):
        for i in range(degree+1):
            Vtemp[i] = points[i]
        
        t1 = 1.0 - t
        
        # Triangle computation
        for i in range(1, degree+1):
            for j in range(degree+1 - i):
                Vtemp[j] = Vtemp[j]*t1 + Vtemp[j + 1]*t
        
        return Vtemp[0] # Q (point on curve at parameter t)
    
    # Approximate unit tangents at endpoints and "center" of digitized curve
    def ComputeLeftTangent(polyline, index):
        return normalized(polyline[index + 1] - polyline[index])
    
    def ComputeRightTangent(polyline, index):
        return normalized(polyline[index - 1] - polyline[index])
    
    def ComputeCenterTangent(polyline, index):
        return normalized(polyline[index - 1] - polyline[index + 1])
    
    return Fit
