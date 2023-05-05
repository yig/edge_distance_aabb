# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: infer_types=True
# distutils: language = c++
# distutils: extra_compile_args = -std=c++11 -O3
## This flag doesn't work:
## distutils: extra_compile_args = -march=native

## Author: Yotam Gingold <yotam@yotamgingold.com>
## License: CC0
## Compile with: cythonize --build edge_distance_aabb.pyx

ctypedef long Index
#ctypedef fused Real:
#    float
#    double
ctypedef double Real

def edges2points_and_linesegments( edges ):
    '''
    Given:
        edges: Edges in the format taken by `distances_to_edges()`
    Returns:
        Points, LineSegments: The format taken by `edge_distance_aabb`
    '''
    import numpy as np
    points = np.ascontiguousarray( edges.reshape(-1,2) )
    linesegments = np.ascontiguousarray( np.arange( len(points) ).reshape(-1,2) )
    return points, linesegments

cdef extern from "edge_distance_aabb.h" namespace "edge_distances":
    cdef cppclass Vector:
        Real x
        Real y
    cdef cppclass LineSegment:
        pass
    
    cdef cppclass Result:
        Real d
        Vector p
    
    cdef cppclass SlowDistanceComputer:
        SlowDistanceComputer() nogil
        void Init( const Vector* Points, Index NumPoints, const LineSegment* LineSegments, Index NumLineSegments ) nogil
        Result Distance( const Vector p ) nogil
    
    cdef cppclass AABBDistanceComputer:
        AABBDistanceComputer() nogil
        void Init( const Vector* Points, Index NumPoints, const LineSegment* LineSegments, Index NumLineSegments ) nogil
        Result Distance( const Vector p ) nogil
    
    cdef cppclass AABBTreeDistanceComputer:
        AABBTreeDistanceComputer() nogil
        void Init( const Vector* Points, Index NumPoints, const LineSegment* LineSegments, Index NumLineSegments ) nogil
        Result Distance( const Vector p ) nogil

###

cdef void SlowDistancesInternal( Real[:,::1] P, Real[:,::1] Points, Index[:,::1] LineSegments, Real[::1] DistancesOut, Real[:,::1] ClosestPointsOut ) nogil:
    cdef SlowDistanceComputer DistanceComputer
    DistanceComputer.Init(
        <const Vector*> &(Points[0,0]), Points.shape[0],
        <LineSegment*> &(LineSegments[0,0]), LineSegments.shape[0]
        )
    for i in range(P.shape[0]):
        result = DistanceComputer.Distance( ( <const Vector*> (&P[i,0]) )[0] )
        DistancesOut[i] = result.d
        ClosestPointsOut[i,0] = result.p.x
        ClosestPointsOut[i,1] = result.p.y

def SlowDistances( Real[:,::1] P, Real[:,::1] Points, Index[:,::1] LineSegments ):
    '''
    Given:
        P: An N-by-2 array of (2D) points whose distance to query.
        Points: A K-by-2 array of (2D) points involved in the line segments.
        LineSegments: An E-by-2 array of pairs of indices into `Points`, one for each line segment.
    Returns:
        A length-N array of distances, one for each point in `P`, from that point to the line segments.
        A length-N array of closest points, one for each point in `P`, from that point to the line segments.
    '''
    import numpy as np
    DistancesOut = np.zeros( P.shape[0] )
    # memoryview `.shape` has extra 0s.
    # Source: https://stackoverflow.com/questions/67557753/cython-memoryview-shape-incorrect
    ClosestPointsOut = np.zeros( ( P.shape[0], P.shape[1] ) )
    SlowDistancesInternal( P, np.ascontiguousarray( Points ), np.ascontiguousarray( LineSegments ), DistancesOut, ClosestPointsOut )
    return DistancesOut, ClosestPointsOut

## This is half the speed of the cpdef version below:
# def SlowDistance( Real[::1] P, Real[:,::1] Points, Index[:,::1] LineSegments ):
#    return SlowDistances( P[None,:], Points, LineSegments )[0]

cpdef (Real,(Real,Real)) SlowDistance( Real[::1] P, Real[:,::1] Points, Index[:,::1] LineSegments ) nogil:
    '''
    Given:
        P: A 2D point whose distance to query.
        Points: A K-by-2 array of (2D) points involved in the line segments.
        LineSegments: An E-by-2 array of pairs of indices into `Points`, one for each line segment.
    Returns:
        The distance from `P` to the line segments.
        The closest point from `P` to the line segments.
    
    For routines which perform pre-computation to accelerate many queries, this
    one-point-at-a-time distance function will be much slower than the
    many-points-at-a-time version.
    '''
    cdef SlowDistanceComputer DistanceComputer
    DistanceComputer.Init(
        <const Vector*> &(Points[0,0]), Points.shape[0],
        <LineSegment*> &(LineSegments[0,0]), LineSegments.shape[0]
        )
    cdef Result result = DistanceComputer.Distance( ( <const Vector*> (&P[0]) )[0] )
    return result.d, (result.p.x, result.p.y)

###

cdef void AABBDistancesInternal( Real[:,::1] P, Real[:,::1] Points, Index[:,::1] LineSegments, Real[::1] DistancesOut, Real[:,::1] ClosestPointsOut ) nogil:
    cdef AABBDistanceComputer DistanceComputer
    DistanceComputer.Init(
        <const Vector*> &(Points[0,0]), Points.shape[0],
        <LineSegment*> &(LineSegments[0,0]), LineSegments.shape[0]
        )
    for i in range(P.shape[0]):
        result = DistanceComputer.Distance( ( <const Vector*> (&P[i,0]) )[0] )
        DistancesOut[i] = result.d
        ClosestPointsOut[i,0] = result.p.x
        ClosestPointsOut[i,1] = result.p.y

def AABBDistances( Real[:,::1] P, Real[:,::1] Points, Index[:,::1] LineSegments ):
    '''
    Given:
        P: An N-by-2 array of (2D) points whose distance to query.
        Points: A K-by-2 array of (2D) points involved in the line segments.
        LineSegments: An E-by-2 array of pairs of indices into `Points`, one for each line segment.
    Returns:
        A length-N array of distances, one for each point in `P`, from that point to the line segments.
        A length-N array of closest points, one for each point in `P`, from that point to the line segments.
    '''
    import numpy as np
    DistancesOut = np.zeros( P.shape[0] )
    # memoryview `.shape` has extra 0s.
    # Source: https://stackoverflow.com/questions/67557753/cython-memoryview-shape-incorrect
    ClosestPointsOut = np.zeros( ( P.shape[0], P.shape[1] ) )
    AABBDistancesInternal( P, np.ascontiguousarray( Points ), np.ascontiguousarray( LineSegments ), DistancesOut, ClosestPointsOut )
    return DistancesOut, ClosestPointsOut

## The SlowDistance version of this is half the speed of the cpdef version:
# def AABBDistance( Real[::1] P, Real[:,::1] Points, Index[:,::1] LineSegments ):
#    return AABBDistances( P[None,:], Points, LineSegments )[0]

cpdef (Real,(Real,Real)) AABBDistance( Real[::1] P, Real[:,::1] Points, Index[:,::1] LineSegments ) nogil:
    '''
    Given:
        P: A 2D point whose distance to query.
        Points: A K-by-2 array of (2D) points involved in the line segments.
        LineSegments: An E-by-2 array of pairs of indices into `Points`, one for each line segment.
    Returns:
        The distance from `P` to the line segments.
        The closest point from `P` to the line segments.
    
    For routines which perform pre-computation to accelerate many queries, this
    one-point-at-a-time distance function will be much slower than the
    many-points-at-a-time version.
    '''
    cdef AABBDistanceComputer DistanceComputer
    DistanceComputer.Init(
        <const Vector*> &(Points[0,0]), Points.shape[0],
        <LineSegment*> &(LineSegments[0,0]), LineSegments.shape[0]
        )
    cdef Result result = DistanceComputer.Distance( ( <const Vector*> (&P[0]) )[0] )
    return result.d, (result.p.x, result.p.y)

###

cdef void AABBTreeDistancesInternal( Real[:,::1] P, Real[:,::1] Points, Index[:,::1] LineSegments, Real[::1] DistancesOut, Real[:,::1] ClosestPointsOut ) nogil:
    cdef AABBTreeDistanceComputer DistanceComputer
    DistanceComputer.Init(
        <const Vector*> &(Points[0,0]), Points.shape[0],
        <LineSegment*> &(LineSegments[0,0]), LineSegments.shape[0]
        )
    for i in range(P.shape[0]):
        result = DistanceComputer.Distance( ( <const Vector*> (&P[i,0]) )[0] )
        DistancesOut[i] = result.d
        ClosestPointsOut[i,0] = result.p.x
        ClosestPointsOut[i,1] = result.p.y

def AABBTreeDistances( Real[:,::1] P, Real[:,::1] Points, Index[:,::1] LineSegments ):
    '''
    Given:
        P: An N-by-2 array of (2D) points whose distance to query.
        Points: A K-by-2 array of (2D) points involved in the line segments.
        LineSegments: An E-by-2 array of pairs of indices into `Points`, one for each line segment.
    Returns:
        A length-N array of distances, one for each point in `P`, from that point to the line segments.
        A length-N array of closest points, one for each point in `P`, from that point to the line segments.
    '''
    import numpy as np
    DistancesOut = np.zeros( P.shape[0] )
    # memoryview `.shape` has extra 0s.
    # Source: https://stackoverflow.com/questions/67557753/cython-memoryview-shape-incorrect
    ClosestPointsOut = np.zeros( ( P.shape[0], P.shape[1] ) )
    AABBTreeDistancesInternal( P, np.ascontiguousarray( Points ), np.ascontiguousarray( LineSegments ), DistancesOut, ClosestPointsOut )
    return DistancesOut, ClosestPointsOut

## The SlowDistance version of this is half the speed of the cpdef version:
# def AABBTreeDistance( Real[::1] P, Real[:,::1] Points, Index[:,::1] LineSegments ):
#    return AABBTreeDistances( P[None,:], Points, LineSegments )[0]

cpdef (Real,(Real,Real)) AABBTreeDistance( Real[::1] P, Real[:,::1] Points, Index[:,::1] LineSegments ) nogil:
    '''
    Given:
        P: A 2D point whose distance to query.
        Points: A K-by-2 array of (2D) points involved in the line segments.
        LineSegments: An E-by-2 array of pairs of indices into `Points`, one for each line segment.
    Returns:
        The distance from `P` to the line segments.
        The closest point from `P` to the line segments.
    
    For routines which perform pre-computation to accelerate many queries, this
    one-point-at-a-time distance function will be much slower than the
    many-points-at-a-time version.
    '''
    cdef AABBTreeDistanceComputer DistanceComputer
    DistanceComputer.Init(
        <const Vector*> &(Points[0,0]), Points.shape[0],
        <LineSegment*> &(LineSegments[0,0]), LineSegments.shape[0]
        )
    cdef Result result = DistanceComputer.Distance( ( <const Vector*> (&P[0]) )[0] )
    return result.d, (result.p.x, result.p.y)
