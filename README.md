# Fast point-to-edge distances

Computes the distance and closest point from a set of points to a set of line segments relatively efficiently. The code is fast because:

* It is written in C++ (as a header-only library). It is exposed to Python with Cython.
* It uses axis-aligned bounding boxes for culling or in a hierarchy.

## Example

The Python function signatures are:

```
def SlowDistances( Real[:,::1] P, Real[:,::1] Points, Index[:,::1] LineSegments ):
def AABBDistances( Real[:,::1] P, Real[:,::1] Points, Index[:,::1] LineSegments ):
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
```

For example:

```
import edge_distance_aabb

## Create some edges
Points = np.array( [ [ 0, 0 ], [ 1, 1 ], [ 2, 0 ] ], float )
LineSegments = np.array( [ [ 0, 1 ], [ 1, 2 ] ] )
print( "Points:", Points )
print( "LineSegments:", LineSegments )

## We want the distance to the following points
P = np.array( [ [ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 2, 1 ], [ 2, 0 ] ], float )
print( "P:", P )

## Compute distances and closest points using AABB to cull edges.
distances, closest_points = edge_distance_aabb.AABBDistances( P, Points, LineSegments )
print( "distances (should be: 0, √2/2, √2/2, √2/2, 0):", distances )
print( "closest_points (should be: (0,0), (½,½), (½ or 1½,½), (1½,½), (2,0) ):", closest_points )
```

## Installation

1. From PyPI:

```
pip install edge_distance_aabb
```

2. By copying `edge_distance_aabb.pyx` and `edge_distance_aabb.h` into your directory and running `cythonize --build edge_distance_aabb.pyx`.

## Dependencies

`numpy` and `Cython`

## Distribution

`python3 -m build`
