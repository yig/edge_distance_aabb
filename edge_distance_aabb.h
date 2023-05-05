#pragma once

// Author: Yotam Gingold <yotam@yotamgingold.com>
// License: CC0

#include <cassert>
#include <cmath> // sqrt
#include <algorithm> // std::min, std::max, std::sort
#include <iostream>
#include <limits>
#include <vector>
#include <queue>
#include <tuple>

/*
Given:
    Nx2 pts
    Ex2 edges

Then:
    SlowDistance( p, pts, edges )
*/

/*
Given:
    Nx2 pts
    Ex2 edges

Prepare:
    struct AABB { 2 min; 2 max }
    AABB[] aabbs = make_AABBs( pts, edges )

Then:
    NotSoFastDistance( p, pts, edges, aabbs )
*/

/*
Given:
    real Nx2 pts
    int Ex2 edges

Prepare:
    struct AABBTreeNode { real 2 min; real 2 max; int 1 left; int 1 right; int 1 edge; }
    AABBTreeNode[] aabbtree = make_AABBTree( pts, edges )

Then:
    FastDistance( p, pts, edges, aabbtree )
*/


namespace edge_distances {

typedef double Real;

using std::min;
using std::max;
Real clamp( Real val, Real minval, Real maxval ) { return max(minval,min(maxval,val)); }

struct Vector {
    Real x = 0, y = 0;
    
    Real operator[]( int index ) const {
        assert( index >= 0 && index < 2 );
        return (&x)[index];
    }
    
    Vector() = default;
    Vector( Real x_, Real y_ ) { x = x_; y = y_; }
};
Vector operator-( const Vector& L, const Vector& R ) {
    return Vector( L.x - R.x, L.y - R.y );
}
Vector operator+( const Vector& L, const Vector& R ) {
    return Vector( L.x + R.x, L.y + R.y );
}
Vector operator*( Real s, const Vector& V ) {
    return Vector( s*V.x, s*V.y );
}
Vector& operator*=( Vector& V, Real s ) {
    V.x *= s;
    V.y *= s;
    return V;
}
Real Dot( const Vector& a, const Vector& b ) {
    return a.x*b.x + a.y*b.y;
}
Real Length( const Vector& d ) {
    return sqrt( Dot( d, d ) );
}
Real DistanceSqr( const Vector& a, const Vector& b ) {
    Vector d( a-b );
    return Dot( d, d );
}
Real Distance( const Vector& a, const Vector& b ) {
    return sqrt( DistanceSqr( a, b ) );
}
Vector Minimum( const Vector& a, const Vector& b ) {
    return Vector(
        min( a.x, b.x ),
        min( a.y, b.y )
    );
}
Vector Maximum( const Vector& a, const Vector& b ) {
    return Vector(
        max( a.x, b.x ),
        max( a.y, b.y )
    );
}
Vector Clamp( const Vector& val, const Vector& min, const Vector& max ) {
    return Vector(
        clamp( val.x, min.x, max.x ),
        clamp( val.y, min.y, max.y )
    );
}

struct Result {
    Real d = -31337;
    Vector p;
    
    Result() = default;
    Result( Real d_, const Vector& p_ ) : d( d_ ), p( p_ ) {}
    bool operator<( const Result& rhs ) const { return std::tie( d, p.x, p.y ) < std::tie( rhs.d, rhs.p.x, rhs.p.y ); }
    
    Result& Sqrt() { d = sqrt(d); return *this; }
};

Result PointLineSegmentDistanceSqr( const Vector& p, const Vector& A, const Vector& B ) {
    // Return point `p` to line segment `AB` distance and closest point
    
    // Get the normalized vector from A to B.
    Vector along( B-A );
    const Real l = Length( along );
    if( l == 0 ) return Result( Distance( p, A ), A );
    along *= 1./l;
    
    // Project p onto the line.
    const Vector p_minus_A( p-A );
    Real t = Dot( p_minus_A, along );
    if( t < 0 ) { return Result( DistanceSqr( p, A ), A ); }
    else if( t > l ) { return Result( DistanceSqr( p, B ), B ); }
    else {
        const Vector q = A + t*along;
        return Result( DistanceSqr( p, q ), q );
    }
}

typedef long Index;

struct LineSegment {
    Index A = -31337;
    Index B = -31337;
};

class SlowDistanceComputer {
public:
    const Vector* Points = nullptr;
    Index NumPoints = 0;
    const LineSegment* LineSegments = nullptr;
    Index NumLineSegments = 0;
    
    void Init( const Vector* Points_, const Index NumPoints_, const LineSegment* LineSegments_, const Index NumLineSegments_ ) {
        Points = Points_;
        NumPoints = NumPoints_;
        LineSegments = LineSegments_;
        NumLineSegments = NumLineSegments_;
        
        assert( Points );
        assert( NumPoints > 0 );
        assert( LineSegments );
        assert( NumLineSegments > 0 );
        for( Index i = 0; i < NumLineSegments; ++i ) {
            assert( LineSegments[i].A >= 0 && LineSegments[i].A < NumPoints );
            assert( LineSegments[i].B >= 0 && LineSegments[i].B < NumPoints );
        }
        /*
        std::cerr << "Points: " << Points << '\n';
        std::cerr << "NumPoints: " << NumPoints << '\n';
        std::cerr << "LineSegments: " << LineSegments << '\n';
        std::cerr << "NumLineSegments: " << NumLineSegments << '\n';
        */
    }
    
    // A convenience method based on the pointers to data stored in this class.
    Result PointLineSegmentIndexDistanceSqr( const Vector& p, Index LineSegment_index ) {
        return PointLineSegmentDistanceSqr( p, Points[ LineSegments[LineSegment_index].A ], Points[ LineSegments[LineSegment_index].B ] );
    }
    
    Result Distance( const Vector& p, Real max_abs_dist = std::numeric_limits<Real>::infinity() ) {
        if( NumLineSegments == 0 ) return Result( -31337, Vector() );
        
        Result min_dist = Result( max_abs_dist, Vector() );
        for( Index i = 0; i < NumLineSegments; ++i ) {
            const Result d = PointLineSegmentIndexDistanceSqr( p, i );
            min_dist = min( min_dist, d );
        }
        
        return min_dist.Sqrt();
    }
};

struct AABB {
    // The two corners of the box.
    Vector min, max;
    // A midpoint getter.
    const Vector mid() { return 0.5*( min + max ); }
    
    /*
    Compute the axis aligned bounding box of a shape.
    For any object made by linear interpolation of any subset of these points
    (e.g. line segments, triangles, polygons, polyhedra), it is sufficient to use
    the points themselves.
    */
    AABB( const Vector* pts, Index num_pts ) {
        assert( pts );
        assert( num_pts > 0 );
        
        min = pts[0];
        max = pts[0];
        
        for( int i = 1; i < num_pts; i++ ) {
            const Vector& pt = pts[i];
            min = Minimum( min, pt );
            max = Maximum( max, pt );
        }
    }
    AABB( const Vector& A, const Vector& B ) {
        min = Minimum( A, B );
        max = Maximum( A, B );
    }
    // AABB from the Union of two other AABBs.
    AABB( const AABB& left, const AABB& right ) {
        min = Minimum( left.min, right.min );
        max = Maximum( left.max, right.max );
    }
    
    Real DistanceLowerBoundSqr( const Vector& p ) const {
        // Clamp each dimension of p to the corresponding dimensions in corner min and max.
        return DistanceSqr( p, Clamp( p, min, max ) );
    }
};

// Distances using a flat list of axis-aligned bounding boxes.
class AABBDistanceComputer : public SlowDistanceComputer {
public:
    // An array of axis-aligned bounding boxes, one for each line segment:
    std::vector< AABB > AABBs;
    
    void Init( const Vector* Points_, const Index NumPoints_, const LineSegment* LineSegments_, const Index NumLineSegments_ ) {
        SlowDistanceComputer::Init( Points_, NumPoints_, LineSegments_, NumLineSegments_ );
        
        // Compute the axis aligned bounding boxes.
        AABBs.reserve( NumLineSegments );
        for( int i = 0; i < NumLineSegments; ++i ) {
            AABBs.emplace_back( Points[ LineSegments[i].A ], Points[ LineSegments[i].B ] );
        }
    }
    
    // Compute the signed distance in O(n) time by iterating over all line segments.
    // The acceleration comes from computing the distance to each
    // line segment's axis-aligned bounding box before computing the actual point-to-line segment
    // distance. If the distance to the bounding box is larger than the best distance
    // we've found so far, then we can skip computing the more expensive line segment distance.
    Result Distance( const Vector& p, Real max_abs_dist = std::numeric_limits<Real>::infinity() ) {
        assert( NumLineSegments == AABBs.size() );
        
        if( NumLineSegments == 0 ) return Result();
        
        Result min_dist = Result( max_abs_dist, Vector() );
        for( Index i = 0; i < NumLineSegments; ++i ) {
            if( AABBs[i].DistanceLowerBoundSqr( p ) < min_dist.d ) {
                const Result d = PointLineSegmentIndexDistanceSqr( p, i );
                min_dist = min( min_dist, d );
            }
        }
        return min_dist.Sqrt();
    }
};

// Mesh signed distance using a binary tree of axis-aligned bounding boxes.
class AABBTreeDistanceComputer : public AABBDistanceComputer {
public:
    // A tree node.
    struct AABBTreeNode {
        AABB aabb;
        // The node is either internal and has two children:
        Index AABBTreeNode_left = -1;
        Index AABBTreeNode_right = -1;
        // Or it is a leaf node and has a line segment:
        Index LineSegment_index = -1;
        
        AABBTreeNode( const AABB& aabb_ ) : aabb( aabb_ ) {};
    };
    
    // The root of our binary tree.
    std::vector< AABBTreeNode > tree;
    Index root = -1;
    
    void Init( const Vector* Points_, const Index NumPoints_, const LineSegment* LineSegments_, const Index NumLineSegments_ ) {
        // Call the superclass Init(). It computes an AABB per line segment.
        AABBDistanceComputer::Init( Points_, NumPoints_, LineSegments_, NumLineSegments_ );
        
        /// Make an AABB tree for all triangles.
        // Reserve space for the nodes. There will be 1 + 1/2 + 1/4 + 1/8 + ... = 2 times
        // the number of line segments.
        tree.reserve( 2*NumLineSegments );
        // Create the sequence of all line segments.
        std::vector< Index > line_segment_indices( NumLineSegments );
        for( Index i = 0; i < NumLineSegments; ++i ) { line_segment_indices[i] = i; }
        // Build the tree recursively.
        root = AABBTreeForLineSegments( line_segment_indices.data(), line_segment_indices.size() );
    }
    
    /*
    Computes the AABB hierarchy for the given list of line segments.
    */
    Index AABBTreeForLineSegments( Index* line_segment_indices, Index num_line_segment_indices ) {
        // There must be at least one triangle.
        assert( num_line_segment_indices > 0 );
        
        // Outline: Take union of all children AABB. Split along midpoint of long axis. Recurse until one there's only triangle per AABB.
        
        Index result = tree.size();
        // AABB has no default constructor, so start with the 0-th line segment.
        // tree.resize( result+1 );
        tree.emplace_back( AABBs[ line_segment_indices[0] ] );
        
        // If we're only storing one triangle, create a leaf node.
        if( num_line_segment_indices == 1 ) {
            // tree[result].aabb = AABBs[ line_segment_indices[0] ];
            tree[result].LineSegment_index = line_segment_indices[0];
        }
        // Otherwise, create a node for all the children.
        else {
            // Union the AABBs of all line segments.
            // tree[result].aabb = AABBs[ line_segment_indices[0] ];
            for( int i = 1; i < num_line_segment_indices; ++i ) {
                tree[result].aabb = AABB( tree[result].aabb, AABBs[ line_segment_indices[i] ] );
            }
            
            // Find the longest coordinate.
            Vector diag = tree[result].aabb.max - tree[result].aabb.min;
            Index coord_index = 0;
            if( diag[1] > diag[0] ) coord_index = 1;
            
            // Sort line_segment_indices according to this dimension.
            std::sort(
                line_segment_indices, line_segment_indices + num_line_segment_indices,
                [&]( Index L, Index R ) { return AABBs[ L ].mid()[coord_index] < AABBs[ R ].mid()[coord_index]; }
                );
            
            tree[result].AABBTreeNode_left = AABBTreeForLineSegments( line_segment_indices, num_line_segment_indices/2 );
            tree[result].AABBTreeNode_right = AABBTreeForLineSegments( line_segment_indices + num_line_segment_indices/2, num_line_segment_indices - num_line_segment_indices/2 );
        }
        
        return result;
    }
    
    Result Distance( const Vector& p, Real max_abs_dist = std::numeric_limits<Real>::infinity() ) {
        assert( NumLineSegments == AABBs.size() );
        
        if( NumLineSegments == 0 ) return Result( -31337, Vector() );
        
        Result min_dist = Result( max_abs_dist, Vector() );
        
        // Create a queue with the root node's distance lower bound.
        std::priority_queue< std::tuple< Real, Index > > q;
        // Store negative distance since `std::priority_queue` keeps the largest item at the front.
        // UPDATE: For some reason using std::greater<> for the std::priority_queue slows it down.
        q.push( std::make_tuple( -tree[root].aabb.DistanceLowerBoundSqr( p ), root ) );
        // While the queue isn't empty and its smallest item is better than the
        // minimum distance seen so far, update the minimum distance seen so far.
        while( !q.empty() && -std::get<0>( q.top() ) < min_dist.d ) {
            // Pop the first element (minimum distance).
            auto front = q.top();
            q.pop();
            
            // Internal node.
            const AABBTreeNode& node = tree[std::get<1>(front)];
            if( node.LineSegment_index == -1 ) {
                assert( node.AABBTreeNode_left >= 0 && node.AABBTreeNode_left < tree.size() );
                assert( node.AABBTreeNode_right >= 0 && node.AABBTreeNode_right < tree.size() );
                
                const Real dleft = tree[node.AABBTreeNode_left].aabb.DistanceLowerBoundSqr( p );
                const Real dright = tree[node.AABBTreeNode_right].aabb.DistanceLowerBoundSqr( p );
                if( dleft < min_dist.d ) {
                    // Store negative distance since `std::priority_queue` keeps the largest item at the front.
                    q.push( std::make_tuple( -dleft, node.AABBTreeNode_left ) );
                }
                if( dright < min_dist.d ) {
                    // Store negative distance since `std::priority_queue` keeps the largest item at the front.
                    q.push( std::make_tuple( -dright, node.AABBTreeNode_right ) );
                }
            }
            // Leaf node
            else {
                assert( node.AABBTreeNode_left == -1 );
                assert( node.AABBTreeNode_right == -1 );
                assert( node.LineSegment_index >= 0 && node.LineSegment_index < NumLineSegments );
                
                const Result d = PointLineSegmentIndexDistanceSqr( p, node.LineSegment_index );
                min_dist = min( min_dist, d );
            }
        }
        return min_dist.Sqrt();
    }
};

}
