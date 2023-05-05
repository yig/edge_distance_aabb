import time
import numpy as np

def distances_to_edges( pts, edges ):
    '''
    Input parameter 'pts' has dimensions (x,y,...) x #pts.
    Input parameter 'edges' has dimensions ... x #edges x 2 endpoints x N coordinates (x,y,...).
    Returns a tuple, where the first element is
    an array of distances with dimensions #edges x #pts
    and the second element is
    an array of gradients with respect to 'pts' with dimensions #edges x N coordinates (x,y,...) x #pts.
    '''
    pts = np.asarray( pts, float )
    ## pts has dimensions N (x,y,...) x #pts
    edges = np.asarray( edges, float )
    ## edges has dimensions ... x #edges x 2 endpoints x N coordinates (x,y,...)
    
    N = pts.shape[0]
    
    assert len( pts.shape ) == 2 and pts.shape[0] == N
    assert edges.shape[-2] == 2 and edges.shape[-1] == N
    
    ## get distance squared to each edge:
    ##   let p = black_pixel_pos, a = endpoint0, b = endpoint1, d = ( b-a ) / dot( b-a,b-a )
    ##   dot( p-a, d ) < 0 => dot( p-a, p-a )
    ##   dot( p-a, d ) > 1 => dot( p-b, p-b )
    ##   else              => dot( p-a - dot( p-a, d ) * d, same )
    p_a = pts[np.newaxis,...] - edges[...,:,0,:,np.newaxis]
    p_b = pts[np.newaxis,...] - edges[...,:,1,:,np.newaxis]
    ## p_a and p_b have dimensions ... x #edges x N coordinates (x,y,...) x #pts
    b_a = edges[...,:,1,:] - edges[...,:,0,:]
    ## b_a has dimensions ... x #edges x N coordinates (x,y,...)
    d = b_a / ( b_a**2 ).sum( -1 )[...,np.newaxis]
    ## d has same dimensions as b_a
    assert b_a.shape == d.shape
    cond = ( p_a * d[...,np.newaxis] ).sum( -2 )
    ## cond has dimensions ... x #edges x #pts
    assert cond.shape[-2:] == (edges.shape[-3], pts.shape[-1])
    
    cond_lt_zero = cond < 0
    cond_gt_one = cond > 1
    cond_else = np.logical_not( np.logical_or( cond_lt_zero, cond_gt_one ) )
    ## cond_* have dimensions ... x #edges x #pts
    
    #distancesSqr = empty( cond.shape, Real )
    ## distancesSqr has dimensions ... x #edges x #pts
    #assert distancesSqr.shape[-2:] == (edges.shape[-3], pts.shape[-1])
    
    ## else case
    # distancesSqr = p_a - cond[:,newaxis,:] * b_a[...,newaxis]
    # distancesSqr = ( distancesSqr**2 ).sum( 1 )
    # <=>
    # distancesSqr = ( ( p_a - cond[:,newaxis,:] * b_a[...,newaxis] )**2 ).sum( -2 )
    # <=>
    # distancesSqr[ cond_else ] = ( ( p_a - cond[:,newaxis,:] * b_a[...,newaxis] )**2 ).sum( -2 )[ cond_else ]
    distancesSqr = ( ( p_a - cond[:,np.newaxis,:] * b_a[...,np.newaxis] )**2 ).sum( -2 )
    # <=>
    # distancesSqr[ cond_else ] = ( (
    #    swapaxes( p_a, -1, -2 )[ cond_else ] - swapaxes( cond[:,newaxis,:] * b_a[...,newaxis], -1, -2 )[ cond_else ]
    #    )**2 ).sum( -1 )
    
    ## < 0 case
    # distancesSqr[ cond < 0 ] = ( p_a**2 ).sum( -2 )[ cond < 0 ]
    # <=>
    distancesSqr[ cond_lt_zero ] = ( p_a**2 ).sum( -2 )[ cond_lt_zero ]
    # <=>
    # distancesSqr[ cond_lt_zero ] = ( swapaxes( p_a, -1, -2 )[ cond_lt_zero ]**2 ).sum( -1 )
    
    ## > 1 case
    # distancesSqr[ cond > 1 ] = ( p_b**2 ).sum( -2 )[ cond > 1 ]
    # <=>
    distancesSqr[ cond_gt_one ] = ( p_b**2 ).sum( -2 )[ cond_gt_one ]
    # <=>
    # distancesSqr[ cond_gt_one ] = ( swapaxes( p_b, -1, -2 )[ cond_gt_one ]**2 ).sum( -1 )
    
    #print 'distancesSqr:', distancesSqr
    #print 'distances:', sqrt( distancesSqr.min(0) )
    
    ## distancesSqr is now distances
    np.sqrt( distancesSqr, distancesSqr )
    ## we'll just rename distancesSqr
    distances = distancesSqr
    ## distances has dimensions ... x #edges x #pts
    del distancesSqr
    return distances

def test_distances_to_edges_simple():
    theta = np.random.uniform() * 2 * np.pi
    R = np.array( [ [ np.cos( theta ), np.sin( theta ) ], [ -np.sin( theta ), np.cos( theta ) ] ], float )
    s = np.random.uniform() + .5
    R *= s
    
    edges = np.array( [ [ R@[ 0, 0 ], R@[ 2, 0 ] ] ], float )
    
    ds = distances_to_edges(
        np.array( [ R@[ -1, 0 ], R@[ -np.sqrt(2)/2, np.sqrt(2)/2 ], R@[ 0, 1 ], R@[ .5, 1 ], R@[ 1, 1 ], R@[ 1.5, 1 ], R@[ 2, 1 ], R@[ 2+np.sqrt(2)/2, -np.sqrt(2)/2 ], R@[ 3, 0 ] ], float ).T,
        edges
        )
    print( 'Distance error:', np.abs( ds - s ).max() )
    
    ds = distances_to_edges(
        np.array( [ R@[ -2, 0 ], R@[ 4, 0 ] ], float ).T,
        edges
        )
    print( 'Distance error:', np.abs( ds - 2*s ).max() )
    
    ds = distances_to_edges(
        np.array( [ R@[ 0, 0 ], R@[ 1, 0 ], R@[ 2, 0 ] ], float ).T,
        edges
        )
    print( 'Distance error:', np.abs( ds ).max() )

def test_AABB_distances_simple():
    import edge_distance_aabb

    ## Create some edges
    Points = np.array( [ [ 0, 0 ], [ 1, 1 ], [ 2, 0 ] ], float )
    LineSegments = np.array( [ [ 0, 1 ], [ 1, 2 ] ] )
    print( "Points:", Points )
    print( "LineSegments:", LineSegments )
    
    ## We want the distance to the following points
    P = np.array( [ [ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 2, 1 ], [ 2, 0 ] ], float )
    print( "P:", P )
    
    ## Compute using AABB to cull edges
    distances, closest_points = edge_distance_aabb.AABBDistances( P, Points, LineSegments )
    print( "distances (should be: 0, √2/2, √2/2, √2/2, 0):", distances )
    print( "closest_points (should be: (0,0), (½,½), (½ or 1½,½), (1½,½), (2,0) ):", closest_points )

def test():
    # import pyximport; pyximport.install()
    import edge_distance_aabb
    import gzip
    
    print( "Loading data..." )
    
    with gzip.GzipFile('distances_to_edges_data.npy.gz') as f:
        pts = np.load(f)
        edges = np.load(f)
    
    print( "...done" )
    
    '''
    edges = np.tile( edges, ( 4,1,1 ) )
    edges[edges.shape[0]//4:,:,:] *= -1
    edges[2*edges.shape[0]//4:,:,0] *= -1
    edges[3*edges.shape[0]//4:,:,1] *= -1
    '''
    
    '''
    edges = np.tile( edges, ( 2,1,1 ) )
    edges[edges.shape[0]//2:,:,:] *= -1
    '''
    
    print( "Testing distances_to_edges()..." )
    start_time = time.time()
    ds_original = distances_to_edges( pts, edges )
    print( "...done. Took seconds:", time.time() - start_time )
    ## Keep only the minimum distance for each input point
    ds_original = ds_original.min(axis=0)
    
    import edge_distance_aabb
    print( "pts.shape:", pts.shape )
    print( "edges.shape:", edges.shape )
    EdgePoints, LineSegments = edge_distance_aabb.edges2points_and_linesegments( edges )
    print( "EdgePoints.shape:", EdgePoints.shape )
    print( "LineSegments.shape:", LineSegments.shape )
    
    pts = np.ascontiguousarray( pts.T )
    
    print( "Testing edge_distance_aabb.SlowDistances()..." )
    start_time = time.time()
    ds_slows, _ = edge_distance_aabb.SlowDistances( pts, EdgePoints, LineSegments )
    print( "...done. Took seconds:", time.time() - start_time )
    print( "Max abs difference:", np.abs( ds_original - ds_slows ).max() )
    
    print( "Testing edge_distance_aabb.SlowDistance()..." )
    start_time = time.time()
    ds_slow = [ edge_distance_aabb.SlowDistance( pt, EdgePoints, LineSegments )[0] for pt in pts ]
    print( "...done. Took seconds:", time.time() - start_time )
    print( "Max abs difference:", np.abs( ds_original - ds_slow ).max() )
    
    print( "Testing edge_distance_aabb.AABBDistances()..." )
    start_time = time.time()
    ds_AABBs, _ = edge_distance_aabb.AABBDistances( pts, EdgePoints, LineSegments )
    print( "...done. Took seconds:", time.time() - start_time )
    print( "Max abs difference:", np.abs( ds_original - ds_AABBs ).max() )
    
    print( "Testing edge_distance_aabb.AABBDistance()..." )
    start_time = time.time()
    ds_AABB = [ edge_distance_aabb.AABBDistance( pt, EdgePoints, LineSegments )[0] for pt in pts ]
    print( "...done. Took seconds:", time.time() - start_time )
    print( "Max abs difference:", np.abs( ds_original - ds_AABB ).max() )
    
    print( "Testing edge_distance_aabb.AABBTreeDistances()..." )
    start_time = time.time()
    ds_AABBTrees, _ = edge_distance_aabb.AABBTreeDistances( pts, EdgePoints, LineSegments )
    print( "...done. Took seconds:", time.time() - start_time )
    print( "Max abs difference:", np.abs( ds_original - ds_AABBTrees ).max() )
    
    print( "Testing edge_distance_aabb.AABBTreeDistance()..." )
    start_time = time.time()
    ds_AABBTree = [ edge_distance_aabb.AABBTreeDistance( pt, EdgePoints, LineSegments )[0] for pt in pts ]
    print( "...done. Took seconds:", time.time() - start_time )
    print( "Max abs difference:", np.abs( ds_original - ds_AABBTree ).max() )
    
    import pdb
    pdb.set_trace()

if __name__ == '__main__':
    test_distances_to_edges_simple()
    test_AABB_distances_simple()
    test()
