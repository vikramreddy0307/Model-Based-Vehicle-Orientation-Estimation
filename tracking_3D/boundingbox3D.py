'''
Created on May 13, 2020

@author: duolu
'''


import numpy as np
import cv2

from calibration import Camera2DGroundModel
from visualization import FrameVis


class BB3D(object):
    '''
    3D bounding box data structure.
    
    "situation" means different situations that different facets of the 3D
        bounding box are visible. Currently, three situations are considered.
        "3vpp": three-vanishing-point-perspective in general
        "xvpp": dominated by the x-axis vanishing point
        "yvpp": dominated by the y-axis vanishing point
    
    "cases" means different cases in each situation, indicating which points
        of the BB3D are visible.
    
    "nbox" means the eight points of BB3D in p1 to p8, determined by the
        sequence of the algorithm.
        
    "pbox" means the eight points of BB3D in pa to ph, relative to the
        orientation of the detected object. For example, pa is always the
        front-left corner close to the ground.
        
    "qbox" means the eight points of BB3D in 3D space, computed from the
        "pbox" using the camera model.
        
    "vx", "vy", "vz" are the orientation vectors of the detected object.
    
    "vpx", "vpy", "vpz" are the three vanishing points in the direction of
        "vx", "vy", "vz". The vanishing points are calculated using the
        camera model.
    
    '''

    # For serialization / deserialization.
    m1 = 2 + 6 # situation, case, orientation
    m2 = 3 * 8 # 3D bounding box on the frame with visibility flag
    m3 = 3 * 8 # 3D bounding box in 3D space
    m4 = 3 # dimension
    m = m1 + m2 + m3 + m4
    
    def __init__(self, situation, cases, pbox, visibility, 
                 qbox, dim, vx, vy, vz):

        # CAUTION: "situation" and "cases" are integers. All others are
        # numpy arrays of floating point numbers.
        self.situation = situation
        self.cases = cases

        self.vx = vx
        self.vy = vy
        self.vz = vz
        
        self.pbox = pbox
        self.visibility = visibility
        
        self.qbox = qbox
        self.dim = dim
        
    @classmethod
    def to_array(cls, bb3d):

        m1 = cls.m1
        m2 = cls.m2
        m3 = cls.m3
        m4 = cls.m4
        
        a1 = np.zeros(m1)
        a2 = np.zeros(m2)
        a3 = np.zeros(m3)
        a4 = np.zeros(m4)

        a1[0] = bb3d.situation
        a1[1] = bb3d.cases
        
        a1[2:4] = bb3d.vx
        a1[4:6] = bb3d.vy
        a1[6:8] = bb3d.vz
        
        for j in range(8):
            
            idx0 = 3 * j
            idx1 = 3 * j + 1
            idx2 = 3 * j + 2
            
            a2[idx0] = bb3d.pbox[j][0]
            a2[idx1] = bb3d.pbox[j][1]
            a2[idx2] = bb3d.visibility[j]

            a3[idx0] = bb3d.qbox[j][0]
            a3[idx1] = bb3d.qbox[j][1]
            a3[idx2] = bb3d.qbox[j][2]
            
        a4[0] = bb3d.dim[0]
        a4[1] = bb3d.dim[1]
        a4[2] = bb3d.dim[2]
        
        c0 = 0
        c1 = m1
        c2 = m1 + m2
        c3 = m1 + m2 + m3
        c4 = m1 + m2 + m3 + m4

        array = np.zeros(c4)
        array[c0:c1] = a1
        array[c1:c2] = a2
        array[c2:c3] = a3
        array[c3:c4] = a4
        
        return array
    
    @classmethod
    def from_array(cls, array):
        
        m1 = cls.m1
        m2 = cls.m2
        m3 = cls.m3
        m4 = cls.m4
        
        c0 = 0
        c1 = m1
        c2 = m1 + m2
        c3 = m1 + m2 + m3
        c4 = m1 + m2 + m3 + m4

        a1 = array[c0:c1]
        a2 = array[c1:c2]
        a3 = array[c2:c3]
        a4 = array[c3:c4]
    
        # CAUTION: "situation" and "cases" are integers. All others are
        # numpy arrays of floating point numbers.
        situation = round(a1[0])
        cases = round(a1[1])
        
        vx = a1[2:4]
        vy = a1[4:6]
        vz = a1[6:8]
        
        pbox = np.zeros((8, 2))
        visibility = np.zeros(8)
        
        qbox = np.zeros((8, 3))
        dim = np.zeros(3)
        
        for j in range(8):
            
            idx0 = 3 * j
            idx1 = 3 * j + 1
            idx2 = 3 * j + 2
            
            pbox[j, 0] = a2[idx0]
            pbox[j, 1] = a2[idx1]
            visibility[j] = a2[idx2]

            qbox[j, 0] = a3[idx0]
            qbox[j, 1] = a3[idx1]
            qbox[j, 2] = a3[idx2]
            
        dim[:] = a4[:]
  
        bb3d = BB3D(situation, cases, pbox, visibility, qbox, dim, vx, vy, vz)
        return bb3d

def line_by_two_points(p0, p1):
    '''
    Return the line passing the given two points.
    '''
    
    p0H = np.ones(3)
    p0H[0:2] = p0
    p1H = np.ones(3)
    p1H[0:2] = p1
    
    l = np.cross(p0H, p1H)
    
    
    return l

def line_intersect(l0, l1):
    '''
    Return the intersection point of the given two lines.
    
    CAUTION: It will raise divide-by-zero error if the two lines are in
    parallel. Caller needs to check the parallel line case.
    
    '''
    
    pH = np.cross(l0, l1)
    pH /= pH[2]
    
    return pH[0:2]

def distance_between_point_and_line(p, l):
    '''
    Return the distance between a point and a line.
    
    CAUTION: The line cannot be a line at infinity, 
    e.g., it cannot be (0, 0, 1)
    '''
    
    n = np.zeros(2)
    n[0] = l[0]
    n[1] = l[1]
    
    nn = np.linalg.norm(n)
    
    d = np.abs(np.dot(n, p) + l[2])
    
    d = d / nn
    
    return d

def distances_between_points_and_line(ps, l):
    '''
    Return the distance between a set of points and a line.
    
    Note that ps is an n-by-2 matrix.
    '''
    
    n = np.zeros(2)
    n[0] = l[0]
    n[1] = l[1]
    
    nn = np.linalg.norm(n)
    
    ds = np.matmul(ps, n.reshape((2, 1)))
    
    ds = np.abs(ds + l[2])
    
    ds = ds / nn
    
    return ds

def select_nearest_point(pc1, pc2, p):
    '''
    Select a point closest to the point p from point candidates 
    pc1 and pc2.
    '''
    
    d1 = np.linalg.norm(pc1 - p)
    d2 = np.linalg.norm(pc2 - p)

    if d1 < d2:
        return pc1
    else:
        return pc2

def select_farthest_point(pc1, pc2, p):
    '''
    Select a point furtherest away from the point p from point 
    candidates pc1 and pc2.
    '''
    
    d1 = np.linalg.norm(pc1 - p)
    d2 = np.linalg.norm(pc2 - p)

    if d1 > d2:
        return pc1
    else:
        return pc2



    
    
def tangent_line(contour_ps, vp):
    '''
    Compute the tangent line respect to a vanishing point 
    given the contour.
    
    CAUTION: contour_ps is an n-by-2 matrix
    
    CAUTION: The returned tangent line is ordered such that for the 
    vanishing points of x-axis and y-axis in the vehicle local frame, the 
    first returned line is at the bottom surface of the 3D bounding box.
    '''
    
    center = np.mean(contour_ps, axis=0)
    cps = contour_ps
    
    # Now the vanishing point is the origin.
    ps = cps.astype(vp.dtype) - vp
    
    # We make the line from vp to center as the x-axis, so that
    # the calculated theta angles are always continuous in [-pi, pi]
    center = center - vp
    
    vx = np.zeros(2)
    vx[:] = center
    vx = vx / np.linalg.norm(vx)
    vy = np.zeros(2)
    vy[0] = vx[1]
    vy[1] = -vx[0]
    
    R = np.zeros((2, 2))
    R[:, 0] = vx
    R[:, 1] = vy
    RT = R.T
    
    #rc = np.matmul(RT, center.T).T
    rps = np.matmul(RT, ps.T).T
    
    # Calculate the theta angles
    
    theta = np.arctan2(rps[:, 1], rps[:, 0])

    min_i = np.argmin(theta)
    max_i = np.argmax(theta)

    p_min = cps[min_i]
    p_max = cps[max_i]

    l_min = line_by_two_points(vp, p_min)
    l_max = line_by_two_points(vp, p_max)
    
    # order the tangent lines
    
    k = center[0]
    
    if k < 0:
        
        return max_i, min_i, p_max, p_min, l_max, l_min

    else:
    
        return min_i, max_i, p_min, p_max, l_min, l_max



    
def check_vps(vpx, vpy, vpz, p_z0, p_z1):
    '''
    Check the three vanishing points and determine the proper cases
    for 3D bounding box calculation.
    
    Note that we use vpz as the anchor.
    
    '''
    
    nzx = vpx - vpz
    nzy = vpy - vpz
    
    nz0 = p_z0 - vpz
    nz1 = p_z1 - vpz
    
    #print(nz0, nz1)
    
    nzx = nzx / np.linalg.norm(nzx)
    nzy = nzy / np.linalg.norm(nzy)
    nz0 = nz0 / np.linalg.norm(nz0)
    nz1 = nz1 / np.linalg.norm(nz1)
    
    cz = np.dot(nz0, nz1)
    
    cx0 = np.dot(nzx, nz0)
    cx1 = np.dot(nzx, nz1)
    
    cy0 = np.dot(nzy, nz0)
    cy1 = np.dot(nzy, nz1)
    
    #print(cz, cx0, cx1, cy0, cy1)
    
    
    if cx0 >= cz and cx1 >= cz:
        
        # case xvpp
        
        return 1
    
    if cy0 >= cz and cy1 >= cz:
        
        # case yvpp
        
        return 2

    return 0

def check_point_respect_to_vpz(p, vpz, p_z0, p_z1):
    
    nzp = p - vpz
    
    nz0 = p_z0 - vpz
    nz1 = p_z1 - vpz
    
    #print(nz0, nz1)
    
    nzp = nzp / np.linalg.norm(nzp)
    nz0 = nz0 / np.linalg.norm(nz0)
    nz1 = nz1 / np.linalg.norm(nz1)
    
    cz = np.dot(nz0, nz1)
    
    cp0 = np.dot(nzp, nz0)
    cp1 = np.dot(nzp, nz1)
    
    #print(cz, cx0, cx1, cy0, cy1)
    
    
    if cp0 >= cz and cp1 >= cz:
        
        # p is inside the cone formed by vpz, lz0, lz1
        
        return 1
    
    else:

        return 0
    

def bb3d_nbox_3vpp(lx0, lx1, ly0, ly1, lz0, lz1, vpx, vpy, vpz, p_z0, p_z1):
    '''
    Calculate the points on the corner of the 3D bounding box.
    
    p1 is the intersection of lx0 and ly0
    p2 is on lx0
    p3 is on ly0
    p4 is invisible

    p5 is above p1
    p6 is above p2
    p7 is above p3
    p8 is above p4
    
    '''
    
    d_th = 3
    alpha = 0.7
    
    p1 = line_intersect(lx0, ly0)
    p8 = line_intersect(lx1, ly1)

    p1_condition = check_point_respect_to_vpz(p1, vpz, p_z0, p_z1)
    d18 = np.linalg.norm(p8 - p1)

    # CAUTION: If p1 is outside the cone formed by vpz, lz0, lz1, 
    # this algorithm will fail!
    # CAUTION: If the cone formed by vpx, lx0, lx1 is too small,
    # or the cone formed by vpy, ly0, ly1, is too small, this
    # algorithm will also fail!
    if p1_condition == 0 or d18 < d_th:
        
        return None

    # CAUTION: In 3vpp situation, lx0, ly0, lz0, lz1 all have finite
    # intersection points! The cases where vpx or vpy is on lz0 or lz1
    # are handed as degenerated cases in the situations of xvpp and yvpp!

    # p23 is on lz0
    p23c1 = line_intersect(lx0, lz0)
    p23c2 = line_intersect(ly0, lz0)
 
    p23 = select_farthest_point(p23c1, p23c2, vpz)
    d23 = np.linalg.norm(p23c1 - p23c2)
 
    # p32 is on lz1
    p32c1 = line_intersect(lx0, lz1)
    p32c2 = line_intersect(ly0, lz1)
 
    p32 = select_farthest_point(p32c1, p32c2, vpz)
    d32 = np.linalg.norm(p32c1 - p32c2)
 
    if p23 is p23c1 and d23 > d_th:
        p2 = p23
        p3 = p32
    else:
        p2 = p32
        p3 = p23
     
     
    if p32 is p32c2 and d32 > d_th:
        p2 = p23
        p3 = p32
    else:
        p2 = p32
        p3 = p23

    # CAUTION: If the cone formed by vpz, lz0, lz1 is too small, 
    # this algorithm will also fail!
    if np.linalg.norm(p2 - p3) < d_th:
        
        return None

    lzp1 = line_by_two_points(p1, vpz)
    lzp2 = line_by_two_points(p2, vpz)
    lzp3 = line_by_two_points(p3, vpz)
    
    
    # CAUTION: There might be degenerated cases where p1 is too
    # close to lz0 or lz1!

    dp1_lz0 = distance_between_point_and_line(p1, lz0)
    dp1_lz1 = distance_between_point_and_line(p1, lz1)
    
    
    if dp1_lz0 < d_th and dp1_lz1 > d_th:
        
        # CAUTION: degenerated case, p1 is too close to lz0
        
        if p2 is p23:
            
            #print('3xpp1')
            # case 3xpp1, p1, p2, p5, p6 are almost colinear
            # and they are all close to lz0.
            
            lzp8 = line_by_two_points(p8, vpz)
            lxp3 = line_by_two_points(p3, vpx)
            
            p4 = line_intersect(lzp8, lxp3)
            
            lyp4 = line_by_two_points(p4, vpy)

            # recompute p2            
            p2 = line_intersect(lyp4, lx0)
            lzp2 = line_by_two_points(p2, vpz)
            
            # CAUTION: adjust p7 if the object is not very "boxy"!
            
            p7a = line_intersect(lzp3, lx1)
            p7b = line_intersect(lzp3, ly1)
            
            p7 = alpha * p7a + (1 - alpha) * p7b
            
            lyp7 = line_by_two_points(p7, vpy)
        
            p5 = line_intersect(lzp1, lyp7)

            lyp8 = line_by_two_points(p8, vpy)
            
            p6 = line_intersect(lyp8, lzp2)
            
            
        else:
        
            #print('3ypp1')
            # case 3ypp1, p1, p2, p5, p7 are almost colinear
            # and they are all close to lz0.
            
            lzp8 = line_by_two_points(p8, vpz)
            lyp2 = line_by_two_points(p2, vpy)
            
            p4 = line_intersect(lzp8, lyp2)
            
            lxp4 = line_by_two_points(p4, vpx)
            
            # recompute p3
            p3 = line_intersect(lxp4, ly0)
            lzp3 = line_by_two_points(p3, vpz)
            
            # CAUTION: adjust p6 if the object is not very "boxy"!
            
            p6a = line_intersect(lzp2, ly1)
            p6b = line_intersect(lzp2, lx1)
    
            p6 = alpha * p6a + (1 - alpha) * p6b        
    
            lxp6 = line_by_two_points(p6, vpx)
        
            p5 = line_intersect(lzp1, lxp6)
            
            lxp8 = line_by_two_points(p8, vpx)
            
            p7 = line_intersect(lxp8, lzp3)
            
        
    elif dp1_lz0 > d_th and dp1_lz1 < d_th:
        
        # CAUTION: degenerated case, p1 is too close to lz1
        
        if p2 is p32:
            
            #print('3xpp2')
            # case 3xpp2, p1, p2, p5, p6 are almost colinear
            # and they are all close to lz1.
            
            lzp8 = line_by_two_points(p8, vpz)
            lxp3 = line_by_two_points(p3, vpx)
            
            p4 = line_intersect(lzp8, lxp3)
            
            lyp4 = line_by_two_points(p4, vpy)
            
            p2 = line_intersect(lyp4, lx0)
            lzp2 = line_by_two_points(p2, vpz)

            # CAUTION: adjust p7 if the object is not very "boxy"!
            
            p7a = line_intersect(lzp3, lx1)
            p7b = line_intersect(lzp3, ly1)
            
            p7 = alpha * p7a + (1 - alpha) * p7b
            
            lyp7 = line_by_two_points(p7, vpy)
        
            p5 = line_intersect(lzp1, lyp7)

            lyp8 = line_by_two_points(p8, vpy)
            
            p6 = line_intersect(lyp8, lzp2)
            
        else:
        
            #print('3ypp2')
            # case 3ypp2, p1, p3, p5, p7 are almost colinear
            # and they are all close to lz1.
            
            lzp8 = line_by_two_points(p8, vpz)
            lyp2 = line_by_two_points(p2, vpy)
            
            p4 = line_intersect(lzp8, lyp2)
            
            lxp4 = line_by_two_points(p4, vpx)
            
            p3 = line_intersect(lxp4, ly0)
            lzp3 = line_by_two_points(p3, vpz)

            # CAUTION: adjust p6 in case the object is not very "boxy"!
            
            p6a = line_intersect(lzp2, ly1)
            p6b = line_intersect(lzp2, lx1)
    
            p6 = alpha * p6a + (1 - alpha) * p6b        
    
            lxp6 = line_by_two_points(p6, vpx)
        
            p5 = line_intersect(lzp1, lxp6)
            
            lxp8 = line_by_two_points(p8, vpx)
            
            p7 = line_intersect(lxp8, lzp3)

        
    elif dp1_lz0 > d_th and dp1_lz1 > d_th:
        
        # normal case
        
    
        lx4 = line_by_two_points(p2, vpy)
        ly4 = line_by_two_points(p3, vpx)
        
        p4 = line_intersect(lx4, ly4)

        p6 = line_intersect(lzp2, ly1)
        p7 = line_intersect(lzp3, lx1)
    
    
        lzp1 = line_by_two_points(p1, vpz)
        lxp6 = line_by_two_points(p6, vpx)
        lyp7 = line_by_two_points(p7, vpy)
    
        # CAUTION: adjust p5 in case the object is not very "boxy"!
        # This may happen if one of lx1 or ly1 does not "bound" the
        # shape very well.
    
        p65 = line_intersect(lzp1, lxp6)
        p75 = line_intersect(lzp1, lyp7)
    
        p5 = select_farthest_point(p65, p75, vpz)
        
        # fix p6 and p7
        
        lxp5 = line_by_two_points(p5, vpx)
        lyp5 = line_by_two_points(p5, vpy)
        
        p6 = line_intersect(lzp2, lxp5)
        p7 = line_intersect(lzp3, lyp5)
    
        lyp6 = line_by_two_points(p6, vpy)
        lxp7 = line_by_two_points(p7, vpx)
    
        # fix p8
    
        p8 = line_intersect(lyp6, lxp7)
        
        
    else:
        
        # The cone formed by vpz, lz0, lz1 is too small at the place of p1,
        # i.e., the distance from p1 to lz0 and the distance from p1 to lz1
        # are both less than one pixel. In this case, the algorithm fails.
        
        return None
    

    return p1, p2, p3, p4, p5, p6, p7, p8


def bb3d_nbox_xvpp(lx0, lx1, ly0, ly1, lz0, lz1, vpx, vpy, vpz):
    '''
    Calculate the four points on the bottom surface of 
    the 3D bounding box.
    
    p1 is on ly0, below p5
    p2 is invisible
    p3 is on ly0, below p7
    p4 is invisible
    
    p5 is the intersection between lx0 and lz0 or lz1, whichever passes p1
    p6 is the intersection between lx0 and ly1
    p7 is the intersection between lx1 and lz0 or lz1, whichever passes p3
    p8 is the intersection between lx1 and ly1
    
    '''

    alpha = 0.7

    # CAUTION: lx0 may be on the left or on the right, but ly0 is the one
    # closer to vpz (among ly0 and ly1). Actually lx0 determines p1.
     
    p1a = line_intersect(lx0, ly0)

    p13c1 = line_intersect(lz0, ly0)
    p13c2 = line_intersect(lz1, ly0)

    p1 = select_nearest_point(p13c1, p13c2, p1a)
    p3 = select_farthest_point(p13c1, p13c2, p1a)
    
    d13 = np.linalg.norm(p3 - p1)
    if d13 < 1:
        
        # The cone formed by vpz, lz0, lz1 is too small at the place of p1,
        # i.e., the distance from p1 to lz0 and the distance from p1 to lz1
        # are both less than one pixel. In this case, the algorithm fails.

        return None
    
    if p1 is p13c1:
        lzp1 = lz0
        lzp3 = lz1
    else:
        lzp1 = lz1
        lzp3 = lz0

    # CAUTION: Check degenerated case!!!
    # In degenerated cases, p6 may be on lzp1, or p8 may be on lzp3,
    # and hence, p5 and p7 cannot be derived by line intersection.

    p6 = line_intersect(ly1, lx0)
    p8 = line_intersect(ly1, lx1)

    d68 = np.linalg.norm(p6 - p8)
    d16 = np.linalg.norm(p1 - p6)
    d38 = np.linalg.norm(p3 - p8)
    if d68 < 1 or d16 < 1 or d38 < 1:
        
        # The cone formed by vpx, lx0, lx1 is too small, or the cone formed
        # by vpy, ly0, ly1 is too small, this algorithm will fail.

        return None
    
    dp6 = distance_between_point_and_line(p6, lzp1)
    dp8 = distance_between_point_and_line(p8, lzp3)

    if dp6 < 1 and dp8 > 1:
        
        # lx0 and lzp1 are the same line, i.e., p1, p5, p6 are colinear
        # CAUTION: adjust p7 if the object is not very "boxy"!
        
        p7a = line_intersect(lzp3, lx1)
        p7b = line_intersect(lzp3, ly1)
        
        p7 = alpha * p7a + (1 - alpha) * p7b
        
        lyp7 = line_by_two_points(p7, vpy)
    
        p5 = line_intersect(lzp1, lyp7)
        
        lxp3 = line_by_two_points(p3, vpx)
        lzp8 = line_by_two_points(p8, vpz)

        p4 = line_intersect(lxp3, lzp8)
        
        lyp4 = line_by_two_points(p4, vpy)
        lxp1 = line_by_two_points(p1, vpx)
        
        p2 = line_intersect(lyp4, lxp1)
        
    
    elif dp6 > 1 and dp8 < 1:
        
        # lx1 and lzp3 are the same line, i.e., p3, p7, p8 are colinear
        
        p5a = line_intersect(lzp1, lx0)
        p5b = line_intersect(lzp1, ly1)

        p5 = alpha * p5a + (1 - alpha) * p5b        
        
        lyp5 = line_by_two_points(p5, vpy)
        
        p7 = line_intersect(lzp3, lyp5)
        
        lxp1 = line_by_two_points(p1, vpx)
        lzp6 = line_by_two_points(p6, vpz)

        p2 = line_intersect(lxp1, lzp6)
        
        lyp2 = line_by_two_points(p2, vpy)
        lxp3 = line_by_two_points(p3, vpx)
        
        p4 = line_intersect(lyp2, lxp3)
        
    else:
    #elif dp6 > 1 and dp8 > 1:
    
        # CAUTION: This is perspective projection, so it is not possible
        # that both dp6 and dp8 are nearly zero.

        # CAUTION: (p5, p7, vpy) must be colinear.
        
        p65a = line_intersect(lzp1, lx0)
        p65b = line_intersect(lzp1, ly1)

        p65 = alpha * p65a + (1 - alpha) * p65b        

        p87a = line_intersect(lzp3, lx1)
        p87b = line_intersect(lzp3, ly1)
        
        p87 = alpha * p87a + (1 - alpha) * p87b
    
        lyp65 = line_by_two_points(p65, vpy)
        lyp87 = line_by_two_points(p87, vpy)
        
        p75 = line_intersect(lzp1, lyp87)
        p57 = line_intersect(lzp3, lyp65)
        
        p5 = select_farthest_point(p65, p75, vpz)
        p7 = select_farthest_point(p87, p57, vpz)

        # CAUTION: Recomupte p6 and p8 here. 
        # (p5, p6, vpx) must be colinear.
        # (p7, p8, vpx) must be colinear.
        
        lxp5 = line_by_two_points(p5, vpx)
        lxp7 = line_by_two_points(p7, vpx)
        
        p6 = line_intersect(lxp5, ly1)
        p8 = line_intersect(lxp7, ly1)
        
        lxp1 = line_by_two_points(p1, vpx)
        lxp3 = line_by_two_points(p3, vpx)
    
        lzp6 = line_by_two_points(p6, vpz)
        lzp8 = line_by_two_points(p8, vpz)
    
        p2 = line_intersect(lzp6, lxp1)
        p4 = line_intersect(lzp8, lxp3)

#         else:
#         
#             # The perspective is lost, i.e., p1, p5, p6 are almost colinear,
#             # p3, p7, p8 are also almost colinear.
#             # CAUTION: It is assumed that this will not happen!
#             
#             return None
        
        
    
    return p1, p2, p3, p4, p5, p6, p7, p8
    
def bb3d_nbox_yvpp(lx0, lx1, ly0, ly1, lz0, lz1, vpx, vpy, vpz):
    '''
    Calculate the four points on the bottom surface of 
    the 3D bounding box.
    
    p1 is on lx0, below p5
    p2 is on lx0, below p6
    p3 is invisible
    p4 is invisible
    
    p5 is the intersection between ly0 and lz0 or lz1, whichever passes p1
    p6 is the intersection between ly1 and lz0 or lz1, whichever passes p2
    p7 is the intersection between ly0 and lx1
    p8 is the intersection between ly1 and lx1
    
    '''
    
    alpha = 0.7
    
    p7 = line_intersect(lx1, ly0)
    p8 = line_intersect(lx1, ly1)
    
    
    p1a = line_intersect(ly0, lx0)

    p12c1 = line_intersect(lz0, lx0)
    p12c2 = line_intersect(lz1, lx0)

    p1 = select_nearest_point(p12c1, p12c2, p1a)
    p2 = select_farthest_point(p12c1, p12c2, p1a)

    d12 = np.linalg.norm(p2 - p1)
    if d12 < 1:
        
        # The cone formed by vpz, lz0, lz1 is too small at the place of p1,
        # i.e., the distance from p1 to lz0 and the distance from p1 to lz1
        # are both less than one pixel. In this case, the algorithm fails.
        
        return None
    
    if p1 is p12c1:
        lzp1 = lz0
        lzp2 = lz1
    else:
        lzp1 = lz1
        lzp2 = lz0


    # CAUTION: Check degenerated case!!!
    # In degenerated cases, p7 may be on lzp1, or p8 may be on lzp3,
    # and hence, p5 and p7 cannot be derived by line intersection.

    dp7 = distance_between_point_and_line(p7, lzp1)
    dp8 = distance_between_point_and_line(p8, lzp2)
    
    d78 = np.linalg.norm(p7 - p8)
    d17 = np.linalg.norm(p1 - p7)
    d28 = np.linalg.norm(p2 - p8)
    if d78 < 1 or d17 < 1 or d28 < 1:
        
        # The cone formed by vpy, ly0, ly1 is too small, or the cone formed
        # by vpx, lx0, lx1 is too small, this algorithm will fail.

        return None
    
    if dp7 < 1 and dp8 > 1:
        
        # ly0 and lzp1 is the same line, i.e., p1, p5, p7 are colinear
        
        p6a = line_intersect(lzp2, ly1)
        p6b = line_intersect(lzp2, lx1)

        p6 = alpha * p6a + (1 - alpha) * p6b        

        lxp6 = line_by_two_points(p6, vpx)
    
        p5 = line_intersect(lzp1, lxp6)
        
        lyp2 = line_by_two_points(p2, vpy)
        lzp8 = line_by_two_points(p8, vpz)

        p4 = line_intersect(lyp2, lzp8)
        
        lxp4 = line_by_two_points(p4, vpx)
        lyp1 = line_by_two_points(p1, vpy)
        
        p3 = line_intersect(lxp4, lyp1)

    elif dp7 > 1 and dp8 < 1:

        # ly1 and lzp2 is the same line, i.e., p2, p6, p8 are colinear
        
        p5a = line_intersect(lzp1, ly0)
        p5b = line_intersect(lzp1, lx1)

        p5 = alpha * p5a + (1 - alpha) * p5b        

        lxp5 = line_by_two_points(p5, vpx)
        
        p6 = line_intersect(lzp2, lxp5)

        lyp1 = line_by_two_points(p1, vpy)
        lzp7 = line_by_two_points(p7, vpz)

        p3 = line_intersect(lyp1, lzp7)
        
        lxp3 = line_by_two_points(p3, vpx)
        lyp2 = line_by_two_points(p2, vpy)
        
        p4 = line_intersect(lxp3, lyp2)

    else:
    #elif dp7 > 1 and dp8 < 1:
        
        # CAUTION: This is perspective projection, so it is not possible
        # that both dp7 and dp8 are nearly zero.
        
        # CAUTION: (p5, p6, vpx) must be colinear.
        
        p75a = line_intersect(lzp1, ly0)
        p75b = line_intersect(lzp1, lx1)

        p75 = alpha * p75a + (1 - alpha) * p75b        

        p86a = line_intersect(lzp2, ly1)
        p86b = line_intersect(lzp2, lx1)

        p86 = alpha * p86a + (1 - alpha) * p86b        
        
        lxp75 = line_by_two_points(p75, vpx)
        lxp86 = line_by_two_points(p86, vpx)
        
        p65 = line_intersect(lzp1, lxp86)
        p56 = line_intersect(lzp2, lxp75)
        
        p5 = select_farthest_point(p75, p65, vpz)
        p6 = select_farthest_point(p86, p56, vpz)

        # CAUTION: Recomupte p7 and p8 here. 
        # (p5, p7, vpy) must be colinear.
        # (p6, p8, vpy) must be colinear.
        
        lyp5 = line_by_two_points(p5, vpy)
        lyp6 = line_by_two_points(p6, vpy)
        
        p7 = line_intersect(lyp5, lx1)
        p8 = line_intersect(lyp6, lx1)
        
        lyp1 = line_by_two_points(p1, vpy)
        lyp2 = line_by_two_points(p2, vpy)
        
        lzp7 = line_by_two_points(p7, vpz)
        lzp8 = line_by_two_points(p8, vpz)
        
        p3 = line_intersect(lyp1, lzp7)
        p4 = line_intersect(lyp2, lzp8)
    
#         else:
#             
#             # The perspective is lost, i.e., p1, p5, p7 are almost colinear,
#             # p2, p6, p8 are also almost colinear.
#             # CAUTION: It is assumed that this will not happen!
#             
#             return None
        


    return p1, p2, p3, p4, p5, p6, p7, p8
    

def bb3d_pbox_3vpp(nbox, vx, vy, vz):
    '''
    Calculate the oriented 3D bounding box.
    
    '''
    
    p1, p2, p3, p4, p5, p6, p7, p8 = nbox
    
    dir_x = np.dot(p2 - p1, vx)
    dir_y = np.dot(p3 - p1, vy)
    dir_z = np.dot(p5 - p1, vz)
    
    # Assign point ABCD
    if dir_x < 0 and dir_y < 0:
        
        # case 0
        cases = 0
        
        pa = p1
        pb = p3
        pc = p2
        pd = p4

        pe = p5
        pf = p7
        pg = p6
        ph = p8
    
        ps_invisible = (pd,)
        flags = (1, 1, 1, 0, 1, 1, 1, 1)
        
    elif dir_x < 0 and dir_y > 0:
        
        # case 1
        cases = 1
        
        pa = p3
        pb = p1
        pc = p4
        pd = p2

        pe = p7
        pf = p5
        pg = p8
        ph = p6
        
        ps_invisible = (pc,)
        flags = (1, 1, 0, 1, 1, 1, 1, 1)

    elif dir_x > 0 and dir_y < 0:
        
        # case 2
        cases = 2
        
        pa = p2
        pb = p4
        pc = p1
        pd = p3
        
        pe = p6
        pf = p8
        pg = p5
        ph = p7
    
        ps_invisible = (pb,)
        flags = (1, 0, 1, 1, 1, 1, 1, 1)


    else: #dir_x > 0 and dir_y > 0:

        # case 3
        cases = 3
        
        pa = p4   
        pb = p2
        pc = p3
        pd = p1
        
        pe = p8
        pf = p6
        pg = p7
        ph = p5
    
        ps_invisible = (pa,)
        flags = (0, 1, 1, 1, 1, 1, 1, 1)
        
    pbox = np.asarray((pa, pb, pc, pd, pe, pf, pg, ph))
    visibility = np.asarray(flags)
        
    return cases, pbox, visibility


def bb3d_pbox_xvpp(nbox, vx, vy, vz):
    '''
    Calculate the oriented 3D bounding box.
    
    '''
    
    p1, p2, p3, p4, p5, p6, p7, p8 = nbox
    
    dir_x = np.dot(p2 - p1, vx)
    dir_y = np.dot(p3 - p1, vy)
    
    
    if dir_x > 0 and dir_y > 0:
        
        cases = 0
        
        pa = p4
        pb = p2
        pc = p3
        pd = p1
        
        pe = p8
        pf = p6
        pg = p7
        ph = p5
        
        ps_invisible = (pa, pb)
        flags = (0, 0, 1, 1, 1, 1, 1, 1)
    
    elif dir_x > 0 and dir_y < 0:
        
        cases = 0
        
        pa = p2
        pb = p4
        pc = p1
        pd = p3
        
        pe = p6
        pf = p8
        pg = p5
        ph = p7
        
        ps_invisible = (pa, pb)
        flags = (0, 0, 1, 1, 1, 1, 1, 1)
    
    elif dir_x < 0 and dir_y < 0:
        
        cases = 1
        
        pa = p1
        pb = p3
        pc = p2
        pd = p4
        
        pe = p5
        pf = p7
        pg = p6
        ph = p8
        
        ps_invisible = (pc, pd)
        flags = (1, 1, 0, 0, 1, 1, 1, 1)
    
    else: # dir_x < 0 and dir_y > 0:
        
        cases = 1
        
        pa = p3
        pb = p1
        pc = p4
        pd = p2
        
        pe = p7
        pf = p5
        pg = p8
        ph = p6
        
        ps_invisible = (pc, pd)
        flags = (1, 1, 0, 0, 1, 1, 1, 1)
    
    pbox = np.asarray((pa, pb, pc, pd, pe, pf, pg, ph))
    visibility = np.asarray(flags)

    return cases, pbox, visibility

    
def bb3d_pbox_yvpp(nbox, vx, vy, vz):
    '''
    Calculate the oriented 3D bounding box.
    
    '''
    
    p1, p2, p3, p4, p5, p6, p7, p8 = nbox
    
    dir_x = np.dot(p2 - p1, vx)
    dir_y = np.dot(p3 - p1, vy)
    
    
    if dir_x > 0 and dir_y > 0:
        
        cases = 0
        
        pa = p4
        pb = p2
        pc = p3
        pd = p1
        
        pe = p8
        pf = p6
        pg = p7
        ph = p5
        
        ps_invisible = (pa, pc)
        flags = (0, 1, 0, 1, 1, 1, 1, 1)
    
    elif dir_x > 0 and dir_y < 0:
        
        cases = 1
        
        pa = p2
        pb = p4
        pc = p1
        pd = p3
        
        pe = p6
        pf = p8
        pg = p5
        ph = p7
        
        ps_invisible = (pb, pd)
        flags = (1, 0, 1, 0, 1, 1, 1, 1)
    
    elif dir_x < 0 and dir_y < 0:
        
        cases = 1
        
        pa = p1
        pb = p3
        pc = p2
        pd = p4
        
        pe = p5
        pf = p7
        pg = p6
        ph = p8
        
        ps_invisible = (pb, pd)
        flags = (1, 0, 1, 0, 1, 1, 1, 1)
        
    else:
        
        cases = 0
        
        pa = p3
        pb = p1
        pc = p4
        pd = p2
        
        pe = p7
        pf = p5
        pg = p8
        ph = p6
        
        ps_invisible = (pa, pc)
        flags = (0, 1, 0, 1, 1, 1, 1, 1)

    pbox = np.asarray((pa, pb, pc, pd, pe, pf, pg, ph))
    visibility = np.asarray(flags)

    return cases, pbox, visibility




def bb3d_height(P, p, qx, qy):
    '''
    Calculate the height of the 3D bounding box in the world frame
    using camera projection.
    
    P (uppercase) is the camera projection matrix.
    p (lowercase) is the pixel of the projected 3D point q
    qx and qy are the x and y coordinates of q in the world frame.
    Note that qz is unknown, i.e., the height we need to calculate.
    
    '''
    
    u = p[0]
    v = p[1]
    
    c1 = P[1, 0] * qx + P[1, 1] * qy + P[1, 3]
    c2 = P[0, 0] * qx + P[0, 1] * qy + P[0, 3]

    height = (v * c2 - u * c1 ) / (u * P[1, 2] - v * P[0, 2])
    
    return height

def bb3d_qbox(camera_model, pbox):
    '''
    Calculate the length, width, and height of the 3D bounding box
    in the world frame.
    
    "camera_model" is the camera model
    
    P is the camera projection matrix.
    H_inv is the inverse of the homography.
    pbox is the oriented 3D bounding box
    
    '''
    
    P = camera_model.P
    
    pa = pbox[0]
    pb = pbox[1]
    pc = pbox[2]
    pd = pbox[3]
    pe = pbox[4]
    pf = pbox[5]
    pg = pbox[6]
    ph = pbox[7]
    
    qa = camera_model.transform_point_image_to_ground(pa)
    qb = camera_model.transform_point_image_to_ground(pb)
    qc = camera_model.transform_point_image_to_ground(pc)
    qd = camera_model.transform_point_image_to_ground(pd)

    

    l1 = np.linalg.norm(qa - qc)
    l2 = np.linalg.norm(qb - qd)
    
    w1 = np.linalg.norm(qa - qb)
    w2 = np.linalg.norm(qc - qd)
    
    h1 = bb3d_height(P, pe, qa[0], qa[1])
    h2 = bb3d_height(P, pf, qb[0], qb[1])
    h3 = bb3d_height(P, pg, qc[0], qc[1])
    h4 = bb3d_height(P, ph, qd[0], qd[1])
    
    qe = qa.copy()
    qe[2] = h1
    qf = qb.copy()
    qf[2] = h2
    qg = qc.copy()
    qg[2] = h3
    qh = qd.copy()
    qh[2] = h4
    
    
    
    l = (l1 + l2) / 2
    w = (w1 + w2) / 2
    h = (h1 + h2 + h3 + h4) / 4
    
    qbox = np.asarray((qa, qb, qc, qd, qe, qf, qg, qh))
    dim = np.asarray((l, w, h))

    
    return dim, qbox


def update_lx0_ly0(contour_ps, lx0, ly0, lz0, lz1, 
        vpx, vpy, vpz, vx, vy, vz):
    
    
    n_limit = 20
    d_limit = 10
    
    cxz_th1 = 0.2
    cxz_th2 = 0.8
    alpha_default = 0.5
    alpha_th = 0.1
    
    n = contour_ps.shape[0]
    
    if n < n_limit:
        return lx0, ly0
    
    p1 = line_intersect(lx0, ly0)
    
    p2c1 = line_intersect(lx0, lz0)
    p2c2 = line_intersect(lx0, lz1)
    p2 = select_nearest_point(p2c1, p2c2, vpx)
    
    d12 = np.linalg.norm(p2 - p1)
    if d12 < d_limit:
        return lx0, ly0

    v1x = p1 - vpx
    v1x = v1x / np.linalg.norm(v1x)
    v1z = p1 - vpz
    v1z = v1z / np.linalg.norm(v1z)
    cxz = np.dot(v1x, v1z)
    cxz = abs(cxz)
    
    
    if cxz > cxz_th2:
        return lx0, ly0
    elif cxz > cxz_th1 and cxz < cxz_th2:
        alpha = alpha_th
    else:
        alpha = alpha_default
    
    
    v1y = p1 - vpy
    v1y = v1y / np.linalg.norm(v1y)
    cyz = np.dot(v1y, v1z)
    
    
    pk = (p1 + p2) / 2
    
    lzk = line_by_two_points(vpz, pk)
    
    contour_psk = contour_ps.copy()
    
    contour_psk[:, 0] -= int(pk[0])
    contour_psk[:, 1] -= int(pk[1])
    
    ds = np.linalg.norm(contour_psk, axis=1)
    
    i = np.argmin(ds)
    dck = ds[i]
    pck = contour_ps[i]
    
    #cv2.circle(img_3dbb, (pck[0], pck[1]), 2, (255, 255, 0), 2)
    
    #print('pck, dck = ', pck, dck)
    
    lxk = line_by_two_points(vpx, pck)
    p1k = line_intersect(lxk, ly0)
    
    lz1k = line_by_two_points(vpz, p1k)
    p1_new = line_intersect(lz1k, lx0)
    
    p1_update = alpha * p1 + (1 - alpha) * p1_new
    
    ly0 = line_by_two_points(vpy, p1_update)
    
    
    
    return lx0, ly0
   
    
    
def bb3d_perspective(pbox, cm, contour, vpx, vpy, vpz, vx, vy, vz):
    ''' 
    Compute the 3D bounding nbox using the tangent line method, 
    given the contour and a perspective camera.
    
    CAUTION: The vanishing points must be finite points!
    
    '''
    
 
    # CAUTION: If the contour is obtained from cv2.findContour(), 
    # it has the shape of (n, 1, 2)
    
    n = contour.shape[0]
    contour_ps = contour.reshape((n, 2))
    
    
    
    x0_i, x1_i, p_x0, p_x1, lx0, lx1 = tangent_line(contour_ps, vpx)
    y0_i, y1_i, p_y0, p_y1, ly0, ly1 = tangent_line(contour_ps, vpy)
    z0_i, z1_i, p_z0, p_z1, lz0, lz1 = tangent_line(contour_ps, vpz)


    situation = check_vps(vpx, vpy, vpz, p_z0, p_z1)

    #print('situation: %d' % situation)
    '''
    if situation == 1:
    
        nbox = bb3d_nbox_xvpp(
            lx0, lx1, ly0, ly1, lz0, lz1, vpx, vpy, vpz)
        
        if not nbox:
            return None
        
        cases, pbox, visibility = bb3d_pbox_xvpp(nbox, vx, vy, vz)
    
    
    elif situation == 2:
        
        nbox = bb3d_nbox_yvpp(
            lx0, lx1, ly0, ly1, lz0, lz1, vpx, vpy, vpz)
        
        if not nbox:
            return None
                    
        cases, pbox, visibility = bb3d_pbox_yvpp(nbox, vx, vy, vz)
        
    
    else:

        lx0, ly0 = update_lx0_ly0(contour_ps, lx0, ly0, lz0, lz1, 
                                  vpx, vpy, vpz, vx, vy, vz)
        
 
        nbox = bb3d_nbox_3vpp(
            lx0, lx1, ly0, ly1, lz0, lz1, vpx, vpy, vpz, p_z0, p_z1)
        
        if not nbox:
            return None
        
        cases, pbox, visibility = bb3d_pbox_3vpp(nbox, vx, vy, vz)
    '''    

    cases = 0
    visibility = np.array((1, 1, 1, 1, 1, 1, 1, 1))
    dim, qbox = bb3d_qbox(cm, pbox)
    
    bb3d = BB3D(situation, cases, pbox, visibility, 
                qbox, dim, vx, vy, vz)
    
    return bb3d



# def bb3d_weak_perspective(cm, contour, vx, vy, vz):
#     '''
#     Compute the 3D bounding box using the tangent line method, 
#     given the contour and a weak perspective camera.
#     
# 
#     '''
#     
#     
#     
#     
#     
#     
#     
#     pass







def test_boundingbox():
    
    id = 2
    
    direction = 0
    postfix = ('_%d' % 0)
    
    prefixes = ['westbound', 'eastbound', 'northbound', 'southbound']
    prefix = prefixes[direction]
    
    folder = './car_data'
    
    
    car_fn = folder + '/' + prefix + '_vehicle_%d.png' % id
    mask_fn = folder + '/boundingbox/' + prefix + '_mask_%d.png' % id
    
    
    img_car = cv2.imread(car_fn)
    img_mask = cv2.imread(mask_fn)
    img_3dbb = cv2.imread(car_fn)


#     cv2.imshow('original', img_car)
#     cv2.imshow('mask', img_mask)
#     
#     cv2.waitKey(0)

    
    img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
    
    
    
    cm = Camera2DGroundModel()
    cm.load_calib_para(folder + '/calibration/', prefix)
    
    
    # load vehicle orientation
    
    orientation_fn = folder + '/boundingbox/orientation_%d.csv' % id
    opp = np.loadtxt(orientation_fn, delimiter=',')
    
    p0 = opp[0]
    p1 = opp[1]
    p2 = opp[2]
    p3 = opp[3]
    p4 = opp[4]
    p5 = opp[5]
    
    vx = p1 - p0
    vy = p3 - p2
    vz = p5 - p4
    
    
    
#     vx = vx / np.linalg.norm(vx)
#     vy = vy / np.linalg.norm(vy)
#     vz = vz / np.linalg.norm(vz)
    
    print(vx)
    print(vy)
    print(vz)
    
    print()
    
    ovx = cm.direction_uv2xyz(p0, p1)
    ovy = cm.direction_uv2xyz(p2, p3)
    #ovx = np.asarray((1, 0, 0))
    #ovy = np.asarray((0, 1, 0))
    ovz = np.asarray((0, 0, 1))
    
    if np.linalg.norm(vx) > np.linalg.norm(vy):
        
        ovy = np.cross(ovz, ovx)
        
    else:
        
        ovx = np.cross(ovy, ovz)
    
    print(ovx)
    print(ovy)
    print(ovz)
    
    print()
    
    # vanishing points
    
    vpx, vpy, vpz = cm.calculate_vps(ovx, ovy, ovz)
    
    print(vpx)
    print(vpy)
    print(vpz)
    
    print()
    
    
    
    # mask and contour
    
    contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #print(len(contours), len(contours[1]))
    
    cv2.drawContours(img_3dbb, contours, 1, (255,255,255), 3)
    
    contour = contours[1]
    
    #print(contour.shape)
    
    # 3D bounding box
    
    
    bb3d = bb3d_perspective(cm, contour, vpx, vpy, vpz, vx, vy, vz, img_3dbb)
    
    pbox = bb3d.pbox
    points_invisible = bb3d.points_invisible
    
    vis = FrameVis()
    
    vis.draw_bb3d(img_3dbb, pbox, points_invisible)
    
    
    #cv2.imshow('original', img_car)
    cv2.imshow('mask', img_mask)
    
    cv2.imshow('tracking_3d', img_3dbb)
    cv2.waitKey(0)
    
    
    pass


if __name__ == '__main__':
    
    
    test_boundingbox()
    pass