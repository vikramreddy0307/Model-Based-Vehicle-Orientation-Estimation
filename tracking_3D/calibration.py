'''
Created on May 9, 2020

@author: duolu
'''

import math
import numpy as np
import cv2
import pywgs84 as pywgs84

from visualization import FrameVis
from visualization import Visualizer
from visualization import MapVis

class GridPlane(object):
    
    def __init__(self, M, N, x_min, x_max, y_min, y_max, x_unit, y_unit):
    
        lin_x = np.linspace(x_min, x_max, M) * x_unit
        lin_y = np.linspace(y_min, y_max, N) * y_unit
        grid_x, grid_y = np.meshgrid(lin_x, lin_y)
    
        grid_xy = np.zeros((2, M * N))
        grid_xy[0, :] = grid_x.flatten()
        grid_xy[1, :] = grid_y.flatten()

        grid_xyz = np.zeros((3, M * N))
        grid_xyz[0:2, :] = grid_xy[0:2, :]
    
        self.M = M
        self.N = N
    
        self.grid_x = grid_x
        self.grid_y = grid_y
        
        self.grid_xy = grid_xy
        self.grid_xyz = grid_xyz





class MapModel(object):
    '''
    The map model and transformations between the XYZ frame and the map frame.

    NOTE: The XYZ frame is usually defined arbitrarily but its z-axis
    always points to the up.

    NOTE: The map reference frame follows the East-South-Down convention.

    '''

    def __init__(self, local_map_image, global_map_image):
        '''
        Initialize the model by loading two map images.
        '''
    
        self.local_map_image = local_map_image
        self.global_map_image = global_map_image

    def get_local_map_size(self):
        '''
        Get the size of the local map, i.e., (width, height) in pixels.
        '''

        shape = self.local_map_image.shape

        return shape[1], shape[0]

    def get_global_map_size(self):
        '''
        Get the size of the global map, i.e., (width, height) in pixels.
        '''

        shape = self.global_map_image.shape

        return shape[1], shape[0]


    def calibrate_xyz_to_local_map(self, pp_esd):
        '''
        Calculate transformation between the local xyz frame and the
        map local frame.

        NOTE: "pp_esd" is a 6-by-2 matrix. Both columns are point
        correspondences of (xm, ym, zm, x, y, z), where (xm, ym, zm)
        is a point on the map and (x, y, z) is the a point in 3D space,
        i.e., local XYZ reference frame. Here the first row is the origin
        of the XYZ reference frame, and the second row is "p1", a point
        on the negative half of the x-axis of the XYZ reference frame.

        '''
        
        # CAUTION: A point is a 3-by-1 matrix.
        origin_xyz_in_lmap = pp_esd[0:3, 0].reshape((3, 1))
        p1_xyz_in_lmap = pp_esd[0:3, 1].reshape((3, 1))
        
        vector_x_in_lmap = origin_xyz_in_lmap - p1_xyz_in_lmap
        
        length_on_lmap = np.linalg.norm(vector_x_in_lmap)
        length_in_xyz = np.linalg.norm(pp_esd[3:6, 1])
        
        #print(length_on_lmap, length_in_xyz)
        
        # CAUTION: in xyz, z is up, while in lmap, d is down
        
        nx = vector_x_in_lmap / np.linalg.norm(vector_x_in_lmap)
        ny = np.asarray((nx[1], -nx[0], 0)).reshape((3, 1))
        nz = np.asarray((0, 0, -1)).reshape((3, 1))
        
        # scale conversion

        scale_xyz2lmap = length_on_lmap / length_in_xyz
        scale_lmap2xyz = length_in_xyz / length_on_lmap

        # from xyz to lmap
        
        R = np.zeros((3, 3))
        R[:, 0:1] = nx
        R[:, 1:2] = ny
        R[:, 2:3] = nz
        
        R_xyz2lmap = R * scale_xyz2lmap
        t_xyz2lmap = origin_xyz_in_lmap
        
        # from lmap to xyz
        
        R_inv = R.T
        
        R_lmap2xyz = R_inv * scale_lmap2xyz
        t_lmap2xyz = -np.matmul(R_inv, t_xyz2lmap) * scale_lmap2xyz
        
        # from xyz to esd
        # NOTE: There is only a rotation between xyz and esd, since the
        # anchor point of esd frame is the origin of xyz.

        R_xyz2esd = R
        t_xyz2esd = np.zeros(3)

        # from esd to xyz

        R_esd2xyz = R_inv
        t_esd2xyz = np.zeros(3)

        # keep calibration results

        self.R_xyz2lmap = R_xyz2lmap
        self.t_xyz2lmap = t_xyz2lmap
        
        self.R_lmap2xyz = R_lmap2xyz
        self.t_lmap2xyz = t_lmap2xyz

        self.R_xyz2esd = R_xyz2esd
        self.t_xyz2esd = t_xyz2esd
        
        self.R_esd2xyz = R_esd2xyz
        self.t_esd2xyz = t_esd2xyz


    def calibrate_local_map_to_wgs(self, pp_wgs):
        '''
        Calculate transformation between the local ESD frame to WGS84 frame.
        '''

        ref_point_lmap = pp_wgs[0:3]
        ref_point_llh = pp_wgs[3:6]

        # The XYZ reference frame and the ESD reference frame have the
        # same origin and this point is the reference point of the ESD
        # tangent plane in WGS84.

        self.ref_point_llh = ref_point_llh

    
    def calibrate_local_map_to_global_map(self, pp_global):
        '''
        Calculate the transformation between the map lmap frame and
        the global map.
        
        "pp_global" is a 6-by-n matrix. Each column is a point correspondence
        of (x_local, y_local, z_local, x_global, y_global, z_global).

        CAUTION: It is assumed that there is only a translation and
        scaling between the two frames in the map plane.
        
        Note that we use linear least square to solve the scale and
        the translation.
        '''
        
        n = pp_global.shape[1]
        
        A = np.zeros((2 * n, 3))
        b = np.zeros((2 * n, 1))
        
        for i in range(n):
            
            idx0 = 2 * i
            idx1 = 2 * i + 1
            
            A[idx0, 0] = pp_global[0, i]
            A[idx0, 1] = 1
            A[idx0, 2] = 0
            A[idx1, 0] = pp_global[1, i]
            A[idx1, 1] = 0
            A[idx1, 2] = 1
    
            b[idx0] = pp_global[3, i]
            b[idx1] = pp_global[4, i]
    
    
        pinv = np.linalg.inv(np.matmul(A.T, A))
        x = np.matmul(pinv, np.matmul(A.T, b))
    
        x = x.flatten()
        s = x[0]
        tx = x[1]
        ty = x[2]
        
        
        R_lmap2gmap = np.zeros((3, 3))
        R_lmap2gmap[0, 0] = s
        R_lmap2gmap[1, 1] = s
        R_lmap2gmap[2, 2] = s
        t_lmap2gmap = np.zeros((3, 1))
        t_lmap2gmap[0, 0] = tx
        t_lmap2gmap[1, 0] = ty
        t_lmap2gmap[2, 0] = 0
    
        R_gmap2lmap = R_lmap2gmap.T
        t_gmap2lmap = -np.matmul(R_lmap2gmap.T, t_lmap2gmap)

        self.R_lmap2gmap = R_lmap2gmap
        self.t_lmap2gmap = t_lmap2gmap
        
        self.R_gmap2lmap = R_gmap2lmap
        self.t_gmap2lmap = t_gmap2lmap
    
        return R_lmap2gmap, t_lmap2gmap, R_gmap2lmap, t_gmap2lmap
    

    def calibrate_local_map_to_hd_map(self, pp_hd):

        ps_lmap = pp_hd[0:3, :]
        ps_hd = pp_hd[3:6, :]

        ps_xyz = self.transform_points_lmap_to_xyz(ps_lmap)

        center_xyz = np.mean(ps_xyz, axis=1).reshape((3, 1))
        center_hd = np.mean(ps_hd, axis=1).reshape((3, 1))

        cs_xyz = ps_xyz - center_xyz
        cs_hd = ps_hd - center_hd

        norm_xyz = np.linalg.norm(cs_xyz, axis=0)
        norm_hd = np.linalg.norm(cs_hd, axis=0)

        scales_xyz2hd = np.divide(norm_hd, norm_xyz)
        scale_xyz2hd = np.mean(scales_xyz2hd)
        scale_hd2xyz = 1 / scale_xyz2hd

        cs_hd_scaled = cs_hd * scale_hd2xyz

        co = np.matmul(cs_xyz, cs_hd_scaled.T)

        u, s, vt = np.linalg.svd(co)

        R = np.matmul(vt.T, u.T)

        if np.linalg.det(R) < 0:

            vt[2, :] *= -1
            R = np.matmul(vt.T, u.T)
        
        # CAUTION: This R is from xyz to scaled hd.

        R_xyz2hdmap = R * scale_xyz2hd
        t_xyz2hdmap = center_hd - np.matmul(R_xyz2hdmap, center_xyz)

        R_inv = R.T

        R_hdmap2xyz = R_inv * scale_hd2xyz
        t_hdmap2xyz = -np.matmul(R_inv, t_xyz2hdmap) * scale_hd2xyz

        self.R_xyz2hdmap = R_xyz2hdmap
        self.t_xyz2hdmap = t_xyz2hdmap

        self.R_hdmap2xyz = R_hdmap2xyz
        self.t_hdmap2xyz = t_hdmap2xyz

        return R_xyz2hdmap, t_xyz2hdmap, R_hdmap2xyz, t_hdmap2xyz


    def save_para_csv(self, folder, prefix, para_name):

        para = getattr(self, para_name)
        fn = folder + '/' + prefix + '_' + para_name + '.csv'
        np.savetxt(fn, para, delimiter=',', fmt='%.6f')
        
    def load_para_csv(self, folder, prefix, para_name):

        fn = folder + '/' + prefix + '_' + para_name + '.csv'
        para = np.loadtxt(fn, delimiter=',')
        setattr(self, para_name, para)

    def save_map_para(self, folder, prefix):
    
        self.save_para_csv(folder, prefix, 'R_xyz2lmap')
        self.save_para_csv(folder, prefix, 't_xyz2lmap')
        self.save_para_csv(folder, prefix, 'R_lmap2xyz')
        self.save_para_csv(folder, prefix, 't_lmap2xyz')

        self.save_para_csv(folder, prefix, 'R_xyz2esd')
        self.save_para_csv(folder, prefix, 't_xyz2esd')
        self.save_para_csv(folder, prefix, 'R_esd2xyz')
        self.save_para_csv(folder, prefix, 't_esd2xyz')

        self.save_para_csv(folder, prefix, 'ref_point_llh')

        self.save_para_csv(folder, prefix, 'R_lmap2gmap')
        self.save_para_csv(folder, prefix, 't_lmap2gmap')
        self.save_para_csv(folder, prefix, 'R_gmap2lmap')
        self.save_para_csv(folder, prefix, 't_gmap2lmap')

        self.save_para_csv(folder, prefix, 'R_xyz2hdmap')
        self.save_para_csv(folder, prefix, 't_xyz2hdmap')
        self.save_para_csv(folder, prefix, 'R_hdmap2xyz')
        self.save_para_csv(folder, prefix, 't_hdmap2xyz')

    
    def load_map_para(self, folder, prefix):

        self.load_para_csv(folder, prefix, 'R_xyz2lmap')
        self.load_para_csv(folder, prefix, 't_xyz2lmap')
        self.load_para_csv(folder, prefix, 'R_lmap2xyz')
        self.load_para_csv(folder, prefix, 't_lmap2xyz')

        self.load_para_csv(folder, prefix, 'R_xyz2esd')
        self.load_para_csv(folder, prefix, 't_xyz2esd')
        self.load_para_csv(folder, prefix, 'R_esd2xyz')
        self.load_para_csv(folder, prefix, 't_esd2xyz')

        self.load_para_csv(folder, prefix, 'ref_point_llh')

        self.load_para_csv(folder, prefix, 'R_lmap2gmap')
        self.load_para_csv(folder, prefix, 't_lmap2gmap')
        self.load_para_csv(folder, prefix, 'R_gmap2lmap')
        self.load_para_csv(folder, prefix, 't_gmap2lmap')

        self.load_para_csv(folder, prefix, 'R_xyz2hdmap')
        self.load_para_csv(folder, prefix, 't_xyz2hdmap')
        self.load_para_csv(folder, prefix, 'R_hdmap2xyz')
        self.load_para_csv(folder, prefix, 't_hdmap2xyz')

        self.t_xyz2lmap = self.t_xyz2lmap.reshape((3, 1))
        self.t_lmap2xyz = self.t_lmap2xyz.reshape((3, 1))

        self.t_xyz2esd = self.t_xyz2esd.reshape((3, 1))
        self.t_esd2xyz = self.t_esd2xyz.reshape((3, 1))

        self.t_lmap2gmap = self.t_lmap2gmap.reshape((3, 1))
        self.t_gmap2lmap = self.t_gmap2lmap.reshape((3, 1))

        self.t_xyz2hdmap = self.t_xyz2hdmap.reshape((3, 1))
        self.t_hdmap2xyz = self.t_hdmap2xyz.reshape((3, 1))


    def transform_points_lmap_to_xyz(self, ps_lmap):

        ps_xyz = np.matmul(self.R_lmap2xyz, ps_lmap) + self.t_lmap2xyz

        return ps_xyz
    
    def transform_points_xyz_to_lmap(self, ps_xyz):

        ps_lmap = np.matmul(self.R_xyz2lmap, ps_xyz) + self.t_xyz2lmap

        return ps_lmap

    def transform_points_esd_to_xyz(self, ps_esd):

        ps_xyz = np.matmul(self.R_esd2xyz, ps_esd) + self.t_esd2xyz

        return ps_xyz
    
    def transform_points_xyz_to_esd(self, ps_xyz):
        # print(self.R_xyz2esd.shape, ps_xyz.shape, self.t_xyz2esd.shape)
        ps_esd = np.matmul(self.R_xyz2esd, ps_xyz) + self.t_xyz2esd

        return ps_esd


    def transform_points_esd_to_llh(self, ps_esd):

        n = ps_esd.shape[1]

        ps_llh = np.zeros((3, n))

        for i in range(n):

            esd = tuple(ps_esd[:, i])
            llh = pywgs84.esd_to_llh(*esd, self.ref_point_llh)
            ps_llh[:, i] = np.asarray(llh)

        return ps_llh

    def transform_points_llh_to_esd(self, ps_llh):

        n = ps_llh.shape[1]

        ps_esd = np.zeros((3, n))

        for i in range(n):

            llh = tuple(ps_llh[:, i])
            esd = pywgs84.llh_to_esd(*llh, self.ref_point_llh)
            ps_esd[:, i] = np.asarray(esd).flatten()

        return ps_esd


    
    def transform_points_lmap_to_gmap(self, ps_lmap):

        ps_gmap = np.matmul(self.R_lmap2gmap, ps_lmap) + self.t_lmap2gmap

        return ps_gmap
    
    def transform_points_gmap_to_lmap(self, ps_global):

        ps_lmap = np.matmul(self.R_gmap2lmap, ps_global) + self.t_gmap2lmap

        return ps_lmap

    def transform_points_xyz_to_hdmap(self, ps_xyz):

        ps_hdmap = np.matmul(self.R_xyz2hdmap, ps_xyz) + self.t_xyz2hdmap

        return ps_hdmap
    
    def transform_points_hdmap_to_xyz(self, ps_hdmap):

        ps_xyz = np.matmul(self.R_hdmap2xyz, ps_hdmap) + self.t_hdmap2xyz

        return ps_xyz


class Camera2DGroundModel(object):
    '''
    The pinhole camera and flat ground model.
    '''

    def __init__(self):
        '''
        Initialize the pinhole camera and flat ground model.
        '''
        pass


    def calib_image_to_map_directly(self, points_lmap, points_uv):
        '''
        Compute a homography that transforms between map points and
        image points directly.
        
        NOTE: p_uv = G * p_xy, p_xy = G_inv * p_uv
        '''
        
        points_es = points_lmap[0:2, :]

        G, mask = cv2.findHomography(points_es.T, points_uv.T)
        #G_inv, mask = cv2.findHomography(points_uv.T, points_es.T)
        G_inv = np.linalg.inv(G)
        
        self.G = G
        self.G_inv = G_inv
        
        return G, G_inv
        
    def warp_image_to_map(self, image, map_width, map_height):
        '''
        Warp an image using the image to map homography.
        '''

        img_warp = cv2.warpPerspective(image, self.G_inv, (map_width, map_height))

        return img_warp
    
    def calib_ground(self, points_xyz, points_uv):
        '''
        Compute the homography.
        
        points_xyz is a 3-by-n matrix
        points_uv is a 2-by-n matrix
        
        Note that the z-coordinates of points_xyz are ignored.
        
        '''
        
        points_xy = points_xyz[0:2, :]

        H, mask = cv2.findHomography(points_xy.T, points_uv.T)
        #H_inv, mask = cv2.findHomography(points_uv.T, points_xy.T) 
        H_inv = np.linalg.inv(H)

        self.H = H
        self.H_inv = H_inv
        
        return H, H_inv
    
    
    def calib_camera(self, points_xyz, points_uv, width, height, f_init=2000):
        '''
        Compute the camera intrinsics and extrinsics.
        
        points_xyz is a 3-by-n matrix
        points_uv is a 2-by-n matrix
        
        NOTE: This assumes the camera has no distortion, square pixel,
        no pixel skew, and the principal point is at the center of the frame.
        Hence, the only unknown variable in the intrinsic matrix is the fs.
        
        NOTE: The input f_init is needed. Otherwise in some cases the
        calibration will fail if limited number of point correspondences
        are provided.
        
        '''
        
        nr_points = points_xyz.shape[1]
        
        ps3d = np.zeros((1, nr_points, 3), dtype=np.float32)
        ps2d = np.zeros((1, nr_points, 2), dtype=np.float32)
        
        ps3d[0] = points_xyz.T
        ps2d[0] = points_uv.T
    
        K_init = np.identity(3)
        K_init[0, 0] = f_init
        K_init[1, 1] = f_init
        K_init[0, 2] = width / 2
        K_init[1, 2] = height / 2
        dist_co_zero = np.zeros((1, 5))
    
        # CAUTION: OpenCV expects the 3D points as n-by-3 matrix.
        # CAUTION: OpenCV expects the 2D points as n-by-2 matrix.
    
        # CAUTION: cv2.CALIB_USE_INTRINSIC_GUESS is needed, 
        # otherwise it may cause spurious results.
        ret, K, dist_co, rvecs, tvecs = cv2.calibrateCamera( \
            ps3d, ps2d, (width, height), K_init, dist_co_zero,
            flags=cv2.CALIB_FIX_PRINCIPAL_POINT
            | cv2.CALIB_USE_INTRINSIC_GUESS
            | cv2.CALIB_FIX_ASPECT_RATIO
            | cv2.CALIB_ZERO_TANGENT_DIST
            | cv2.CALIB_FIX_K1
            | cv2.CALIB_FIX_K2
            | cv2.CALIB_FIX_K3)
        
        # CAUTION: rvecs is 1-by-3, tvecs is 1-by-3.

        R, _jacobian = cv2.Rodrigues(rvecs[0])
        t = tvecs[0]
        
        P = np.zeros((3, 4))
        P[:, 0:3] = R
        P[:, 3:4] = t
        
        P = np.matmul(K, P)
    
        # recompute the homography
    
        H = np.zeros((3, 3))
        H[:, 0:2] = R[:, 0:2]
        H[:, 2:3] = t
        H = np.matmul(K, H)
        H_inv = np.linalg.inv(H)
        
        
        self.P = P
        self.R = R
        self.t = t
        self.K = K
        
        self.H = H
        self.H_inv = H_inv
        
        
        return P, R, t, K, H, H_inv
    
    def get_camera_pose(self):
        '''
        Obtain the camera pose (R', t') in the world xyz frame.
        '''
        
        R_prime = self.R.T
        t_prime = -np.matmul(self.R.T, self.t)
        
        return R_prime, t_prime
    
    def save_para_csv(self, folder, prefix, para_name):

        para = getattr(self, para_name)
        fn = folder + '/' + prefix + '_' + para_name + '.csv'
        np.savetxt(fn, para, delimiter=',', fmt='%.6f')
        
    def load_para_csv(self, folder, prefix, para_name):

        fn = folder + '/' + prefix + '_' + para_name + '.csv'
        para = np.loadtxt(fn, delimiter=',')
        setattr(self, para_name, para)
    
    def save_calib_para(self, folder, prefix):

        self.save_para_csv(folder, prefix, 'P')
        self.save_para_csv(folder, prefix, 'R')
        self.save_para_csv(folder, prefix, 't')
        self.save_para_csv(folder, prefix, 'K')

        self.save_para_csv(folder, prefix, 'H')
        self.save_para_csv(folder, prefix, 'H_inv')
        self.save_para_csv(folder, prefix, 'G')
        self.save_para_csv(folder, prefix, 'G_inv')

    def load_calib_para(self, folder, prefix):
        
        self.load_para_csv(folder, prefix, 'P')
        self.load_para_csv(folder, prefix, 'R')
        self.load_para_csv(folder, prefix, 't')
        self.load_para_csv(folder, prefix, 'K')

        self.load_para_csv(folder, prefix, 'H')
        self.load_para_csv(folder, prefix, 'H_inv')
        self.load_para_csv(folder, prefix, 'G')
        self.load_para_csv(folder, prefix, 'G_inv')


    def transform_point_image_to_lmap(self, p):
        '''
        Transform a 2D image point (u, v) to a 3D ground point (x, y, 0), 
        using the inverse of the homography.
        
        CAUTION: (u, v).T = np.matmul(H, (x, y, 0).T), the homography is
        defined in this way, i.e., from the world frame to the image frame. 
        '''
        
        pH = np.ones((3, 1))
        
        pH[0:2, :] = p.reshape((2, 1))
        
        qH = np.matmul(self.G_inv, pH)
        
        if qH[2, 0] != 0:
        
            qH /= qH[2, 0]
            qH[2, 0] = 0
        
            return qH.flatten()
        
        else:
            
            return None
    
    
    def transform_point_lmap_to_image(self, q):
        '''
        Transform a 3D ground point (x, y, 0) to a 2D image point (u, v),
        using the homography.
        
        CAUTION: (u, v).T = np.matmul(H, (x, y, 0).T), the homography is
        defined in this way, i.e., from the world frame to the image frame. 
        '''
        
        qH = np.ones((3, 1))
        
        qH[0:2, :] = q.reshape((3, 1))[0:2, :]
        
        pH = np.matmul(self.G, qH)
        
        if pH[2, 0] != 0:
            
            pH /= pH[2, 0]
            
            return pH[0:2].flatten()
        
        else:
            
            return None


    def transform_points_image_to_lmap(self, ps):
        '''
        Transform a 3D ground point (x, y, 0) to a 2D image point (u, v),
        using the homography.

        The input ps is an 2-by-n matrix.

        '''
        
        n = ps.shape[1]
        
        pH = np.ones((3, n))
        
        pH[0:2, :] = ps
        
        qH = np.matmul(self.G_inv, pH)
        
        qH[0, :] = np.divide(qH[0, :], qH[2, :])
        qH[1, :] = np.divide(qH[1, :], qH[2, :])
        
        qH[2, 0] = 0
        
        return qH
        

    def transform_points_lmap_to_image(self, qs):
        '''
        Transform a 3D ground point (x, y, 0) to a 2D image point (u, v),
        using the homography.

        The input ps is an 3-by-n matrix.

        '''
        
        n = qs.shape[1]

        qH = np.ones((3, n))
        
        qH[0:2, :] = qs[0:2, :]
        
        pH = np.matmul(self.G, qH)
        
        #print(pH.shape)

        pH[0, :] = np.divide(pH[0, :], pH[2, :])
        pH[1, :] = np.divide(pH[1, :], pH[2, :])

        return pH[0:2, :]


    def transform_point_image_to_ground(self, p):
        '''
        Transform a 2D image point (u, v) to a 3D ground point (x, y, 0), 
        using the inverse of the homography.
        
        CAUTION: (u, v).T = np.matmul(H, (x, y, 0).T), the homography is
        defined in this way, i.e., from the world frame to the image frame. 
        '''
        
        pH = np.ones((3, 1))
        
        pH[0:2, :] = p.reshape((2, 1))
        
        qH = np.matmul(self.H_inv, pH)
        
        if qH[2, 0] != 0:
        
            qH /= qH[2, 0]
            qH[2, 0] = 0
        
            return qH.flatten()
        
        else:
            
            return None
    


    def transform_point_ground_to_image(self, q):
        '''
        Transform a 3D ground point (x, y, 0) to a 2D image point (u, v),
        using the homography.
        
        CAUTION: (u, v).T = np.matmul(H, (x, y, 0).T), the homography is
        defined in this way, i.e., from the world frame to the image frame. 
        '''
        
        qH = np.ones((3, 1))
        
        qH[0:2, :] = q.reshape((3, 1))[0:2, :]
        
        pH = np.matmul(self.H, qH)
        
        if pH[2, 0] != 0:
            
            pH /= pH[2, 0]
            
            return pH[0:2].flatten()
        
        else:
            
            return None

    def transform_points_image_to_ground(self, ps):
        '''
        Transform a 3D ground point (x, y, 0) to a 2D image point (u, v),
        using the homography.

        The input ps is an 2-by-n matrix.

        '''
        
        n = ps.shape[1]
        
        pH = np.ones((3, n))
        
        pH[0:2, :] = ps
        
        qH = np.matmul(self.H_inv, pH)
        
        qH[0, :] = np.divide(qH[0, :], qH[2, :])
        qH[1, :] = np.divide(qH[1, :], qH[2, :])
        
        qH[2, 0] = 0
        
        return qH
        

    def transform_points_ground_to_image(self, qs):
        '''
        Transform a 3D ground point (x, y, 0) to a 2D image point (u, v),
        using the homography.

        The input ps is an 3-by-n matrix.

        '''
        
        n = qs.shape[1]

        qH = np.ones((3, n))
        
        qH[0:2, :] = qs[0:2, :]
        
        pH = np.matmul(self.H, qH)
        
        #print(pH.shape)

        pH[0, :] = np.divide(pH[0, :], pH[2, :])
        pH[1, :] = np.divide(pH[1, :], pH[2, :])

        return pH[0:2, :]

    def project_point(self, q):
        '''
        Project a single point in the world reference frame to the image.
        
        The input is a vector of three element, i.e., (x, y, z)
        '''
        
        qH = np.ones((4, 1))
        qH[0:3, :] = q.reshape((3, 1))
        
        pH = np.matmul(self.P, qH)
        
        pH[0, :] = np.divide(pH[0, :], pH[2, :])
        pH[1, :] = np.divide(pH[1, :], pH[2, :])
        
        return pH[0:2, :].flatten()

    def project_points(self, qs):
        '''
        Project a single point in the world reference frame to the image.
        
        The input is a vector of three element, i.e., (x, y, z)
        '''
        
        n = qs.shape[1]
        
        qsH = np.ones((4, n))
        qsH[0:3, :] = qs.reshape((3, n))
        
        psH = np.matmul(self.P, qsH)
        
        psH[0, :] = np.divide(psH[0, :], psH[2, :])
        psH[1, :] = np.divide(psH[1, :], psH[2, :])
        
        return psH[0:2, :]

    
    def calculate_one_vp(self, v):
        '''
        Given the camera projection matrix and a vector indicating 
        a direction,calculate the vanishing point in that direction.
        
        CAUTION: It is possible that the vanishing point is not a finite point!
        In this case, it will calculate an approximated vanishing point!
        
        '''
        
        vH = np.zeros((4, 1))
        vH[0:3, :] = v.reshape((3, 1))
        
        vpH = np.matmul(self.P, vH)
        
        vpH = vpH.flatten()
        
        if vpH[2] >= 0 and vpH[2] < 1e-6:
            vpH[2] = 1e-6

        if vpH[2] < 0 and vpH[2] > -1e-6:
            vpH[2] = -1e-6
        
        vpH /= vpH[2]
        
        return vpH[0:2]
    
    def calculate_vps(self, vx, vy, vz):
        '''
        Calculate the vanishing point in the world frame given the xyz-axes 
        of a local frame.
        
        The local frame xyz are pointing to the front-left-down of a vehicle.
        This function calculates the vanishing point respect to a specific
        vehicle pose.
        '''
        
        vpx = self.calculate_one_vp(vx)
        vpy = self.calculate_one_vp(vy)
        vpz = self.calculate_one_vp(vz)
        
        
        return vpx, vpy, vpz








def test_joint_calib_2d(camera_id):
    
    
    folder = '../avacar_data'
    
    
    prefixes = ['westbound', 'eastbound', 'northbound', 'southbound', 'osburn1004']
    prefix = prefixes[camera_id]

    calibration_folder = folder + '/calibration_2d/' + prefix
    
    bg_fn = calibration_folder + '/'  + prefix + '_bg_ref.png'
    map_global_fn = calibration_folder + '/' + prefix + '_map_global.png'
    map_local_fn = calibration_folder + '/' + prefix + '_map_local.png'

    # Load the background frame.

    img_bg = cv2.imread(bg_fn)

    frame_width = img_bg.shape[1]
    frame_height = img_bg.shape[0]

    # Load the point correspondences for calibration.

    pp_fn = calibration_folder + '/' + prefix + "_map_local_to_uv.csv"
    pp_esd_fn = calibration_folder + '/' + prefix + "_map_local_to_xyz.csv"
    pp_wgs_fn = calibration_folder + '/' + prefix + "_map_local_to_wgs.csv"
    pp_global_fn = calibration_folder + '/' + prefix + "_map_local_to_global.csv"
    pp_hdmap_fn = calibration_folder + '/' + prefix + "_map_local_to_map_hd.csv"
    
    pp = np.loadtxt(pp_fn, delimiter=',')
    pp_esd = np.loadtxt(pp_esd_fn, delimiter=',')
    pp_wgs = np.loadtxt(pp_wgs_fn, delimiter=',')
    pp_global = np.loadtxt(pp_global_fn, delimiter=',')
    pp_hd = np.loadtxt(pp_hdmap_fn, delimiter=',')
    
    # NOTE: All points are 3-by-n or 2-by-n stored in files. They are 
    # transposed here to n-by-3 or n-by-2.

    # NOTE: "pp_wgs" has only one row.
    
    pp_calib_lmap = pp[:, 0:3].T
    pp_calib_uv = pp[:, 3:5].T
    pp_esd = pp_esd.T
    pp_global = pp_global.T
    pp_hd = pp_hd.T

    
    #draw anchor points and calibration points
     
    # map_local_vis = cv2.imread(map_local_fn)

    # map_local_vis.astype(np.float)
    # map_local_vis //= 2
    # map_local_vis += 127
    # map_local_vis.astype(np.uint8)
  
    # vis1 = Visualizer()
  
    # vis1.draw_pp(map_local_vis, img_bg, pp_calib_lmap, pp_calib_uv, 0)
 
    # cv2.imshow('map_local', map_local_vis)
    # cv2.imshow('img_bg', img_bg)
     
    # cv2.waitKey(0)
    
    
    map_global = cv2.imread(map_global_fn)
    map_local = cv2.imread(map_local_fn)
    mm = MapModel(map_local, map_global)
    
    mm.calibrate_xyz_to_local_map(pp_esd)
    mm.calibrate_local_map_to_wgs(pp_wgs)
    mm.calibrate_local_map_to_global_map(pp_global)
    mm.calibrate_local_map_to_hd_map(pp_hd)
    
    mm.save_map_para(calibration_folder, prefix)
    mm.load_map_para(calibration_folder, prefix)
    
    pp_calib_xyz = mm.transform_points_lmap_to_xyz(pp_calib_lmap)

    print('---------- map xyz ----------')
    print(pp_calib_xyz.T)
    print()

    

    pp_calib_esd = mm.transform_points_xyz_to_esd(pp_calib_xyz)
    pp_calib_llh = mm.transform_points_esd_to_llh(pp_calib_esd)

    print('---------- map esd ----------')
    xx_esd = pp_calib_esd.T
    n = xx_esd.shape[0]
    for i in range(n):
        print(i, '%.3f, %.3f, %.3f' % (xx_esd[i, 0], xx_esd[i, 1], xx_esd[i, 2]))
    print()

    print('---------- map llh ----------')
    xx_llh = pp_calib_llh.T
    n = xx_llh.shape[0]
    for i in range(n):
        print(i, xx_llh[i])
    print()

    print('---------- map hd ----------')

    ps_lmap = pp_hd[0:3, :]
    ps_hd = pp_hd[3:6, :]
    ps_xyz = mm.transform_points_lmap_to_xyz(ps_lmap)
    ps_hd_tt = mm.transform_points_xyz_to_hdmap(ps_xyz)

    print(ps_hd.T)
    print(ps_hd_tt.T)
    print()

    


    cm = Camera2DGroundModel()

    # Direct calibration between the image and the map.
    
    G, G_inv = cm.calib_image_to_map_directly(pp_calib_lmap, pp_calib_uv)

    
    # Calibration the homography of the ground
    
    H, H_inv = cm.calib_ground(pp_calib_xyz, pp_calib_uv)

    print('---------- ground-frame homography ----------')
    print(H)
    print(H_inv)
    print()
    
    # Generate grid
    
    # This is for the intersection
    gp = GridPlane(M=11, N=9, x_min=-8, x_max=2, y_min=-4, y_max=4, 
                     x_unit=10, y_unit=3.7)
    
    # This for osburn0723.
    #gp = GridPlane(M=11, N=9, x_min=-8, x_max=2, y_min=-1, y_max=7, 
    #                 x_unit=10, y_unit=3.7)
    
    grid_xyz = gp.grid_xyz
    grid_xy = gp.grid_xy
    grid_uv = cm.transform_points_ground_to_image(gp.grid_xy)


    #print(grid_xy.shape)
    #print(grid_uv.shape)


    
    cm.calib_camera(grid_xyz, grid_uv, frame_width, frame_height)
    
    
    cm.save_calib_para(calibration_folder, prefix)
    cm.load_calib_para(calibration_folder, prefix)
    
    print(cm.H)
    print(cm.H_inv)
    print()

    _R, _t = cm.get_camera_pose()

    print(_t)
    
    mvis = MapVis()
    

    fov_roi = (0, 110, frame_width, frame_height)
    mvis.draw_camera_FOV_on_map(map_local, cm, mm, fov_roi)
    mvis.draw_camera_FOV_on_global_map(map_global, cm, mm, fov_roi)
    
    
    fvis = FrameVis()
    
    img_grid = cv2.imread(bg_fn)
    fvis.draw_ground_grid_on_image(img_grid, gp.M, gp.N, grid_uv)
    
    img_xyz_axes = cv2.imread(bg_fn)
    fvis.draw_ground_xyz_axes_on_image(img_xyz_axes, cm.P, gp.M, gp.N, grid_xyz)
    
    img_bg = cv2.imread(bg_fn)
    img_warp = cm.warp_image_to_map(img_bg, map_local.shape[1], map_local.shape[0])


    map_local_resize = (map_local.shape[1] // 2, map_local.shape[0] // 2)
    map_local = cv2.resize(map_local, map_local_resize)

    img_warp_resize = (img_warp.shape[1] // 2, img_warp.shape[0] // 2)
    img_warp = cv2.resize(img_warp, img_warp_resize)
    
    cv2.imshow('map_global', map_global)
    cv2.imshow('map_local', map_local)
    cv2.imshow('img_warp', img_warp)
    
    
    cv2.imshow('img_grid', img_grid)
    cv2.imshow('img_xyz_axes', img_xyz_axes)
    
    cv2.waitKey(0)


    
def test_global_map():
    
    
    frame_width = 1280
    frame_height = 720
    
    folder = './car_data/calibration'
    
    prefixes = ['westbound', 'eastbound', 'northbound', 'southbound', 'osburn0723']

    prefix = prefixes[0]

    colors = [(255, 255, 0), (255, 0, 255), (0, 255, 255), 
              (255, 0, 0), (0, 255, 0), (0, 0, 255)]
    
    map_global_fn = folder + '/' + prefix + '_map_global.png'
    map_global = cv2.imread(map_global_fn)
    
    mvis = MapVis()
    fov_roi = (0, 120, frame_width, frame_height)

    
    for direction in range(4):
        
        prefix = prefixes[direction]
        
        map_fn = folder + '/' + prefix + '_map.png'
        
        mm = MapModel()
        mm.load_map_para(folder, prefix)
    
    
        cm = Camera2DGroundModel()
        cm.load_calib_para(folder, prefix)
        
        color = colors[direction]
        
        mvis.draw_camera_FOV_on_global_map(map_global, cm, mm, fov_roi, color=color)
    
    cv2.imshow('map_global', map_global)
    
    cv2.waitKey(0)
    







if __name__ == '__main__':
    
    #print(cv2.__version__)
    
    test_joint_calib_2d(3)

    # for camera_id in range(0, 4):
    #     test_joint_calib_2d(camera_id)
    
    
    #test_global_map()
    
    pass










