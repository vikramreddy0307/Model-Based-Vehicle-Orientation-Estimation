'''
Created on May 13, 2020

@author: duolu
'''

import math
import os
import random
import sys
import warnings
from collections import Counter

import cv2
import numpy as np
from tqdm import tqdm

from background import CameraShakeRectifier
from background import KLTOpticalFlowTracker
from boundingbox3D import BB3D
from boundingbox3D import bb3d_perspective
from calibration import Camera2DGroundModel
from calibration import MapModel
from detection import DetectorTorchvisionMaskRCNN
from detection import DetectorWithSavedResults
from detection import calculate_box_size
from detection import calculate_mask_overlap
from detection import check_box_bottom_occlusion
from detection import check_box_on_mask_bg
from detection import check_box_overlap
from detection import compute_center_from_box
from detection import compute_contour_from_mask
from visualization import FrameVis
from visualization import MapVis


class Instance(object):
    '''
    Instance of object detected by the detector on each frame.
    '''

    # For serialization / deserialization.
    # id, motion, heading, flags, box, center, label, score, 3d flag
    m1 = 3 + 5 + 3 + 2 + 4 + 2 + 1 + 1 + 1
    m = m1 + BB3D.m

    def __init__(self, detection_id, box, center, mask, contour, label, score):

        self.detection_id = detection_id
        self.tracking_id = -1

        # The following attributes are obtained directly from the detector.

        self.box = box
        self.center = center
        self.mask = mask
        self.contour = contour
        self.label = label
        self.score = score

        # The following attributes are obtained from the tracker.

        # Corners detected on this frame.
        # NOTE: corners_to_next is needed for inferring missing instances.
        self.corners_to_prev = []
        self.corresponding_corners_on_prev = []

        self.corners_to_next = []
        self.corresponding_corners_on_next = []

        # Sparse optical flow vectors associated with the corners, from
        # this frame to the previous frame.
        # NOTE: sofs_to_next is needed for inferring missing instances.
        self.sofs_to_prev = []
        self.sofs_to_next = []

        # Inliers of corners and sparse optical flow vectors computed
        # in motion estimation.
        self.corners_to_prev_inliers = np.zeros((0, 2))
        self.sofs_to_prev_inliers = np.zeros((0, 2))

        # Motion and heading.
        # NOTE: Momentum saturate at a certain threshold.

        self.motion = np.zeros(2)
        self.momentum = np.zeros(2)
        self.motion_flag = 0

        self.heading = np.zeros(2)
        self.heading_flag = 0

        # Instance flag: 0 - bad, 1 - good (detected), >= 2 - good (inferred)
        self.instance_flag = 1

        # Occlusion flag:
        #     0 - no occlusion,
        #     1 - maybe occlusion (i.e., bounding box overlaps with others)
        self.occlusion_flag = 0

        # 3D bounding box and 3D states.

        self.bb3d = None

        self.states_3d = None

        # Instances are arranged in doubly-linked list by the tracker.
        self.prev = None
        self.next = None
        self.list_idx = 0

        # Link to the vehicle object.
        self.vehicle = None

    def to_array(self):

        m1 = self.m1
        m2 = BB3D.m

        a1 = np.zeros(m1)

        a1[0] = self.tracking_id
        a1[1] = self.detection_id
        a1[2] = 0

        a1[3:5] = self.motion
        a1[5:7] = self.momentum
        a1[7] = self.motion_flag

        a1[8:10] = self.heading
        a1[10] = self.heading_flag

        a1[11] = self.instance_flag
        a1[12] = self.occlusion_flag

        a1[13:17] = self.box
        a1[17:19] = self.center
        a1[19] = self.label
        a1[20] = self.score

        bb3d = self.bb3d

        if bb3d is None:

            a1[21] = 0  # The 3d flag is 0.

            a2 = np.zeros(m2)

        else:

            a1[21] = 1  # The 3d flag is 1.

            a2 = BB3D.to_array(bb3d)

        return np.concatenate((a1, a2))

    def from_array(self, array):

        m1 = self.m1
        m2 = BB3D.m

        c0 = 0
        c1 = m1
        c2 = m1 + m2

        a1 = array[c0:c1]
        a2 = array[c1:c2]

        self.tracking_id = round(a1[0])
        self.detection_id = round(a1[1])

        self.motion = a1[3:5]
        self.momentum = a1[5:7]
        self.motion_flag = round(a1[7])

        self.heading = a1[8:10]
        self.heading_flag = round(a1[10])

        self.instance_flag = round(a1[11])
        self.occlusion_flag = round(a1[12])

        self.box = a1[13:17]
        self.center = a1[17:19]
        self.label = a1[19]
        self.score = a1[20]

        if round(a1[21]) > 0:
            self.bb3d = BB3D.from_array(a2)

    def get_mask_local(self):

        box = self.box
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        mask_local = self.mask[y1:y2, x1:x2]

        return mask_local

    def set_mask_local(self, mask_local, frame_width, frame_height):

        box = self.box
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        self.mask = np.zeros((frame_height, frame_width), dtype=bool)
        self.mask[y1:y2, x1:x2] = mask_local

    def merge_with(self, instance):
        '''
        Merge instance onto this instance.
        '''

        box = merge_boxes(self.box, instance.box)
        center = compute_center_from_box(box)
        mask = np.logical_or(self.mask, instance.mask)
        contour = compute_contour_from_mask(mask)

        self.box = box
        self.center = center
        self.mask = mask
        self.contour = contour

        # labels and scores are not changed.

    def check_mask_box_ratio(self):

        mask_area = np.sum(self.mask)

        box = self.box
        p1x = box[0]
        p1y = box[1]
        p2x = box[2]
        p2y = box[3]

        w = p2x - p1x
        h = p2y - p1y

        box_area = w * h

        return mask_area / box_area


class Instance3DStates(object):
    '''
    3D states of a detected instance computed using the perspective geometry.
    
    Note that this is only an intermediate data structure used between the
    tracking and the state estimation.
    '''

    def __init__(self, instance):

        self.instance = instance
        self.states_flag = 0
        # Point states.
        # CAUTION: heading, pos and vel are 3D vectors!

        self.heading = None

        self.pos = None
        self.pos_flag = -1
        self.pos_backup = None
        self.pos_backup_flag = -1

        self.pos_uc = None
        self.pos_backup_uc = None

        self.vel = None
        self.vel_n = 0

        self.vel_uc = None

        self.dim = None
        self.dim_uc = None

    def get_center(self):

        dim = self.dim
        heading = self.heading

        pos_flag = self.pos_flag
        pos = self.pos

        nv = heading / np.linalg.norm(heading)
        nt = np.asarray((-nv[1], nv[0], 0))

        if pos_flag == 0:

            # pos is (qa + qb) / 2
            center = pos - nv * dim[0] / 2

        elif pos_flag == 1:

            # pos is (qa + qc) / 2
            center = pos - nt * dim[1] / 2

        elif pos_flag == 2:

            # pos is (qb + qd) / 2
            center = pos + nt * dim[1] / 2

        elif pos_flag == 3:

            # pos is (qc + qd) / 2
            center = pos + nv * dim[0] / 2

        else:

            # BUG
            print('BUG in get_center(), pos_flag = %d' % pos_flag)
            pass

        return center

    def get_volumn(self):

        return self.dim[0] * self.dim[1] * self.dim[2]


class Vehicle(object):
    '''
    A vehicle object tracked over a sequence of frames.
    
    "vid" means the vehicle ID, assigned by the tracker.
    
    "init_frame_id" is the first frame that this vehicle appears on.
    
    "instance_list" is a list of instances detected on each frame.
    
    "states_list" is a list of computed 3D states of the corresponding 
        instance on each frame.
    
    "prediction_iteration" is the number of prediction iterations running
        before an update operation, which is used to determine when the
        vehicle leaves the field of view.
    
    The vehicle is currently modeled as a box shaped rigid body. The position
    of the vehicle is represented by a single point (i.e., the "pos" state).
    The velocity of the vehicle is represented by a vector.
    
    The point representing the position may be located in the middle point
    of the bottom edge of the 3D bounding box.
    
    "pos_flag" indicates the location of the point.
    
    "perspective_change" indicates whether the location of the point needs
        to be changed because of the relative change of perspective.
    
    '''

    m1 = 3 + 3 + 3  # id, flags, heading in 3D
    m2 = 3 + 3 + 3 + 9 + 3 + 9  # dim, pos, vel, with cov
    m = m1 + m2

    def __init__(self, vid):

        self.vid = vid
        # self.instance = None

        self.ss = np.zeros((4, 1))
        self.ss_cov = np.zeros((4, 4))

        # Initialize dimension states.

        self.dim = [5, 2, 1.8]
        self.dim_uc = [1.0, 0.2, 0.2]

        self.index = None

        self.association = []

        self.dim_fuse = None
        self.pose_fuse = None
        self.hp_fuse = None

    def state_init(self, instance, states_3d):

        self.instance = instance

        # Initialize heading, position and velocity states.
        self.prediction_iteration = 0
        # self.pos_flag = 8
        self.pos_flag = states_3d.pos_flag
        self.perspective_change = 0

        self.heading = states_3d.heading

        pos = states_3d.get_center()
        pos_uc = states_3d.pos_uc

        vel = states_3d.vel
        vel_uc = states_3d.vel_uc

        ss = np.zeros((4, 1))
        ss[0:2, 0] = pos[0:2]
        ss[2:4, 0] = vel[0:2]

        std_px = pos_uc
        std_py = pos_uc
        std_vx = vel_uc
        std_vy = vel_uc

        ss_cov = np.zeros((4, 4))
        ss_cov[0, 0] = std_px * std_px
        ss_cov[1, 1] = std_py * std_py
        ss_cov[2, 2] = std_vx * std_vx
        ss_cov[3, 3] = std_vy * std_vy

        self.ss = ss
        self.ss_cov = ss_cov

        # Initialize dimension states.

        self.dim = [5, 2, 1.8]
        self.dim_uc = [1.0, 0.2, 0.2]

        self.dim_update(states_3d.dim, states_3d.dim_uc)

        # print(self.vid, 'init')

    def state_predict(self, delta_t):
        '''
        
        The prefix "tm_" means "state transfering matrix".
        The suffix "_p" means "prediction".
        '''

        self.prediction_iteration += 1
        self.instance = None

        tm_pos_std = 0.2
        tm_vel_std = 0.2

        tm = np.identity(4)
        tm[0, 2] = delta_t
        tm[1, 3] = delta_t

        tm_std_px = tm_pos_std
        tm_std_py = tm_pos_std
        tm_std_vx = tm_vel_std
        tm_std_vy = tm_vel_std

        tm_cov = np.zeros((4, 4))
        tm_cov[0, 0] = tm_std_px * tm_std_px
        tm_cov[1, 1] = tm_std_py * tm_std_py
        tm_cov[2, 2] = tm_std_vx * tm_std_vx
        tm_cov[3, 3] = tm_std_vy * tm_std_vy

        ss = self.ss
        ss_cov = self.ss_cov

        ss_p = np.matmul(tm, ss)
        ss_cov_p = np.matmul(np.matmul(tm, ss_cov), tm.T) + tm_cov

        self.ss = ss_p
        self.ss_cov = ss_cov_p

        # print(self.vid, 'predict', self.prediction_iteration)

    def state_update(self, instance, states_3d):
        '''
        
        The prefix "mm_" means "measturement matrix".
        The suffix "_m" means "measurement".
        The suffix "_p" means "prediction".
        The suffix "_u" means "updated".
        
        '''

        self.prediction_iteration = 0
        self.instance = instance

        # Average filtering on heading if the vehicle moves.
        ss_vel = self.ss[2:4].flatten()
        vel_n = np.linalg.norm(ss_vel)
        if vel_n > 0.5:
            self.heading = states_3d.heading
            # self.heading = (self.heading + states_3d.heading) / 2
            # self.heading = self.heading / np.linalg.norm(self.heading)

        if self.pos_flag == states_3d.pos_flag:

            mm_pos = states_3d.pos
            mm_pos_uc = states_3d.pos_uc

            self.perspective_change = 0

        else:

            if self.pos_flag == states_3d.pos_backup_flag:

                mm_pos = states_3d.pos_backup
                mm_pos_uc = states_3d.pos_backup_uc

                self.perspective_change += 1

            else:

                self.adjust_pos_states(states_3d)
                mm_pos = states_3d.pos
                mm_pos_uc = states_3d.pos_uc

                self.perspective_change = 0

        # print(instance.heading_flag, instance.heading, instance.motion_flag, instance.motion)

        # mm_pos = states_3d.get_center()
        # mm_pos_uc = states_3d.pos_uc

        mm_vel = states_3d.vel
        mm_vel_uc = states_3d.vel_uc

        # CAUTION: It is assumed that all states are directly observeable,
        # i.e., z = Hx, where H is the identity matrix.

        ss_m = np.zeros((4, 1))
        ss_m[0:2, 0] = mm_pos[0:2]
        ss_m[2:4, 0] = mm_vel[0:2]

        mm_std_px = mm_pos_uc
        mm_std_py = mm_pos_uc
        mm_std_vx = mm_vel_uc
        mm_std_vy = mm_vel_uc

        mm_cov = np.zeros((4, 4))
        mm_cov[0, 0] = mm_std_px * mm_std_px
        mm_cov[1, 1] = mm_std_py * mm_std_py
        mm_cov[2, 2] = mm_std_vx * mm_std_vx
        mm_cov[3, 3] = mm_std_vy * mm_std_vy

        ss_p = self.ss
        ss_cov_p = self.ss_cov

        k_gain = np.matmul(ss_cov_p, np.linalg.inv(ss_cov_p + mm_cov))

        ss_u = ss_p + np.matmul(k_gain, ss_m - ss_p)
        ss_cov_u = np.matmul(np.identity(4) - k_gain, ss_cov_p)

        self.ss = ss_u
        self.ss_cov = ss_cov_u

        self.dim_update(states_3d.dim, states_3d.dim_uc)

    def adjust_pos_states(self, states_3d):

        # TODO: use current dimension to do the adjustment

        self.ss[0:2, 0] = states_3d.pos[0:2]
        self.pos_flag = states_3d.pos_flag

        pass

    def dim_update(self, dim, dim_uc):

        for i in range(3):
            d = self.dim[i]
            uc = self.dim_uc[i]

            d_new = dim[i]
            uc_new = dim_uc[i]

            gain = uc * uc / (uc * uc + uc_new * uc_new)

            d_update = d + gain * (d_new - d)
            uc_update = math.sqrt(gain * uc_new * uc_new)

            self.dim[i] = d_update
            self.dim_uc[i] = uc_update

    def get_pos_vel(self):

        pos = np.zeros(3)
        pos[0:2] = self.ss[0:2].flatten()

        vel = np.zeros(3)
        vel[0:2] = self.ss[2:4].flatten()

        return pos, vel, self.pos_flag

    def get_pos_vel_uc(self):

        return math.sqrt(self.ss_cov[0, 0]), math.sqrt(self.ss_cov[2, 2])

    def get_3D_anchor_points(self):
        '''
        Get a few 3D anchor points on this vehicle.

        This method is used in vehicle re-association and multiple view
        vehicle association. 
        '''

        dim = self.dim
        heading = self.heading

        pos_flag = self.pos_flag
        pos = np.zeros(3)
        pos[0:2] = self.ss[0:2].flatten()

        nv = heading / np.linalg.norm(heading)
        nt = np.asarray((-nv[1], nv[0], 0))
        nu = np.cross(nv, nt)

        if pos_flag == 0:

            # pos is (qa + qb) / 2
            center = pos - nv * dim[0] / 2

        elif pos_flag == 1:

            # pos is (qa + qc) / 2
            center = pos - nt * dim[1] / 2

        elif pos_flag == 2:

            # pos is (qb + qd) / 2
            center = pos + nt * dim[1] / 2

        elif pos_flag == 3:

            # pos is (qc + qd) / 2
            center = pos + nv * dim[0] / 2

        elif pos_flag == 8:

            center = pos

        else:

            # BUG
            print('BUG in get_center(), pos_flag = %d' % pos_flag)
            pass

        qx = center + nv * dim[0] / 2
        qy = center + nt * dim[1] / 2
        qz = center + nu * dim[2] / 2

        heading_point = center + nv

        return pos, center, heading_point, (qx, qy, qz), (nv, nt, nu)

    def get_volumn(self):

        return self.dim[0] * self.dim[1] * self.dim[2]

    # --------------- These methods are for single view re-association. -------

    def check_pos_vel_heading_similarity(self, states_3d):

        self_pos, self_center, _hp, _q, (self_nv, _nt, _nu) = self.get_3D_anchor_points()
        self_pos, self_vel, self_pos_flag = self.get_pos_vel()

        ob_center = states_3d.get_center()
        ob_vel = states_3d.vel
        ob_nv = states_3d.heading / np.linalg.norm(states_3d.heading)

        distance = np.linalg.norm(self_center - ob_center)

        # CAUTION: nv_instance or nv_vehicle may be zero
        nv_instance = np.linalg.norm(ob_vel)
        nv_vehicle = np.linalg.norm(self_vel)
        if nv_instance > 0.1 and nv_vehicle > 0.1:
            direction_sim = np.dot(ob_vel, self_vel)
            direction_sim = direction_sim / nv_instance / nv_vehicle
        else:
            direction_sim = 1.0

        heading_sim = np.dot(ob_nv, self_nv)

        return distance, direction_sim, heading_sim

    # --------------- These methods are for multiple view fusion. -------------

    def compute_states_global(self, map_model):

        mm = map_model

        pos_flag = self.pos_flag

        pos, center, hp, (qx, qy, qz), (nv, nt, nu) = \
            self.get_3D_anchor_points()

        qs = np.zeros((3, 6))

        qs[:, 0] = pos
        qs[:, 1] = center
        qs[:, 2] = hp
        qs[:, 3] = qx
        qs[:, 4] = qy
        qs[:, 5] = qz

        rs = mm.transform_points_xyz_to_lmap(qs)
        rs = mm.transform_points_lmap_to_gmap(rs)

        self.pos_gmap = rs[:, 0]
        self.center_gmap = rs[:, 1]
        self.hp_gmap = rs[:, 2]
        self.dx_gmap = rs[:, 3] - rs[:, 1]
        self.dy_gmap = rs[:, 4] - rs[:, 1]
        self.dz_gmap = rs[:, 5] - rs[:, 1]

    def compare_heading_global(self, center_request, hp_request):

        v0 = self.hp_gmap - self.center_gmap
        v1 = hp_request - center_request

        v0 = v0 / np.linalg.norm(v0)
        v1 = v1 / np.linalg.norm(v1)

        return np.dot(v0, v1)

    def fuse_dimension(self, vehicles):

        dim_fuse = [0, 0, 0]
        dx = 0
        dy = 0
        dz = 0
        d_uc = [1e6, 1e6, 1e6]

        for vehicle in vehicles:

            if vehicle.dim_uc[0] < d_uc[0]:
                dim_fuse[0] = vehicle.dim[0]
                dx = vehicle.dx_gmap
                d_uc[0] = vehicle.dim_uc[0]

            if vehicle.dim_uc[1] < d_uc[1]:
                dim_fuse[1] = vehicle.dim[1]
                dy = vehicle.dy_gmap
                d_uc[1] = vehicle.dim_uc[1]

            if vehicle.dim_uc[2] < d_uc[2]:
                dim_fuse[2] = vehicle.dim[2]
                dz = vehicle.dz_gmap
                d_uc[2] = vehicle.dim_uc[2]

        self.dx_fuse = dx
        self.dy_fuse = dy
        self.dz_fuse = dz
        self.dim_fuse = dim_fuse

    def fuse_states(self):

        if len(self.association) == 0:
            self.center_fuse = self.center_gmap
            self.hp_fuse = self.hp_gmap
            self.dx_fuse = self.dx_gmap
            self.dy_fuse = self.dy_gmap
            self.dz_fuse = self.dz_gmap
            self.dim_fuse = self.dim

            return

        # CAUTION: Currently a vehicle can be observed by at most two views.

        all = []
        obs = []

        if self.prediction_iteration == 0:
            obs.append(self)

        for vehicle in self.association:

            if vehicle.prediction_iteration == 0:
                obs.append(vehicle)

            all.append(vehicle)

        all.append(self)

        n = len(all)
        m = len(obs)

        if m == 0:

            # If this vehicle can only be observed in one view, just
            # use the states calculated in this view.

            sv = self
            for v in self.association:

                if v.prediction_iteration < sv.prediction_iteration:
                    sv = v

            self.center_fuse = sv.center_gmap
            self.hp_fuse = sv.hp_gmap
            self.dx_fuse = sv.dx_gmap
            self.dy_fuse = sv.dy_gmap
            self.dz_fuse = sv.dz_gmap
            self.dim_fuse = sv.dim

        else:

            # If this vehicle can be observed in multiple views, use
            # the average of the states calculated in these views.

            center_ob = np.zeros((m, 3))
            hp_ob = np.zeros((m, 3))

            for i, v in enumerate(obs):
                center_ob[i] = v.center_gmap
                hp_ob[i] = v.hp_gmap

            center_fuse = np.mean(center_ob, axis=0)
            hp_fuse = np.mean(hp_ob, axis=0)

            self.center_fuse = center_fuse
            self.hp_fuse = hp_fuse
            self.fuse_dimension(obs)

    def to_array(self, i):

        m1 = self.m1
        m2 = self.m2

        a1 = np.zeros(m1)
        a2 = np.zeros(m2)

        a1[0] = i
        a1[1] = self.vid
        a1[2] = 0
        a1[3] = self.prediction_iteration
        a1[4] = self.pos_flag
        a1[5] = self.perspective_change

        a1[6:9] = self.heading

        a2[0:3] = self.dim
        a2[3:6] = self.dim_uc

        a2[6] = self.ss[0]
        a2[7] = self.ss[1]
        a2[8] = 0
        a2[9] = self.ss_cov[0, 0]
        a2[10] = self.ss_cov[0, 1]
        a2[11] = 0
        a2[12] = self.ss_cov[1, 0]
        a2[13] = self.ss_cov[1, 1]
        a2[14] = 0
        a2[15] = 0
        a2[16] = 0
        a2[17] = 0

        a2[18] = self.ss[2]
        a2[19] = self.ss[3]
        a2[20] = 0
        a2[21] = self.ss_cov[2, 2]
        a2[22] = self.ss_cov[2, 3]
        a2[23] = 0
        a2[24] = self.ss_cov[3, 2]
        a2[25] = self.ss_cov[3, 3]
        a2[26] = 0
        a2[27] = 0
        a2[28] = 0
        a2[29] = 0

        return np.concatenate((a1, a2))

    def from_array(self, array):

        m1 = self.m1
        m2 = self.m2

        c0 = 0
        c1 = m1
        c2 = m1 + m2

        a1 = array[c0:c1]
        a2 = array[c1:c2]

        self.prediction_iteration = a1[3]
        self.pos_flag = a1[4]
        self.perspective_change = a1[5]

        self.heading = a1[6:9]

        self.dim = a2[0:3]
        self.dim_uc = a2[3:6]

        self.ss[0] = a2[6]
        self.ss[1] = a2[7]
        self.ss_cov[0, 0] = a2[9]
        self.ss_cov[0, 1] = a2[10]
        self.ss_cov[1, 0] = a2[12]
        self.ss_cov[1, 1] = a2[13]

        self.ss[2] = a2[18]
        self.ss[3] = a2[19]
        self.ss_cov[2, 2] = a2[21]
        self.ss_cov[2, 3] = a2[22]
        self.ss_cov[3, 2] = a2[24]
        self.ss_cov[3, 3] = a2[25]


class TrackerConf(object):

    def __init__(self, folder, prefix, postfix):

        self.folder = folder
        self.prefix = prefix
        self.postfix = postfix

    def create_folders(self, folder, prefix, postfix, fmt):

        subfolders = ['instance', 'tracking', 'vehicle', 'multiple_view']
        if not os.path.isdir(folder + '/tracking'):
            os.mkdir(folder + '/tracking')
        for subfolder in subfolders:
            subfolder = folder + '/tracking/' + subfolder + '_' + fmt + '_' + prefix + postfix
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)
        pass

    def get_fn_per_frame(self, subfolder, fmt, attribute_type, frame_id):

        fid_str = '_%d' % frame_id

        fn = self.folder + '/tracking/' + subfolder + '_' + fmt \
             + '_' + self.prefix + self.postfix + '/' \
             + self.prefix + self.postfix + '_' + attribute_type + fid_str

        return fn

    def get_fn_per_frame_per_instance(self, subfolder, fmt, attribute_type, frame_id, tracking_id):

        fid_str = '_%d' % frame_id
        tid_str = '_%d' % tracking_id

        fn = self.folder + '/tracking/' + subfolder + '_' + fmt \
             + '_' + self.prefix + self.postfix + '/' \
             + self.prefix + self.postfix + '_' + attribute_type + fid_str + tid_str

        return fn

    def get_calibration_folder(self):

        return self.folder + '/calibration_2d/' + self.prefix

    def get_bg_fn(self):

        folder = self.get_calibration_folder()
        return folder + '/' + self.prefix + '_bg_ref.png'

    def get_mask_fn(self):

        folder = self.get_calibration_folder()
        return folder + '/' + self.prefix + '_bg_mask.png'

    def get_map_local_fn(self):

        folder = self.get_calibration_folder()
        return folder + '/' + self.prefix + '_map_local.png'

    def get_map_global_fn(self):

        folder = self.get_calibration_folder()
        return folder + '/' + self.prefix + '_map_global.png'

    def get_video_fn(self, extension):

        # return self.folder + '/video/' + self.prefix + self.postfix + '.mpg'
        return self.folder + '/video/' + self.prefix + self.postfix + extension


class TrackerStats(object):

    def __init__(self):
        self.ob = 0

        self.d_tp = 0
        self.d_fp = 0
        self.d_fn = 0

        self.t_tp = 0
        self.t_fp = 0
        self.t_fn = 0
        self.t_mme = 0

    def calculate_metrics(self):
        moda = 1 - (self.d_fn + self.d_fp) / self.ob
        mota = 1 - (self.t_fn + self.t_fp + self.t_mme) / self.ob


class Tracker(object):
    '''
    '''

    def __init__(self, map_model, klt_tracker, detector, camera_model,
                 frame_width, frame_height, h_margin, v_margin, fps):

        self.map_model = map_model
        self.klt_tracker = klt_tracker
        self.detector = detector
        self.camera_model = camera_model

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.h_margin = h_margin
        self.v_margin = v_margin
        self.fps = fps
        self.frame_interval = 1 / fps

        self.vid = 0

        # A dictionary of current vehicles.
        self.current_vehicles = {}

        # A list of vehicles that is removed at the current frame.
        self.removed_vehicles = []

        # A grid hash table to index the vehicle locations on the global
        # map. Note that the unit is global map unit (i.e., map pixels)
        self.location_indices = []
        self.idx_n = 20
        self.idx_m = 20

        for i in range(self.idx_n):
            grids = [[] for _ in range(self.idx_m)]

            self.location_indices.append(grids)

        self.corners_this = []
        self.corners_prev = []

    def setup_first_frame(self, init_frame_id, frame):
        '''
        Setup the tracker with the first frame.
        '''

        self.init_flag = True

        self.frame_current = frame

        # Detect instances.
        instances = self.detect_instances(init_frame_id, frame)
        instances = self.detection_postprocess(instances)

        self.instances = instances
        self.instances_prev = None

    def setup_first_frame_load_and_replay(self, init_frame_id, frame, fmt):

        self.init_flag = True

        self.frame_current = frame

        self.instances = self.load_instances(self.conf, init_frame_id, fmt)
        self.instances_prev = None

    def track_iteration_online(self, frame_id, frame_current):
        '''
        Track vehicle on the current frame.
        
        '''
        print("Frame ID: ", frame_id)
        self.frame_current = frame_current

        # Detect instances.
        instances = self.detect_instances(frame_id, frame_current)
        instances = self.detection_postprocess(instances)

        self.instances = instances
        instances_prev = self.instances_prev

        # Compute sparse optical flow.
        corners_this, corners_prev = self.klt_tracker.track_iteration_online(
            self.frame_prev, frame_current)

        self.corners_this = corners_this
        self.corners_prev = corners_prev

        # Associate instances.
        self.associate_instances(instances, instances_prev,
                                 corners_this, corners_prev)

        # Estimate motion of instance using optical flow.
        self.estimate_motion(instances)

        # Infer missing instances.
        # CAUTION: This does not change instances on the previous frame!
        # Hoever, it may change the current instance list!
        # instances = self.infer_instances(instances, instances_prev, frame_id)

        # Estimate heading.
        self.estimate_heading(instances)

        # Compute 3D bounding box
        instances_3d_states = self.compute_3d_states(
            self.camera_model, self.map_model, instances, frame_id)

        for i, instance in enumerate(instances):
            instance.tracking_id = i

            # print('\t', i, instance.instance_flag, instance.motion_flag, instance.heading_flag, end='')
            # if instance.states_3d is not None:
            #     print(', ', instance.states_3d.states_flag)
            # else:
            #     print(', ', 'None')

        # Associate detection results to vehicle states,
        # infer the states if necessary.
        self.estimate_vehicle_states(frame_id, instances, instances_3d_states)

        # Validating the states of one specific vehicle.
        # mm = self.map_model
        # vid_x = 8
        # if vid_x in self.current_vehicles:

        #     vehicle_x = self.current_vehicles[vid_x]

        #     pos, center, _hp, _qs, _ns = vehicle_x.get_3D_anchor_points()

        #     center_esd = mm.transform_points_xyz_to_esd(center.reshape(3, 1))
        #     center_llh = mm.transform_points_esd_to_llh(center_esd)
        #     center_llh = center_llh.flatten()
        #     print(frame_id, center[0], center[1], center_llh[0], center_llh[1])

        self.instances = instances
        self.instances_3d_states = instances_3d_states

        self.init_flag = False

    def detect_instances(self, frame_id, frame):

        score_th = 0.80

        boxes, centers, masks, contours, labels, scores = \
            self.detector.detect(frame_id, frame)

        instances = []

        n = boxes.shape[0]
        for i in range(n):

            if np.sum(masks[i]) <= 50:
                continue

            if scores[i] < score_th:
                continue

            if not check_box_on_mask_bg(boxes[i], self.mask_bg):
                continue

            instance = Instance(i, boxes[i], centers[i],
                                masks[i], contours[i], labels[i], scores[i])

            instances.append(instance)

        return instances

    def detection_postprocess(self, instances):

        processed_instances = []

        n = len(instances)

        for i in range(n):

            instance_i = instances[i]
            box_i = instance_i.box
            center_i = instance_i.center
            score_i = instance_i.score

            cx = int(center_i[0])
            cy = int(center_i[1])

            duplicate = False
            for j in range(i + 1, n):

                instance_j = instances[j]
                box_j = instance_j.box
                mask_j = instance_j.mask

                if mask_j[cy, cx]:
                    duplicate = True
                    break

            for j in range(n):

                if i == j:
                    continue

                instance_j = instances[j]
                box_j = instance_j.box

                # print(i, j)
                if check_box_bottom_occlusion(box_i, box_j):
                    instance_i.occlusion_flag = 1

            label_good = self.check_label(instance_i.label)
            if not duplicate and label_good:
                processed_instances.append(instance_i)

        return processed_instances

    def check_label(self, label):
        '''
        Check whether the label is "car", "truck", or "bus".
        '''

        # return label == 2 or label == 5 or label == 7
        return label == 3 or label == 6 or label == 8

    def associate_instances(self, instances, instances_prev, corners, corners_prev):
        '''
        Associate instances on the current frame and the previous frame.
        '''

        n = len(instances)

        # Associate sparse optical flow vectors to instances.
        self.cluster_corners_and_sofs(
            corners, corners_prev, instances, instances_prev)

        # Associate with previous instance using the sparse optical flow, if
        # there are more than a certain number of corners correctly tracked.
        self.associate_instances_by_optical_flow(instances, instances_prev)

        # Associate the remaining instances using masks, if a mask on the
        # previous frame overlaps with a mask on the current frame with
        # a certain percentage.

        self.associate_instances_by_mask_overlap(instances, instances_prev)

        # Remove one-to-many and many-to-one associations.

        self.check_association(instances, instances_prev)

        for instance in instances:

            if instance.prev is not None:
                instance.list_idx = instance.prev.list_idx + 1

    def cluster_corners_and_sofs(self,
                                 corners, corners_prev, instances, instances_prev):
        '''
        Cluster the corners and sparse optical flow vectors with instances.
        '''
        n = len(instances)
        m = len(instances_prev)

        for cc, pc in zip(corners, corners_prev):

            # CAUTION: a is the row, i.e., the y coordinate, 
            # and b is the column, i.e., the x coordinate.

            a, b = cc.ravel()
            c, d = pc.ravel()

            ai = int(round(a))
            bi = int(round(b))
            ci = int(round(c))
            di = int(round(d))

            if di < 0 or di >= self.frame_height \
                    or ci < 0 or ci >= self.frame_width \
                    or bi < 0 or bi >= self.frame_height \
                    or ai < 0 or ai >= self.frame_width:
                # print('SOF vector outside view: (%d, %d) -> (%d, %d)' % (c, d, a, b))

                continue

            for i in range(n):

                instance = instances[i]
                mask = instance.mask

                if mask[bi, ai]:
                    # CAUTION: direction is from frame to frame_prev
                    sof_to_prev = np.asarray((c - a, d - b))

                    instance.corners_to_prev.append(cc)
                    instance.corresponding_corners_on_prev.append(pc)
                    instance.sofs_to_prev.append(sof_to_prev)

                    break

            for j in range(m):

                instance_prev = instances_prev[j]
                mask_prev = instance_prev.mask

                if mask_prev[di, ci]:
                    # CAUTION: direction is from frame_prev  to frame
                    sof_to_next = np.asarray((a - c, b - d))

                    instance_prev.corners_to_next.append(pc)
                    instance_prev.corresponding_corners_on_next.append(cc)
                    instance_prev.sofs_to_next.append(sof_to_next)

                    break

                # CAUTION: It is possible that a corner is not 
                # associated with any instance.

        return instances, instances_prev

    def associate_instances_by_optical_flow(self, instances, instances_prev):
        '''
        Associate instances using the sparse optical flow.
        '''

        nr_link_threshold = 3

        n = len(instances)
        m = len(instances_prev)

        # Associate each instance to instance_prev using sparse optical flow.
        # CAUTION: There might be many-to-one mapping! 
        for i in range(n):

            instance = instances[i]
            corners = instance.corresponding_corners_on_prev
            links = []

            for corner in corners:

                x_prev, y_prev = corner.ravel()

                for j in range(m):

                    instance_prev = instances_prev[j]
                    mask_prev = instance_prev.mask

                    if mask_prev[int(round(y_prev)), int(round(x_prev))]:
                        links.append(j)

                        break

            if len(links) >= nr_link_threshold:
                votes = Counter(links)
                majority_j, count = votes.most_common(1)[0]
                if count >= nr_link_threshold:
                    instance_prev = instances_prev[majority_j]
                    instance.prev = instance_prev

        # Associate each mask_2 to mask_1 using sparse optical flow.
        # CAUTION: There might be many-to-one mapping! 
        for j in range(m):

            instance_prev = instances_prev[j]
            corners = instance_prev.corresponding_corners_on_next
            links = []

            for corner in corners:

                x, y = corner.ravel()

                for i in range(n):

                    instance = instances[i]
                    mask = instance.mask

                    if mask[int(round(y)), int(round(x))]:
                        links.append(i)

                        break

            if len(links) >= nr_link_threshold:
                votes = Counter(links)
                majority_i, count = votes.most_common(1)[0]
                if count >= nr_link_threshold:
                    instance_prev.next = instances[majority_i]

        return instances, instances_prev

    def associate_instances_by_mask_overlap(self, instances, instances_prev):
        '''
        Associate instances using the mask overlap.
        '''

        iox_th = 0.3

        n = len(instances)
        m = len(instances_prev)

        for i in range(n):

            instance = instances[i]

            if instance.prev is not None:
                continue

            for j in range(m):

                instance_prev = instances_prev[j]

                if instance_prev.next is not None:
                    continue

                box_overlap = check_box_overlap(
                    instance.box, instance_prev.box)

                if not box_overlap:
                    continue

                iou, io1, io2 = calculate_mask_overlap(
                    instance.mask, instance_prev.mask)

                if iou > iox_th or io1 > iox_th or io2 > iox_th:
                    instance.prev = instance_prev
                    instance_prev.next = instance

        return instances, instances_prev

    def check_association(self, instances, instances_prev):
        '''
        Remove bad associations.
        '''

        # CAUTION: If "iox_th" is very high, associations of instances
        # that move fast may be wrongly rejected.
        iox_th = 0.3

        # Remove one-to-many and many-to-one associations.

        for instance in instances:

            if instance.prev is None:
                continue

            instance_prev = instance.prev
            instance_prev_next = instance_prev.next

            # CAUTION: "instance_prev_next" could be None!
            if instance is not instance_prev_next:
                instance.prev = None
                instance.list_idx = 0

        for instance_prev in instances_prev:

            if instance_prev.next is None:
                continue

            instance_prev_next = instance_prev.next
            instance_prev_next_prev = instance_prev_next.prev

            # CAUTION: "instance_prev_next_prev" could be None!
            if instance_prev is not instance_prev_next_prev:
                instance_prev.next = None

        for i, instance in enumerate(instances):

            if instance.prev is None:
                continue

            instance_prev = instance.prev

            iou, io1, io2 = calculate_mask_overlap(
                instance.mask, instance_prev.mask)

            if not (iou > iox_th or io1 > iox_th or io2 > iox_th):
                # print('\tremove %d.prev' % i, iou, io1, io2)

                instance.prev = None
                instance_prev.next = None
                instance.list_idx = 0

    def estimate_motion(self, instances):
        '''
        Calculate apparent motion of each instance.
        '''

        momentum_n_th = 10
        motion_reject_th = 12
        motion_override_th = 12

        # Estimate heading from sparse optical flow.

        n = len(instances)

        for i in range(n):

            # print('---- instance %d -----' % i)
            instance = instances[i]

            # CAUTION: Do not estimate motion if this instance is only 
            # partially observable.
            # if self.check_box_outside_view(instance.box):
            #    continue

            corners = instance.corners_to_prev
            sofs = instance.sofs_to_prev

            sof_mean_inlier, sof_mean_all, motion_flag, corners_inlier, sofs_inlier \
                = self.calculate_motion_from_sof(corners, sofs)

            corners_inlier = np.asarray(corners_inlier).reshape((-1, 2))
            sofs_inlier = np.asarray(sofs_inlier).reshape((-1, 2))

            # CAUTION: Check whether sparse optical flow vectors agrees with
            # the motion of the 2D bounding box. Override it if necessary. 

            # motion_from_boxes = self.calculate_motion_from_boxes(instance)
            # if motion_from_boxes is not None and motion_flag == 4:

            #     motion_diff = np.linalg.norm(motion_from_boxes - sof_mean_inlier)
            #     if motion_diff > motion_reject_th:

            #         #print('\t', i, 'reject, ', motion_from_boxes, sof_mean_inlier)
            #         motion_flag = 0
            #         sof_mean_inlier = np.zeros(2)
            #         sof_mean_all = np.zeros(2)
            #         corners_inlier = np.zeros((0, 2))
            #         sofs_inlier = np.zeros((0, 2))

            # if motion_flag == 0 and motion_from_boxes is not None \
            #     and np.linalg.norm(motion_from_boxes) > motion_override_th:

            #     #print('\t', i, 'override, ', motion_from_boxes, sof_mean_inlier)
            #     motion_flag = 2
            #     sof_mean_inlier = motion_from_boxes
            #     sof_mean_all = motion_from_boxes
            #     corners_inlier = np.zeros((0, 2))
            #     sofs_inlier = np.zeros((0, 2))

            # TODO: Check whether the instance has no motion using the frame
            # difference.

            # diff_aggregate = self.check_no_motion_by_frame_difference(instance)

            # print('\t', i, motion_flag, diff_aggregate)

            # CAUTION: The sparse optical flow vectors are from the corners
            # on the current frame to those on the previous frame. Hence,
            # the motion is on the opposite direction to the flow vectors.

            # CAUTION: Make a fresh copy of these for motion and momentum!!!

            if motion_flag == 4 or motion_flag == 2:
                instance.motion = -sof_mean_inlier.copy()
                momentum = -sof_mean_inlier.copy()
            else:
                instance.motion = -sof_mean_all.copy()

                if instance.occlusion_flag > 0:
                    momentum = np.zeros(2)
                else:
                    momentum = -sof_mean_all.copy()

            instance_prev = instance.prev
            if instance_prev is not None:
                momentum += instance_prev.momentum
            momentum_n = np.linalg.norm(momentum)
            if momentum_n > momentum_n_th:
                momentum = momentum / momentum_n * momentum_n_th

            instance.momentum = momentum

            instance.motion_flag = motion_flag

            instance.corners_to_prev_inliers = corners_inlier
            instance.sofs_to_prev_inliers = sofs_inlier

    def calculate_motion_from_sof(self, corners, sofs):
        '''
        Calculate apparent motion on the image from sparse optical flow.
        '''

        n = 5
        d = 3
        cos_sim_t = 0.99
        flow_th = 1

        k = len(corners)

        default_motion = np.zeros(2)

        if k < d:
            # Too few corner points.
            return default_motion, default_motion, 0, [], []

        corners_array = np.asarray(corners)
        sofs_array = np.asarray(sofs)

        sofs_norm = np.linalg.norm(sofs_array, axis=1)

        # Select those sparse optical flow vectors that have similar length. 
        sofs_norm_median = np.median(sofs_norm)

        idx0 = np.argwhere(sofs_norm < (sofs_norm_median * 3))
        m0 = len(idx0)

        if m0 > 0:
            sofs_low = sofs_array[idx0].reshape((m0, 2))
            sof_mean_all = np.mean(sofs_low, axis=0)
        else:
            sof_mean_all = default_motion

        if m0 < d:
            # Flow vectors do not agree with each other.
            return default_motion, default_motion, 0, [], []

        # Select those corner features with a sparse optical flow larger than a threshold.
        idx1 = np.argwhere(sofs_norm > flow_th)

        m1 = len(idx1)

        if m1 < d:
            # Majority of the corners are not moving.
            idx1x = np.argwhere(sofs_norm < flow_th)
            m1x = len(idx1x)
            corners_inlier = corners_array[idx1x].reshape((m1x, 2))
            sofs_inlier = sofs_array[idx1x].reshape((m1x, 2))

            return default_motion, sof_mean_all, 1, corners_inlier, sofs_inlier

        corners_array = corners_array[idx1].reshape((m1, 2))
        sofs_array = sofs_array[idx1].reshape((m1, 2))

        sofs_n = sofs_array.copy()
        sofs_norm = sofs_norm[idx1].reshape(m1)

        sofs_n[:, 0] = np.divide(sofs_array[:, 0], sofs_norm)
        sofs_n[:, 1] = np.divide(sofs_array[:, 1], sofs_norm)

        # print('sofs_n.shape', sofs_n.shape)

        # ov_mean = np.mean(sofs_n, axis=0)

        # RANSAC

        m2 = 0
        idx2 = []

        for ii in range(n):

            # consensus set
            idx_t = []
            mt = 0

            # Choose two sparse optical flow vectors to find a vanishing point.
            # Then check whether the other sparse optical flow vectors can
            # meet at this vanishing point. Those who can meet there are among
            # the consensus set.

            choices = random.choices(range(m1), k=2)
            i = choices[0]
            j = choices[1]

            pi = corners_array[i]
            piH = np.ones(3)
            piH[0:2] = pi

            vi = sofs_n[i]
            viH = np.zeros(3)
            viH[0:2] = vi

            li = np.cross(piH, viH)

            pj = corners_array[j]
            pjH = np.ones(3)
            pjH[0:2] = pj

            vj = sofs_n[j]
            vjH = np.zeros(3)
            vjH[0:2] = vj

            lj = np.cross(pjH, vjH)

            vpH = np.cross(li, lj)

            if vpH[2] < 1e-8:

                # li and lj are basically parallel.

                cos_sim = np.matmul(sofs_n, vi.T).flatten()
                idx_t = np.argwhere(cos_sim > cos_sim_t).flatten()
                tm = len(idx_t)

            else:

                # li and lj meet at a vanishing point vp.
                vpH[0] /= vpH[2]
                vpH[1] /= vpH[2]
                vp = vpH[0:2]

                idx_t.append(i)
                idx_t.append(j)
                for k in range(m1):

                    if k == i or k == j:
                        continue

                    pk = corners_array[k]
                    vk = sofs_n[k]

                    uk = vp - pk
                    uk = uk / np.linalg.norm(uk)
                    if np.dot(uk, vi) < 0:
                        uk = -uk

                    cos_sim_uk = np.dot(uk, vk)
                    if cos_sim_uk > cos_sim_t:
                        idx_t.append(k)

                tm = len(idx_t)

            if tm > d and tm > m2:
                idx2 = idx_t
                m2 = tm

        m2 = len(idx2)

        # CAUTION: The motion is valid if more than half of the corners
        # with sparse optical flow agree with each other. 
        if m2 < m0 // 2:
            # Corners do not agree with each other.
            return default_motion, default_motion, 0, [], []

        # print(m1, m2, idx2)

        corners_inlier = corners_array[idx2]
        sofs_inlier = sofs_array[idx2]
        # sofs_n_inlier = sofs_n[idx2]

        sof_mean_inlier = np.mean(sofs_inlier, axis=0)
        # sof_mean = np.mean(sofs_n_inlier, axis=0)

        # print('corners_inlier.shape', corners_inlier.shape)

        # Finally a consensus set of sparse optical flow is computed.

        return sof_mean_inlier, sof_mean_all, 4, corners_inlier, sofs_inlier

    def calculate_motion_from_boxes(self, instance):
        '''
        Calculate apparent motion on the image from the 2D bounding boxes
        on the current frame and the previous frame.
        '''

        if instance.prev is None:
            return

        box1 = instance.box
        box2 = instance.prev.box

        x11 = box1[0]
        y11 = box1[1]
        x12 = box1[2]
        y12 = box1[3]

        x21 = box2[0]
        y21 = box2[1]
        x22 = box2[2]
        y22 = box2[3]

        v1 = np.asarray((x21 - x11, y21 - y11))
        v2 = np.asarray((x22 - x12, y22 - y12))

        return (v1 + v2) / 2

    def check_no_motion_by_frame_difference(self, instance):

        frame_current = self.frame_current
        frame_prev = self.frame_prev

        box = instance.box
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        diff_color = frame_current[y1:y2, x1:x2] - frame_prev[y1:y2, x1:x2]
        diff = np.mean(diff_color, axis=2)

        mask_local = instance.mask[y1:y2, x1:x2]
        mask_size = np.sum(mask_local)
        diff_aggregate = np.sum(diff, where=mask_local) / mask_size

        return diff_aggregate

    def infer_instances(self, instances, instances_prev, frame_id):
        '''
        Infer misdetected instances as needed.
        '''

        size_th = 400
        ratio_th = 0.7

        # If we do not want to run instance inference, just return now.
        # return

        n = len(instances)
        m = len(instances_prev)

        # Infer instances that are detected on the previous frame but missing 
        # on the current frame.
        inferred_id = n

        for j in range(m):

            instance_prev = instances_prev[j]

            if instance_prev.next is not None:
                continue

            if instance_prev.instance_flag == 0:
                continue

            # CAUTION: Do not infer instances that are partially outside
            # the view.
            # if self.check_box_outside_view(instance_prev.box,
            #    self.frame_width, self.frame_height, self.margin):
            #    continue

            box_size = calculate_box_size(instance_prev.box)
            if box_size < size_th:
                continue

            # CAUTION: Do not infer instances based on inferred instances, 
            # if this inferred instance is partially outside of view.
            view_flag = self.check_box_outside_view(instance_prev.box)
            if view_flag and instance_prev.instance_flag > 1:
                continue

            # CAUTION: Do not infer instances based on inferred instances, 
            # if this inferred instance is partially overlapped with others.
            if instance_prev.occlusion_flag > 0:

                if instance_prev.instance_flag > 1:

                    continue

                elif instance_prev.instance_flag == 1 \
                        and instance_prev.check_mask_box_ratio() < ratio_th:

                    continue

            # CAUTION: Do not infer instances based on inferred instances, 
            # if it is already inferred for more than a few frames.
            if instance_prev.occlusion_flag == 0 \
                    and instance_prev.instance_flag >= 10:
                continue

            # CAUTION: Do not infer instances if its motion cannot be reliably
            # determined.
            if instance_prev.motion_flag == 0:
                continue

            # Now infer this instance.
            instance = self.infer_one_instance(instance_prev)
            inferred_id += 1

            # Fix the occlusion_flag.
            for instance_old in instances:

                if check_box_overlap(instance_old.box, instance.box):
                    instance_old.occlusion_flag = 1
                    instance.occlusion_flag = 1

            instances.append(instance)

        # Handle instance splitting.

        #         for i in range(n):
        #
        #             instance = instances[i]
        #             instance_prev = instance.prev
        #
        #             if instance_prev is None:
        #
        #                 pass
        #
        #             else:
        #
        #                 instance_current = instance_prev.next
        #
        #                 if instance is instance_current:
        #
        #                     pass
        #
        #                 else:
        #
        #                     mask = instance.mask
        #                     mask_current = instance_current.mask
        #
        #                     # "instance" and "instance_current" are split from
        #                     # the same instance on the previous frame
        #                     instance.instance_flag = 0
        #                     merge_instances(instance, instance_current)
        #
        #         instances_updated = []
        #         for instance in instances:
        #             if instance.instance_flag > 0:
        #                 instances_updated.append(instance)

        # No special work needed for handling instances merging.

        return instances

    def infer_one_instance(self, instance_prev):
        '''
        Infer a miss-detected instance using the the previous frame
        '''

        momentum_n_th = 5

        corners = instance_prev.corners_to_next
        sofs = instance_prev.sofs_to_next

        # CAUTION: Here sof_mean_xxx is from the previous frame to the 
        # current frame!
        sof_mean_inlier, sof_mean_all, motion_flag, corners_inlier, sofs_inlier = \
            self.calculate_motion_from_sof(corners, sofs)

        corners_inlier = np.asarray(corners_inlier).reshape((-1, 2))
        sofs_inlier = np.asarray(sofs_inlier).reshape((-1, 2))

        # Compute the corners and sparse optical flow vectors on 
        # the current frame to the previous frame.
        corners_to_prev_inliers = corners_inlier + sofs_inlier
        sofs_to_prev_inliers = -sofs_inlier

        # CAUTION: Make a fresh copy of these for motion and momentum!!!
        if motion_flag == 4:
            motion = sof_mean_inlier.copy()
            momentum = sof_mean_inlier.copy()
        else:
            motion = sof_mean_all.copy()
            momentum = sof_mean_all.copy()

        momentum += instance_prev.momentum
        momentum_n = np.linalg.norm(momentum)
        if momentum_n > momentum_n_th:
            momentum = momentum / momentum_n * momentum_n_th

        # TODO: If there is not enough sparse optical flow vectors available,
        # use dense optical flow instead.

        box, center, mask, contour = self.infer_box_and_mask(
            instance_prev.box, instance_prev.mask, motion)

        instance = Instance(-1, box, center, mask, contour,
                            instance_prev.label, instance_prev.score)

        instance.occlusion_flag = instance_prev.occlusion_flag

        instance.corners_to_prev_inliers = corners_inlier
        instance.sofs_to_prev_inliers = sofs_inlier

        instance.motion = motion
        instance.momentum = momentum
        instance.motion_flag = motion_flag

        instance.instance_flag = instance_prev.instance_flag + 1

        instance.prev = instance_prev
        instance_prev.next = instance

        return instance

    def infer_box_and_mask(self, box, mask, motion):
        '''
        Infer the 2D bounding box and msk of a miss-detected instance.
        '''

        frame_width = self.frame_width
        frame_height = self.frame_height

        box_new = box.copy()

        box_new[0] += motion[0]
        box_new[1] += motion[1]
        box_new[2] += motion[0]
        box_new[3] += motion[1]

        center_new = compute_center_from_box(box)

        # Copy the region mask to mask_new, with an offset (i.e., the motion vector).

        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        u1 = int(box_new[0])
        v1 = int(box_new[1])
        u2 = int(box_new[2])
        v2 = int(box_new[3])

        if x1 < 0:
            x1 = 0

        if y1 < 0:
            y1 = 0

        if x2 >= frame_width:
            x2 = frame_width

        if y2 >= frame_height:
            y2 = frame_height

        if u1 < 0:
            u1 = 0

        if v1 < 0:
            v1 = 0

        if u2 >= frame_width:
            u2 = frame_width

        if v2 >= frame_height:
            v2 = frame_height

        w1 = u2 - u1
        h1 = v2 - v1
        w2 = x2 - x1
        h2 = y2 - y1

        w = min(w1, w2)
        h = min(h1, h2)

        u2 = u1 + w
        v2 = v1 + h
        x2 = x1 + w
        y2 = y1 + h

        mask_new = np.zeros(mask.shape, dtype=mask.dtype)
        mask_new.fill(False)

        # CAUTION: There is a weird problem on ROI size difference!
        # Here we always stick to the target mask ROI size!        
        mask_new[v1:v2, u1:u2] = mask[y1:y2, x1:x2]

        contour_new = compute_contour_from_mask(mask_new)

        return box_new, center_new, mask_new, contour_new

    def check_box_outside_view(self, box, h1=0, h2=0, v1=0, v2=0):
        '''
        Check whether a 2D bounding box is outside of view.
        '''

        w = self.frame_width
        h = self.frame_height
        h_margin = self.h_margin
        v_margin = self.v_margin

        p1x = box[0]
        p1y = box[1]
        p2x = box[2]
        p2y = box[3]

        return p1x < (h_margin + h1) or p2x > (w - h_margin - h2) \
               or p1y < (v_margin + v1) or p2y > (h - v_margin - v2)

    def estimate_heading(self, instances):
        '''
        Estimate the heading based on the motion for each instance.
        '''

        heading_n_th = 2

        for instance in instances:

            motion = instance.motion
            momentum = instance.momentum
            motion_flag = instance.motion_flag

            instance_prev = instance.prev
            if instance_prev is not None:

                # If this is not the first frame that this instance appears,
                # use both the motion observed from the current frame and that
                # from the previous frame to estimate heading.

                motion_prev = instance_prev.motion
                motion_flag_prev = instance_prev.motion_flag

                if motion_flag == 4:

                    # The vehicle on the current frame is moving and heading is
                    # calculated from sparse optical flow.
                    if motion_flag_prev == 4 or motion_flag_prev == 2:
                        heading = motion * 0.6 + motion_prev * 0.4
                        heading_flag = 5
                    else:
                        heading = motion.copy()
                        heading_flag = 4

                elif motion_flag == 2:

                    # The vehicle on the current frame is moving and heading is
                    # calculated from the motion of the 2D bounding boxes.
                    if motion_flag_prev == 4 or motion_flag_prev == 2:
                        heading = motion * 0.6 + motion_prev * 0.4
                        heading_flag = 5
                    else:
                        heading = motion.copy()
                        heading_flag = 4

                elif motion_flag == 1:

                    # The vehicle on the current frame is not moving, or the
                    # movement is very small.
                    if motion_flag_prev == 4 or motion_flag_prev == 2:
                        heading = motion_prev
                        heading_flag = 3
                    else:
                        heading = momentum.copy()
                        heading_flag = 2

                else:

                    # No sparse optical flow points are associated with this
                    # instance. 
                    if motion_flag_prev == 4 or motion_flag_prev == 2:
                        heading = motion_prev
                        heading_flag = 3
                    else:
                        heading = momentum.copy()
                        heading_flag = 2

                if np.linalg.norm(heading) < heading_n_th:
                    heading = momentum.copy()
                    heading_flag = 2

            else:

                # If this is the first frame where this instance appears,
                # just use the motion observed from the current frame
                # to estimate the heading.

                if motion_flag == 4:
                    heading = motion.copy()
                    heading_flag = 4
                else:
                    heading = momentum.copy()
                    heading_flag = motion_flag

                if np.linalg.norm(heading) < heading_n_th:
                    heading = momentum.copy()
                    heading_flag = motion_flag

            if heading_flag < 2 or np.linalg.norm(heading) < heading_n_th:
                # vp = np.array((790, 20))
                vp = np.array((670, 30))
                center = instance.center
                heading = center - vp
                heading = heading / np.linalg.norm(heading)

                heading_flag = 2

            # print(heading_flag)

            heading = heading / np.linalg.norm(heading)
            if instance.prev is not None and instance.prev.heading_flag >= 2:
                heading = 0.5 * heading + 0.5 * instance.prev.heading
            
            print(heading)
            instance.heading = heading
            instance.heading_flag = heading_flag

    def compute_3d_states(self, camera_model, map_model, instances, frame_id):
        '''
        Compute the 3D bounding box, position, velocity, and other states.
        '''
        list_idx_th = 1
        size_th = 200
        heading_n_th = 0.5
        ratio_th = 0.2
        occlusion_size_th = 400

        d = 10
        v1 = 60
        v2 = 10

        vehicle_height_min = 1.0
        vehicle_height_max = 3.0

        instances_3d_states = []
        f = open(os.path.join(r"/scratch/rparikh4/egonet_old/intermediate_results/","{:06d}".format(frame_id)+'.txt'),"r")
        bbox_3d = []
        for line in f.readlines():
            point = line.split()
            bbox_3d.append(np.array([float(point[0]), float(point[1])]))
        centroids = []
        for i in range(len(bbox_3d)//8):
            x = np.mean(bbox_3d[i*8:(i+1)*8], axis=0)
            centroids.append(x)
        centroids = np.array(centroids)
        image = cv2.imread(os.path.join(r"/scratch/rparikh4/egonet_old/qualitative_results/","{:06d}".format(frame_id)+'.png'))
        #print("Image Test Size: ", image.shape)
        image = cv2.resize(image, (1280,720))
        for i in range(len(bbox_3d)):
            cv2.circle(image, (int(bbox_3d[i][0]), int(bbox_3d[i][1])), 1, (0,0,255), -1)
        for i in range(len(centroids)):
            cv2.circle(image, (int(centroids[i][0]), int(centroids[i][1])), 1, (255,0,0), -1)
        cv2.imwrite(os.path.join(r"/scratch/rparikh4/tracking_3d/frames_annotated/",'{:06d}'.format(frame_id) + ".png"),image)
        #print("BBox 3d: ", bbox_3d[0:8])
        #print("Centroids: ", centroids)
        for i, instance in enumerate(instances):

            states_3d = Instance3DStates(instance)
            instances_3d_states.append(states_3d)

            # Only work on those instances that has been consistently tracked.
            if instance.list_idx < list_idx_th:
                continue

            # If the heading is undetermined, BB3D cannot be computed.
            if instance.heading_flag < 2:
                continue

            if np.linalg.norm(instance.heading) < heading_n_th:
                continue

            # Do not compute BB3D if this vehicle is only partially 
            # observable on the frame or close to the horizon.
            if self.check_box_outside_view(instance.box, v1=v1, v2=v2):
                continue

            # Do not compute BB3D if the instance is small. This usually
            # indicates that the vehicle is far away or occluded.
            if calculate_box_size(instance.box) < size_th:
                continue

            # If we already know there are vehicle-vehicle occlusion,
            # be careful on computing the BB3D, since that step requires
            # a complete mask. Otherwise, the computed BB3D could be really
            # bad.
            # print(instance.occlusion_flag)
            if instance.occlusion_flag > 0:
                continue

            if instance.occlusion_flag > 0 and \
                    calculate_box_size(instance.box) < occlusion_size_th:
                continue

            if instance.occlusion_flag > 0 and \
                    instance.check_mask_box_ratio() < ratio_th:
                continue

            # Calculate heading in 3D.

            box = instance.box
            heading = instance.heading.copy()
            heading /= np.linalg.norm(heading)

            p1x = box[0]
            p1y = box[1]
            p2x = box[2]
            p2y = box[3]

            cx = (p1x + p2x) / 2
            cy = (p1y + p2y) / 2
            
            #hx_hy = line.split()
            hx = cx + heading[0] * d
            hy = cy + heading[1] * d
            #print(hx_hy)
            #hx = float(hx_hy[0])
            #hy = float(hx_hy[1])
            print("Hx: ", hx, "Hy: ", hy)
            # CAUTION: Here the reference point of the vehicle is the center 
            # of the bottom edge of the 2D bounding box for calculating
            # the heading!

            p0 = np.asarray((cx, cy))
            px = np.asarray((hx, hy))

            # q0_lmap = camera_model.transform_point_image_to_lmap(p0)
            # qx_lmap = camera_model.transform_point_image_to_lmap(px)

            # q0 = map_model.transform_points_lmap_to_xyz(q0_lmap.reshape((3, 1))).flatten()
            # qx = map_model.transform_points_lmap_to_xyz(qx_lmap.reshape((3, 1))).flatten()

            q0 = camera_model.transform_point_image_to_ground(p0)
            qx = camera_model.transform_point_image_to_ground(px)

            # CAUTION: Make sure these are finite points!!!
            if q0 is None or qx is None:
                continue

            ovx = qx - q0
            ovx = ovx / np.linalg.norm(ovx)
            ovy = np.asarray((-ovx[1], ovx[0], 0))
            qy = q0 + ovy
            ovz = np.asarray((0, 0, 1))
            qz = q0 + ovz

            py = camera_model.transform_point_ground_to_image(qy)
            pz = camera_model.project_point(qz)

            pvx = px - p0
            pvy = py - p0
            pvz = pz - p0

            pvx = pvx / np.linalg.norm(pvx)
            pvy = pvy / np.linalg.norm(pvy)
            pvz = pvz / np.linalg.norm(pvz)

            vpx, vpy, vpz = self.camera_model.calculate_vps(ovx, ovy, ovz)

            #             print('---------- %d ----------' % k)
            #
            #             self.bb3d_detector.vis = False
            #             if k == 2:
            #                 self.bb3d_detector.frame_vis = self.frame_vis3
            #                 self.bb3d_detector.vis = True

            # CAUTION: bb3d might be "None"!!!
            pbox = bbox_3d[i*8:(i+1)*8]
            print(pbox)
            final_pbox = []
            final_pbox.append(pbox[1])
            final_pbox.append(pbox[3])
            final_pbox.append(pbox[5])
            final_pbox.append(pbox[7])
            final_pbox.append(pbox[0])
            final_pbox.append(pbox[2])
            final_pbox.append(pbox[4])
            final_pbox.append(pbox[6])
            print(final_pbox)
            bb3d = bb3d_perspective(bbox_3d, final_pbox, centroids, camera_model, instance.contour,
                                    vpx, vpy, vpz, pvx, pvy, pvz)

            if bb3d is not None:
                # Reject bad BB3D using the height
                # too_small = bb3d.dim[2] < vehicle_height_min
                # too_big = bb3d.dim[2] > vehicle_height_max
                # not_a_truck = int(round(instance.label)) != 8
                # if  too_small or (too_big and not_a_truck):
                #     continue

                # Compute 3D position and velocity states.

                instance.bb3d = bb3d

                states_3d.heading = ovx

                self.compute_pos_vel_states(camera_model, map_model, instance, states_3d)

                # Compute dimension uncertainty.
                # CAUTION: This must be done after compute_pos_vel_states()!
                states_3d.dim = bb3d.dim
                states_3d.dim_uc = self.calculate_dim_uc(camera_model, instance, states_3d)

                states_3d.states_flag = 1
                instance.states_3d = states_3d

        # Try to infer 3D position and velocity states.
        for instance, states_3d in zip(instances, instances_3d_states):

            if instance.states_3d is not None:
                continue

            if self.check_box_outside_view(instance.box, v1=v1, v2=v2):
                continue

            if instance.prev is not None and instance.prev.states_3d is not None:
                states_3d_prev = instance.prev.states_3d

                self.infer_3d_states(instance, instance.prev,
                                     states_3d, states_3d_prev)

                states_3d.states_flag = states_3d_prev.states_flag + 1
                instance.states_3d = states_3d

                # print('\tinfer_3d_states ', i)

        return instances_3d_states

    def compute_pos_vel_states(self, camera_model, map_model, instance, states):
        '''
        Compute the position and velocity states 
        
        Note that the states are computed from the 3D bounding box and
        the appearant motion detected on the frame.
        '''

        p_uc = 3
        pv_uc = 2

        k_min = 4
        k_max = 10
        motion_th = 5

        # CAUTION: The "pos" state depends on which facets are visible
        # on the 3D bounding box.

        bb3d = instance.bb3d

        condition = bb3d.situation
        cases = bb3d.cases

        pa, pb, pc, pd, pe, pf, pg, ph = bb3d.pbox
        qa, qb, qc, qd, qe, qf, qg, qh = bb3d.qbox

        if condition == 1:

            # xvpp

            if cases == 0:

                point = (pc + pd) / 2
                pos = (qc + qd) / 2
                pos_flag = 3

            else:  # cases == 1

                point = (pa + pb) / 2
                pos = (qa + qb) / 2
                pos_flag = 0

            point_backup = None
            pos_backup = None
            pos_backup_flag = -1

        elif condition == 2:

            # yvpp

            if cases == 0:

                point = (pb + pd) / 2
                pos = (qb + qd) / 2
                pos_flag = 2

            else:  # cases == 1

                point = (pa + pc) / 2
                pos = (qa + qc) / 2
                pos_flag = 1

            point_backup = None
            pos_backup = None
            pos_backup_flag = -1

        else:

            # 3vpp

            if cases == 0:

                pl = np.linalg.norm(pa - pc)
                pw = np.linalg.norm(pa - pb)

                if pl > pw:

                    point = (pa + pc) / 2
                    pos = (qa + qc) / 2
                    pos_flag = 1

                    point_backup = (pa + pb) / 2
                    pos_backup = (qa + qb) / 2
                    pos_backup_flag = 0

                else:

                    point = (pa + pb) / 2
                    pos = (qa + qb) / 2
                    pos_flag = 0

                    point_backup = (pa + pc) / 2
                    pos_backup = (qa + qc) / 2
                    pos_backup_flag = 1


            elif cases == 1:

                pl = np.linalg.norm(pb - pd)
                pw = np.linalg.norm(pa - pb)

                if pl > pw:

                    point = (pb + pd) / 2
                    pos = (qb + qd) / 2
                    pos_flag = 2

                    point_backup = (pa + pb) / 2
                    pos_backup = (qa + qb) / 2
                    pos_backup_flag = 0

                else:

                    point = (pa + pb) / 2
                    pos = (qa + qb) / 2
                    pos_flag = 0

                    point_backup = (pb + pd) / 2
                    pos_backup = (qb + qd) / 2
                    pos_backup_flag = 2

            elif cases == 2:

                pl = np.linalg.norm(pa - pc)
                pw = np.linalg.norm(pc - pd)

                if pl > pw:

                    point = (pa + pc) / 2
                    pos = (qa + qc) / 2
                    pos_flag = 1

                    point_backup = (pc + pd) / 2
                    pos_backup = (qc + qd) / 2
                    pos_backup_flag = 3

                else:

                    point = (pc + pd) / 2
                    pos = (qc + qd) / 2
                    pos_flag = 3

                    point_backup = (pa + pc) / 2
                    pos_backup = (qa + qc) / 2
                    pos_backup_flag = 1

            else:  # cases == 3

                pl = np.linalg.norm(pb - pd)
                pw = np.linalg.norm(pc - pd)

                if pl > pw:

                    point = (pb + pd) / 2
                    pos = (qb + qd) / 2
                    pos_flag = 2

                    point_backup = (pc + pd) / 2
                    pos_backup = (qc + qd) / 2
                    pos_backup_flag = 3

                else:

                    point = (pc + pd) / 2
                    pos = (qc + qd) / 2
                    pos_flag = 3

                    point_backup = (pb + pd) / 2
                    pos_backup = (qb + qd) / 2
                    pos_backup_flag = 2

        states.pos = pos
        states.pos_flag = pos_flag
        states.pos_backup = pos_backup
        states.pos_backup_flag = pos_backup_flag

        states.pos_uc = self.calculate_pos_uc(
            camera_model, map_model, point, p_uc, pos)
        if point_backup is not None:
            states.pos_backup_uc = self.calculate_pos_uc(
                camera_model, map_model, point_backup, p_uc, pos_backup)

        motion, vel_n = self.compute_robust_motion(
            instance, k_min, k_max, motion_th)
        # print('motion: {}, vel_n: {}'.format(motion, vel_n))
        pvx = point - motion

        # q0_lmap = camera_model.transform_point_image_to_lmap(point)
        # qvx_lmap = camera_model.transform_point_image_to_lmap(pvx)

        # q0 = map_model.transform_points_lmap_to_xyz(q0_lmap.reshape((3, 1))).flatten()
        # qvx = map_model.transform_points_lmap_to_xyz(qvx_lmap.reshape((3, 1))).flatten()

        q0 = camera_model.transform_point_image_to_ground(point)
        qvx = camera_model.transform_point_image_to_ground(pvx)

        # CAUTION: It is assumed that qvx and q0 are all in metric unit.
        # Recovering the scale in calibration is crucial!
        vel = (q0 - qvx) / self.frame_interval / vel_n

        qvx_uc = self.calculate_pos_uc(camera_model, map_model, pvx, pv_uc, qvx)
        vel_uc = qvx_uc / self.frame_interval / vel_n

        states.vel = vel
        states.vel_n = vel_n

        states.vel_uc = vel_uc

    def calculate_pos_uc(self, camera_model, map_model, point, p_uc, pos):
        '''
        Calculate the position uncertainty using the camera-ground model.
        '''

        p_uc1 = point + p_uc
        p_uc2 = point - p_uc

        # q_uc1_lmap = camera_model.transform_point_image_to_lmap(p_uc1)
        # q_uc2_lmap = camera_model.transform_point_image_to_lmap(p_uc2)

        # q_uc1 = map_model.transform_points_lmap_to_xyz(q_uc1_lmap.reshape((3, 1))).flatten()
        # q_uc2 = map_model.transform_points_lmap_to_xyz(q_uc2_lmap.reshape((3, 1))).flatten()

        q_uc1 = camera_model.transform_point_image_to_ground(p_uc1)
        q_uc2 = camera_model.transform_point_image_to_ground(p_uc2)

        n_uc1 = np.linalg.norm(q_uc1 - pos)
        n_uc2 = np.linalg.norm(q_uc2 - pos)

        pos_uc = (n_uc1 + n_uc2) / 2

        return pos_uc

    def calculate_dim_uc(self, camera_model, instance, states):
        '''
        Calculate dimension uncertainty.
        
        Note that currently these are mainly based on guessing.
        '''

        pos_uc = states.pos_uc
        bb3d = instance.bb3d

        situation = bb3d.situation

        if situation == 1:

            # xvpp
            dim_uc = (pos_uc * 10, pos_uc, pos_uc * 4)

        elif situation == 2:

            # yvpp
            dim_uc = (pos_uc, pos_uc * 10, pos_uc * 4)

        else:

            # 3vpp
            dim_uc = (pos_uc * 4, pos_uc * 4, pos_uc * 4)

        return dim_uc

    def compute_robust_motion(self, instance, k_min, k_max, motion_th):
        '''
        Compute a more robust motion by tracking back for a few frames.
        '''

        motion = instance.motion.copy()
        n = 1

        for i in range(k_max - 1):

            instance = instance.prev

            if instance is not None:

                if instance.motion_flag > 0:
                    motion += instance.motion
                    n += 1

            else:

                break

            if i > k_min - 1 and np.linalg.norm(motion) > motion_th:
                break

        return motion, n

    def infer_3d_states(self, instance, instance_prev, states_3d, states_3d_prev):
        '''
        Infer the 3D states if it cannot be computed directly for this instance.
        '''

        motion_flag = instance.motion_flag
        motion_2d = instance.motion
        motion_flag_prev = instance_prev.motion_flag
        motion_2d_prev = instance_prev.motion

        states_3d.heading = states_3d_prev.heading

        if motion_flag <= 1:
            states_3d.vel = np.zeros(3)
        else:
            states_3d.vel = states_3d_prev.vel

        states_3d.vel_n = states_3d_prev.vel_n
        states_3d.vel_uc = states_3d_prev.vel_uc

        states_3d.pos = states_3d_prev.pos + states_3d.vel * self.frame_interval
        states_3d.pos_flag = states_3d_prev.pos_flag

        if states_3d_prev.pos_backup is None:
            states_3d.pos_backup = None
        else:
            states_3d.pos_backup = states_3d_prev.pos_backup + states_3d.vel * self.frame_interval
        states_3d.pos_backup_flag = states_3d_prev.pos_backup_flag

        states_3d.pos_uc = states_3d_prev.pos_uc
        states_3d.pos_backup_uc = states_3d_prev.pos_backup_uc

        states_3d.dim = states_3d_prev.dim
        states_3d.dim_uc = states_3d_prev.dim_uc

        pass

    def estimate_vehicle_states(self, frame_id, instances, instances_states_3d):
        '''
        Estimate vehicle position and velocity states (Kalman filtering).
        '''

        remove_th = 30
        box_size_th = 200

        reassociate_distance = 10

        v1 = 60
        v2 = 20

        # v1 = self.frame_height // 6

        # Run the predition step of the filter for each current vehicle.

        for vid, vehicle in self.current_vehicles.items():
            vehicle.state_predict(self.frame_interval)

        n = len(instances)

        # Associate instance to current vehicles using the tracking results.

        instance_xx = []
        states_3d_xx = []

        for instance, states_3d in zip(instances, instances_states_3d):

            # if instance.instance_flag == 0:
            #     continue

            # box_outside_view = self.check_box_outside_view(instance.box)

            # if box_outside_view:
            #     continue

            instance_prev = instance.prev

            if instance_prev is not None and instance_prev.vehicle is not None:

                # This instance can be linked to an existing vehicle.

                vehicle = instance_prev.vehicle
                instance.vehicle = vehicle
                vehicle.instance = instance

                # Run the update step of the filter for this vehicle.
                if instance.states_3d is not None:
                    vehicle.state_update(instance, states_3d)
                else:
                    if not self.check_box_outside_view(instance.box, v1=v1, v2=v2):
                        vehicle.prediction_iteration -= 1


            else:

                instance_xx.append(instance)
                states_3d_xx.append(states_3d)

        # for instance in instances:

        #     if instance.vehicle is not None:
        #         print('\t', instance.tracking_id, instance.vehicle.vid)
        #     else:
        #         print('\t', instance.tracking_id, -1)

        # Vehicles do not dissappear suddenly and appear from nowhere.
        # Some instance might not be correctly detected previously.
        # Now try to associate those instances that are not associated
        # using the tracking results.

        # CAUTION: There might be wrong associations!!!

        for instance, states_3d in zip(instance_xx, states_3d_xx):

            if instance.instance_flag == 0:
                continue

            box_outside_view = self.check_box_outside_view(instance.box)

            if box_outside_view:
                continue

            # CAUTION: An instance may not have 3D states computed yet.
            if instance.states_3d is None:
                continue

            found = False
            distance_last = 1e6
            heading_sim_last = 0
            vehicle_last = None
            associated = False

            # Check whether this instance can be associated to a vehicle
            # that is not detected in the previous frame.
            for vid, vehicle in self.current_vehicles.items():

                # CAUTION: If vehicle.prediction_iteration == 0, it must
                # have been associated to an instance already!
                if vehicle.prediction_iteration == 0 or vehicle.instance is not None:
                    continue

                distance, direction_sim, heading_sim \
                    = vehicle.check_pos_vel_heading_similarity(states_3d)

                if distance < distance_last:
                    distance_last = distance
                    vehicle_last = vehicle
                    heading_sim_last = heading_sim
                    found = True

                    # print('\t\t', instance.tracking_id, vid, distance, direction_sim)

            if found and distance_last < reassociate_distance:

                if heading_sim_last > 0.8:

                    instance.vehicle = vehicle_last

                    # Run the update step of the filter for this vehicle.
                    if instance.states_3d is not None:
                        instance.vehicle.state_update(instance, states_3d)

                else:

                    # CAUTION: Wrong heading calculation!
                    instance.states_3d = None
                    instance.bb3d = None

                associated = True

            # If this instance is still not associated with any existing
            # vehicle, create new vehicle for it as needed.
            if not associated:

                box_size = calculate_box_size(instance.box)

                if box_size > box_size_th and instance.bb3d is not None:
                    vid = self.vid
                    self.vid += 1

                    vehicle = Vehicle(vid)

                    vehicle.state_init(instance, states_3d)

                    instance.vehicle = vehicle

                    self.current_vehicles[vid] = vehicle

        # If a vehicle has not been observed for a while, remove it from the 
        # list of current vehicles. 

        removed_vids = []
        removed_vehicles = []

        for vid, vehicle in self.current_vehicles.items():

            if vehicle.prediction_iteration > remove_th:
                removed_vids.append(vid)
                removed_vehicles.append(vehicle)

        for vid in removed_vids:
            self.current_vehicles.pop(vid)

        self.removed_vehicles = removed_vehicles

    def save_data_file(self, fn, data, fmt, is_integer):

        if fmt == 'csv':
            if is_integer:
                np.savetxt(fn + '.csv', data, delimiter=',', fmt='%d')
            else:
                np.savetxt(fn + '.csv', data, delimiter=',', fmt='%.6f')
        elif fmt == 'npy':
            np.save(fn, data)
        else:
            raise ValueError('No such format: ' + fmt)

    def load_data_file(self, fn, fmt, is_integer):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if fmt == 'csv':
                if is_integer:
                    array = np.loadtxt(fn + '.csv', delimiter=',', dtype=np.int)
                else:
                    array = np.loadtxt(fn + '.csv', delimiter=',')
            elif fmt == 'npy':
                array = np.load(fn + '.npy')
            else:
                raise ValueError('No such format: ' + fmt)

        # CAUTION: If it is saved in CSV format, it may be an empty file.
        if len(array.shape) == 0 or array.shape[0] == 0:
            return None

        if len(array.shape) == 1:
            m = array.shape[0]
            array = array.reshape((1, m))

        if len(array.shape) != 2:
            raise ValueError('Invalid array shape: ' + str(array.shape))

        return array

    def save_instances(self, conf, frame_id, instances, fmt):
        '''
        Save per-frame tracker instances to files.
        
        '''

        n = len(instances)
        m = Instance.m

        ss = np.zeros((n, m))

        for i, instance in enumerate(instances):
            ss[i] = instance.to_array()

            fn_mask = conf.get_fn_per_frame_per_instance(
                'instance', fmt, 'mask', frame_id, i)
            fn_contour = conf.get_fn_per_frame_per_instance(
                'instance', fmt, 'contour', frame_id, i)
            fn_sof = conf.get_fn_per_frame_per_instance(
                'instance', fmt, 'sof', frame_id, i)

            mask_local = instance.get_mask_local()
            contour = instance.contour
            corners = instance.corners_to_prev_inliers
            sofs = instance.sofs_to_prev_inliers

            corners_sofs = np.concatenate((corners, sofs), axis=1)

            self.save_data_file(fn_mask, mask_local, fmt, is_integer=True)
            self.save_data_file(fn_contour, contour, fmt, is_integer=True)
            self.save_data_file(fn_sof, corners_sofs, fmt, is_integer=False)

        fn_instances = conf.get_fn_per_frame('tracking', fmt, 'tracking', frame_id)

        self.save_data_file(fn_instances, ss, fmt, is_integer=False)

    def load_instances(self, conf, frame_id, fmt):

        instances = []

        fn_instances = conf.get_fn_per_frame('tracking', fmt, 'tracking', frame_id)
        array = self.load_data_file(fn_instances, fmt, is_integer=False)

        if array is None:
            n = 0
        else:
            n = array.shape[0]

        for i in range(n):

            instance = Instance(-1, None, None, None, None, None, None)
            instance.from_array(array[i])

            fn_mask = conf.get_fn_per_frame_per_instance(
                'instance', fmt, 'mask', frame_id, i)
            fn_contour = conf.get_fn_per_frame_per_instance(
                'instance', fmt, 'contour', frame_id, i)
            fn_sof = conf.get_fn_per_frame_per_instance(
                'instance', fmt, 'sof', frame_id, i)

            mask_local = self.load_data_file(fn_mask, fmt, is_integer=True)
            contour = self.load_data_file(fn_contour, fmt, is_integer=True)
            corners_sofs = self.load_data_file(fn_sof, fmt, is_integer=False)

            if corners_sofs is not None:
                corners = corners_sofs[:, 0:2]
                sofs = corners_sofs[:, 2:4]
            else:
                corners = np.zeros((0, 2))
                sofs = np.zeros((0, 2))

            instance.set_mask_local(mask_local, self.frame_width, self.frame_height)
            instance.contour = contour
            instance.corners_to_prev_inliers = corners
            instance.sofs_to_prev_inliers = sofs

            instances.append(instance)

        return instances

    def save_instance_instance_association(self, conf, frame_id,
                                           instances, instances_prev, fmt):

        fn_c2p = conf.get_fn_per_frame(
            'tracking', fmt, 'c2p', frame_id)
        fn_p2c = conf.get_fn_per_frame(
            'tracking', fmt, 'p2c', frame_id)

        n_c2p = len(instances)
        n_p2c = len(instances_prev)

        array_c2p = np.zeros((n_c2p, 2), dtype=np.int)
        array_p2c = np.zeros((n_p2c, 2), dtype=np.int)

        for i, instance in enumerate(instances):

            array_c2p[i, 0] = i

            if instance.prev is None:
                array_c2p[i, 1] = -1
            else:
                array_c2p[i, 1] = instance.prev.tracking_id

        for j, instance_prev in enumerate(instances_prev):

            array_p2c[j, 0] = j

            if instance_prev.next is None:
                array_p2c[j, 1] = -1
            else:
                array_p2c[j, 1] = instance_prev.next.tracking_id

        self.save_data_file(fn_c2p, array_c2p, fmt, is_integer=True)
        self.save_data_file(fn_p2c, array_p2c, fmt, is_integer=True)

    def load_instance_instance_association(self, conf, frame_id,
                                           instances, instances_prev, fmt):

        fn_c2p = conf.get_fn_per_frame(
            'tracking', fmt, 'c2p', frame_id)
        fn_p2c = conf.get_fn_per_frame(
            'tracking', fmt, 'p2c', frame_id)

        array_c2p = self.load_data_file(fn_c2p, fmt, is_integer=True)
        array_p2c = self.load_data_file(fn_p2c, fmt, is_integer=True)

        if array_c2p is None or array_p2c is None:
            return

        for i, instance in enumerate(instances):

            idx_i = array_c2p[i, 1]
            if idx_i >= 0:
                instance.prev = instances_prev[idx_i]

        for j, instance_prev in enumerate(instances_prev):

            idx_j = array_p2c[j, 1]
            if idx_j >= 0:
                instance_prev.next = instances[idx_j]

    def save_vehicles(self, conf, frame_id, current_vehicles, fmt):
        '''
        Save the states of current vehicles to files.
        
        '''

        fn_vehicle = conf.get_fn_per_frame(
            'vehicle', fmt, 'vehicle', frame_id)

        n = len(current_vehicles)
        m = Vehicle.m

        array = np.zeros((n, m))

        for i, (vid, vehicle) in enumerate(current_vehicles.items()):
            array[i] = vehicle.to_array(i)

        self.save_data_file(fn_vehicle, array, fmt, is_integer=False)

    def load_vehicles(self, conf, frame_id, current_vehicles, fmt):

        fn_vehicle = conf.get_fn_per_frame(
            'vehicle', fmt, 'vehicle', frame_id)

        array = self.load_data_file(fn_vehicle, fmt, is_integer=False)

        # CAUTION: If the array is empty, we still need to process 
        # removed vehicles.
        if array is None:
            n = 0
        else:
            n = array.shape[0]

        current_vids = set()

        for i in range(n):

            vid = int(round(array[i, 1]))
            current_vids.add(vid)

            if vid in current_vehicles:

                vehicle = current_vehicles[vid]
                vehicle.from_array(array[i])

            else:

                vehicle = Vehicle(vid)
                vehicle.from_array(array[i])

                current_vehicles[vid] = vehicle

        removed_vids = []
        removed_vehicles = []

        for vid, vehicle in current_vehicles.items():

            if vid not in current_vids:
                removed_vids.append(vid)
                removed_vehicles.append(vehicle)

        for vid in removed_vids:
            self.current_vehicles.pop(vid)

        self.removed_vehicles = removed_vehicles

    def save_vehicle_instance_association(self, conf, frame_id,
                                          current_vehicles, instances, fmt):

        fn_v2i = conf.get_fn_per_frame(
            'vehicle', fmt, 'v2i', frame_id)
        fn_i2v = conf.get_fn_per_frame(
            'vehicle', fmt, 'i2v', frame_id)

        n = len(current_vehicles)
        m = len(instances)

        array_v2i = np.zeros((n, 2), dtype=np.int)
        array_i2v = np.zeros((m, 2), dtype=np.int)

        for i, (vid, vehicle) in enumerate(current_vehicles.items()):

            array_v2i[i, 0] = vid

            if vehicle.instance is None:
                array_v2i[i, 1] = -1
            else:
                array_v2i[i, 1] = vehicle.instance.tracking_id

        for j, instance in enumerate(instances):

            array_i2v[j, 0] = j

            if instance.vehicle is None:
                array_i2v[j, 1] = -1
            else:
                array_i2v[j, 1] = instance.vehicle.vid

        self.save_data_file(fn_v2i, array_v2i, fmt, is_integer=True)
        self.save_data_file(fn_i2v, array_i2v, fmt, is_integer=True)

    def load_vehicle_instance_association(self, conf, frame_id,
                                          current_vehicles, instances, fmt):

        fn_v2i = conf.get_fn_per_frame(
            'vehicle', fmt, 'v2i', frame_id)
        fn_i2v = conf.get_fn_per_frame(
            'vehicle', fmt, 'i2v', frame_id)

        array_v2i = self.load_data_file(fn_v2i, fmt, is_integer=True)
        array_i2v = self.load_data_file(fn_i2v, fmt, is_integer=True)

        if array_v2i is None or array_i2v is None:
            return

        n = array_v2i.shape[0]
        m = array_i2v.shape[0]

        for i in range(n):

            vid = array_v2i[i, 0]
            idx_i = array_v2i[i, 1]

            if vid in current_vehicles and idx_i != -1:
                current_vehicles[vid].instance = instances[idx_i]
            else:
                # print('BUG1', vid, array_v2i, array_i2v)
                pass

        for j in range(m):

            idx_j = array_i2v[j, 0]
            vid = array_i2v[j, 1]

            if vid in current_vehicles and idx_j != -1:
                instances[idx_j].vehicle = current_vehicles[vid]
            else:
                # print('BUG2', vid, array_v2i, array_i2v)
                pass

    def save_all(self, conf, frame_id, fmt):

        self.save_instances(conf, frame_id, self.instances, fmt)
        self.save_instance_instance_association(
            conf, frame_id, self.instances, self.instances_prev, fmt)
        self.save_vehicles(
            conf, frame_id, self.current_vehicles, fmt)
        self.save_vehicle_instance_association(
            conf, frame_id, self.current_vehicles, self.instances, fmt)

    def track_iteration_load_and_replay(self, conf, frame_id, frame_current, fmt):

        self.frame_current = frame_current

        # Load instances and instance-instance association.

        self.instances = self.load_instances(conf, frame_id, fmt)
        instances = self.instances
        instances_prev = self.instances_prev
        self.load_instance_instance_association(
            conf, frame_id, instances, instances_prev, fmt)

        # Load vehicles and vehicle-instance association.

        current_vehicles = self.current_vehicles

        self.load_vehicles(conf, frame_id, current_vehicles, fmt)

        self.load_vehicle_instance_association(
            conf, frame_id, current_vehicles, instances, fmt)

    def track_iteration_finish(self):

        self.frame_prev = self.frame_current

        if self.instances_prev is not None:
            for instance_prev in self.instances_prev:
                instance_prev.mask = None

        self.instances_prev = self.instances

        pass

    # ----------------- multiple tracker vehicle association ---------------

    def index_vehicle_location(self):
        '''
        Build a grid table to index the location of all current vehicles
        in the global map coordinate frame.
        '''

        mm = self.map_model

        vehicle_items = self.current_vehicles.items()
        location_indices = self.location_indices
        removed_vehicles = self.removed_vehicles

        n = len(vehicle_items)

        locations = np.zeros((n, 2))

        for vid, vehicle in vehicle_items:

            vehicle.compute_states_global(mm)

            center_gmap = vehicle.center_gmap

            i, j = self.calculate_location_indices(center_gmap)

            if vehicle.index is not None:

                ii, jj = vehicle.index
                if ii != i or jj != j:
                    location_indices[ii][jj].remove(vehicle)
                    location_indices[i][j].append(vehicle)
                    vehicle.index = (i, j)

            else:

                vehicle.index = (i, j)
                location_indices[i][j].append(vehicle)

        for vehicle in removed_vehicles:
            ii, jj = vehicle.index
            location_indices[ii][jj].remove(vehicle)

    def calculate_location_indices(self, center_gmap):
        '''
        Calculate the indices of a cell in the grid hash table given the 
        location.
        '''

        mm = self.map_model
        x_max, y_max = mm.get_global_map_size()
        x_min = 0
        y_min = 0

        n = self.idx_n
        m = self.idx_m

        step_x = (x_max - x_min) / n
        step_y = (y_max - y_min) / m

        x = center_gmap[0]
        y = center_gmap[1]

        i = int((x - x_min) / step_x)
        j = int((y - y_min) / step_y)

        if i < 0:
            i = 0
        if i > n - 1:
            i = n - 1

        if j < 0:
            j = 0
        if j > m - 1:
            j = m - 1

        return i, j

    def search_nearest_vehicle(self, pos_request):
        '''
        Search the nearest vehicle given the requested center position.
        '''

        n = self.idx_n
        m = self.idx_m
        i, j = self.calculate_location_indices(pos_request)

        found = False
        dist_min = 1e6
        vehicle_found = None

        cells = [(i, j), (i, j - 1), (i - 1, j), (i, j + 1), (i + 1, j),
                 (i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1), (i + 1, j + 1)]

        for ii, jj in cells:

            if ii < 0 or ii >= n:
                continue
            if jj < 0 or jj >= m:
                continue

            for vehicle in self.location_indices[ii][jj]:

                center = vehicle.center_gmap
                dist = np.linalg.norm(pos_request - center)
                if dist < dist_min or not found:
                    found = True
                    dist_min = dist
                    vehicle_found = vehicle

        return found, dist_min, vehicle_found


def init_pipeline(conf, init_frame_id, need_rectification, replay, extension):
    folder = conf.folder
    prefix = conf.prefix
    postfix = conf.postfix

    calibration_folder = conf.get_calibration_folder()

    # The background image is one frame from the video with no vehicles
    # inside the camera field of view. This image is used in manual 
    # calibration, and the program remove the slight camera shake such
    # that the rectified image aligns with this background image.

    # The background mask is essentially a mask of ROI. Inside this ROI
    # corner features will be detected. Mainly this ROI eliminate the
    # a narrow region in the top of the frame where the timestamp is
    # displayed.

    bg_fn = conf.get_bg_fn()
    mask_fn = conf.get_mask_fn()
    # print(bg_fn, mask_fn)

    frame_bg = cv2.imread(bg_fn)
    mask_bg = cv2.imread(mask_fn)
    #frame_bg = cv2.resize(frame_bg, (635,1130))
    #mask_bg = cv2.resize(mask_bg, (635,1130))
    frame_width = frame_bg.shape[1]
    frame_height = frame_bg.shape[0]

    # print(bg_fn)

    assert frame_bg is not None and mask_bg is not None

    # The map image is a screenshot of Google Map satellite image.

    map_local_fn = conf.get_map_local_fn()
    map_global_fn = conf.get_map_global_fn()

    map_local = cv2.imread(map_local_fn)
    map_global = cv2.imread(map_global_fn)

    # print(map_local_fn)

    assert map_local is not None and map_global is not None

    # The camera model contains the camera parameters and 
    # the ground-to-frame homography.

    camera_model = Camera2DGroundModel()
    camera_model.load_calib_para(calibration_folder, prefix)

    # The map model converts xyz-coordinates to esd-coordinates.

    map_model = MapModel(map_local, map_global)
    map_model.load_map_para(calibration_folder, prefix)

    # Initialize the subcomponents of the tracker.

    csr = CameraShakeRectifier(frame_bg, mask_bg)

    klt_tracker = KLTOpticalFlowTracker(mask_bg)

    detection_folder = folder + '/detection/' + prefix + postfix

    if not replay:
        detector = DetectorTorchvisionMaskRCNN(detection_folder, prefix, postfix)
    else:
        detector = DetectorWithSavedResults(detection_folder, prefix, postfix)

    # Initialize the tracker.

    tracker = Tracker(map_model, klt_tracker, detector, camera_model,
                      frame_width=frame_width, frame_height=frame_height,
                      h_margin=15, v_margin=10, fps=30)

    tracker.conf = conf

    mask_bg_gray = cv2.cvtColor(mask_bg, cv2.COLOR_BGR2GRAY)
    mask_bg_th = mask_bg_gray > 0
    tracker.mask_bg = mask_bg_th

    # Now open the video file. Skip a few frames as needed.

    video_fn = conf.get_video_fn(extension)

    cap = cv2.VideoCapture(video_fn)
    #sys.exit()
    assert cap.isOpened()

    for i in range(init_frame_id):
        cap.read()
    # # cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame_id-1)
    # total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # factor = (cap.get(cv2.CAP_PROP_FPS)/30.0)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # vis_video_frames = total_frames/factor
    #
    # print(f'init_frame_id : {init_frame_id}, '
    #       f'#total: {total_frames}, '
    #       f'fps: {fps} factor: {factor}, '
    #       f'vis_frame_count: {vis_video_frames}')
    #
    # print(f'factor: {factor}, init_frame_id: {init_frame_id}, factored_val: {init_frame_id * factor}')
    #
    # cap.set(cv2.CAP_PROP_POS_FRAMES, int(init_frame_id-1) * factor)

    ret, first_frame = cap.read()
    #first_frame = cv2.resize(first_frame, (635,1130))
    if need_rectification:
        first_frame = csr.rectify(first_frame)

    return tracker, cap, csr, map_local, map_global, first_frame


def pipeline_multiple_tracker_association(trackers):
    dist_th = 40
    heading_sim_th = 0.85

    m = len(trackers)

    # Build index for current vehicles
    for tracker in trackers:
        tracker.index_vehicle_location()

    associations = []
    # Associate vehicles
    for i, tracker_i in enumerate(trackers):

        vehicles_i = tracker_i.current_vehicles
        n_i = len(vehicles_i)
        association_i = np.zeros((n_i, m), dtype=np.int)
        association_i.fill(-1)

        for k, (vid_i, vehicle_i) in enumerate(vehicles_i.items()):

            vehicle_i.association = []

            for j, tracker_j in enumerate(trackers):

                if i == j:
                    continue

                center_request = vehicle_i.center_gmap
                hp_request = vehicle_i.hp_gmap

                found, dist, vj = tracker_j.search_nearest_vehicle(center_request)

                if found and dist < dist_th:

                    sim = vj.compare_heading_global(center_request, hp_request)

                    if sim > heading_sim_th:
                        association_i[k, i] = vid_i
                        association_i[k, j] = vj.vid

                        vehicle_i.association.append(vj)

                        print('%d:%d <-- %2.1f --> %d:%d' % (i, vid_i, dist, j, vj.vid))

        associations.append(association_i)

    return associations


def pipeline_save_multiple_tracker_association(trackers, associations, frame_ids, fmt):
    for tracker, association_i, frame_id in zip(trackers, associations, frame_ids):
        conf = tracker.conf
        fn_mva = conf.get_fn_per_frame('multiple_view', fmt, 'vva', frame_id)
        tracker.save_data_file(fn_mva, association_i, fmt, is_integer=True)


def pipeline_load_multiple_tracker_association(trackers, frame_ids, fmt):
    associations = []

    for tracker, frame_id in zip(trackers, frame_ids):
        conf = tracker.conf
        fn_mva = conf.get_fn_per_frame('multiple_view', fmt, 'vva', frame_id)

        association_i = tracker.load_data_file(fn_mva, fmt, is_integer=True)

        associations.append(association_i)

        # TODO: update vehicle_i.association

    return associations


def pipeline_multiple_tracker_fusion(trackers, associations):
    for i, tracker_i in enumerate(trackers):

        vehicles_i = tracker_i.current_vehicles

        for vid_i, vehicle_i in vehicles_i.items():
            vehicle_i.fuse_states()

    pass


def pipeline_statistics_one_iteration(tracker, stat, frame_id, c):
    h_line = 97

    instances = tracker.instances

    ob = 0

    d_tp = 0
    d_fn = 0

    t_tp = 0
    t_fn = 0
    t_mme = 0

    for instance in instances:

        y = instance.center[1]

        if y > h_line:
            ob += 1

    if c & 0xFF == ord('1'):

        # Missing one in detection (also one in tracking).

        d_tp = ob
        t_tp = ob

        ob += 1
        d_fn = 1
        t_fn = 1

    elif c & 0xFF == ord('2'):

        # Missing one in tracking (it is detected).

        d_tp = ob
        t_tp = ob

        t_fn = 1

    elif c & 0xFF == ord('3'):

        d_tp = ob
        t_tp = ob - 2

        t_mme += 2

    else:

        d_tp = ob
        t_tp = ob

    stat.ob += ob

    stat.d_tp += d_tp
    stat.d_fn += d_fn

    stat.t_tp += t_tp
    stat.t_fn += t_fn

    stat.t_mme += t_mme

    print(frame_id, ob, '\t', d_tp, d_fn, t_tp, t_fn, t_mme, '\t\t',
          stat.ob, '\t', stat.d_tp, stat.d_fn, stat.t_tp, stat.t_fn, stat.t_mme)


def pipeline_visualize(tracker, frame_rect, frame_id, fvis, mvis, rot_map, h_line, writer, bbox_3d, render=False):
    # Generate several copies of the frame and the map for drawing
    # stuff on each of them

    frame_vis1 = frame_rect.copy()
    frame_vis2 = frame_rect.copy()
    frame_vis3 = frame_rect.copy()

    frame_width = frame_rect.shape[1]
    # cv2.line(frame_vis1, (0, h_line), (frame_width, h_line), (255, 0, 0), 2)
    # cv2.line(frame_vis2, (0, h_line), (frame_width, h_line), (255, 0, 0), 2)
    # cv2.line(frame_vis3, (0, h_line), (frame_width, h_line), (255, 0, 0), 2)

    # c1 = (0, 0, 255)
    # c2 = (0, 0, 255)
    # fvis.draw_sof_whole_frame(frame_vis1, 
    #     tracker.corners_this, tracker.corners_prev, c1, c2)    

    fvis.draw_vehicle_detection(frame_vis1, tracker.instances)

    fvis.draw_vehicle_segmentation(frame_vis2, tracker.instances)

    fvis.draw_bb3ds_on_frame(frame_vis3, tracker.instances)
    for i in range(len(bbox_3d)):
        cv2.circle(frame_vis3, (int(bbox_3d[i][0]), int(bbox_3d[i][1])), 1, (0,0,255), -1)
    # Show the frame and the map with annotations on the screen.
    w = 1280
    h = 720
    frame_resize = (w, h)
    frame_vis1 = cv2.resize(frame_vis1, frame_resize)
    frame_vis2 = cv2.resize(frame_vis2, frame_resize)
    frame_vis3 = cv2.resize(frame_vis3, frame_resize)

    # frame_vis1[0:36, 0:] = 0
    # frame_vis2[0:36, 0:] = 0
    # frame_vis3[0:36, 0:] = 0

    cv2.putText(frame_vis1, 'frame #%d (optical flow)' % frame_id, (60, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame_vis2, 'frame #%d (mask)' % frame_id, (60, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame_vis3, 'frame #%d (3D bounding box)' % frame_id, (60, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    # Generate several copies of the frame and the map for drawing
    # stuff on each of them
    map_local = tracker.map_model.local_map_image
    map_vis = map_local.copy()

    # Draw the field of view of the camera as a cone shaped polygon
    # on the map.
    FOV_ROI = (0, h_line, tracker.frame_width, tracker.frame_height)
    # mvis.draw_camera_FOV_on_map(map, camera_model, map_model, FOV_ROI)
    mvis.draw_camera_FOV_on_map(map_vis, tracker.camera_model, tracker.map_model, FOV_ROI)

    #         mvis.draw_bb3ds_on_map(map_vis1, tracker.map_model,
    #             tracker.instances, tracker.instances_3d_states, frame_id)
    #         mvis.draw_observed_states_on_map(map_vis2, tracker.map_model,
    #             tracker.instances, tracker.instances_3d_states, tracker.frame_interval)
    mvis.draw_vehicle_states_on_map(map_vis, tracker.map_model,
                                    tracker.current_vehicles, tracker.frame_interval)

    # Show the warped frame as the bird eye view.
    # frame_warp = tracker.camera_model.warp_image_to_map(
    #    frame_rect, map_local.shape[1], map_local.shape[0])

    # Currently, the map is 2560 * 1440, resize it to 1280 * 720.
    map_resize = (map_local.shape[1] // 2, map_local.shape[0] // 2)
    map_vis = cv2.resize(map_vis, map_resize)

    if rot_map:
        map_vis = cv2.rotate(map_vis, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # cv2.imshow('map_original',map_original)
    # cv2.imshow('map_raw',map_vis1)
    # cv2.imshow('map_uncertainty',map_vis2)

    # cv2.putText(map_vis, 'ASU Active Perception Group', (100, 40),
    #    cv2.FONT_HERSHEY_SIMPLEX, 1, (127, 127, 127), 2)
    if render:
        cv2.imshow('tracking', frame_vis1)
        cv2.imshow('mask', frame_vis2)
        cv2.imshow('bb3d', frame_vis3)
        cv2.imshow('map_filtered', map_vis)

    # The warped frame has the same resolution as the map. Also resize it.
    # frame_warp = cv2.resize(frame_warp, map_resize)

    # cv2.imshow('warp', frame_warp)

    if writer is not None:
        writer_frame = np.zeros((h * 2, w * 2, 3), np.uint8)
        writer_frame[:h, :w] = frame_vis1
        writer_frame[h:, :w] = frame_vis2
        writer_frame[:h, w:] = frame_vis3
        writer_frame[h:, w:] = map_vis

        writer.write(writer_frame)


def pipeline_map_gmap_visualize(tracker, mvis, map_gmap_vis, direction, color):
    frame = tracker.frame_current

    FOV_ROI = (0, 100, tracker.frame_width, tracker.frame_height)
    mvis.draw_camera_FOV_on_global_map(map_gmap_vis,
                                       tracker.camera_model, tracker.map_model, FOV_ROI, color=color)

    mvis.draw_vehicle_states_on_global_map(map_gmap_vis, tracker.map_model,
                                           tracker.current_vehicles, tracker.frame_interval)

    mh = map_gmap_vis.shape[0]
    mw = map_gmap_vis.shape[1]

    fh = frame.shape[0]
    fw = frame.shape[1]

    fw = fw // 3
    fh = fh // 3

    frame_vis = cv2.resize(frame, (fw, fh))

    if direction == 0:

        map_gmap_vis[0:fh, mw - fw:mw] = frame_vis

    elif direction == 1:

        map_gmap_vis[mh - fh:mh, 0:fw] = frame_vis

    elif direction == 2:

        map_gmap_vis[mh - fh:mh, mw - fw:mw] = frame_vis

    elif direction == 3:

        map_gmap_vis[0:fh, 0:fw] = frame_vis

    else:

        pass

    cv2.imshow('map_gmap', map_gmap_vis)


def pipeline_map_gmap_fuse_visualize(tracker, mvis, map_gmap_vis, direction, color):
    frame = tracker.frame_current

    FOV_ROI = (0, 100, tracker.frame_width, tracker.frame_height)
    mvis.draw_camera_FOV_on_global_map(map_gmap_vis,
                                       tracker.camera_model, tracker.map_model, FOV_ROI, color=color)

    mvis.draw_fused_states_on_global_map(map_gmap_vis, tracker.map_model,
                                         tracker.current_vehicles, tracker.frame_interval)

    mh = map_gmap_vis.shape[0]
    mw = map_gmap_vis.shape[1]

    fh = frame.shape[0]
    fw = frame.shape[1]

    fw = fw // 3
    fh = fh // 3

    frame_vis = cv2.resize(frame, (fw, fh))

    if direction == 0:

        map_gmap_vis[0:fh, mw - fw:mw] = frame_vis

    elif direction == 1:

        map_gmap_vis[mh - fh:mh, 0:fw] = frame_vis

    elif direction == 2:

        map_gmap_vis[mh - fh:mh, mw - fw:mw] = frame_vis

    elif direction == 3:

        map_gmap_vis[0:fh, 0:fw] = frame_vis

    else:

        pass

    cv2.imshow('map_gmap_fuse', map_gmap_vis)


def pipeline_segmentation_joint_visualize(trackers, fvis):
    w = trackers[0].frame_width
    h = trackers[0].frame_height

    shape = (h * 2, w * 2, 3)
    frame_joint = np.zeros(shape, dtype=np.uint8)

    for i, tracker in enumerate(trackers):

        frame = tracker.frame_current
        frame_vis = frame.copy()

        fvis.draw_segmentation(frame_vis, tracker.instances)

        if i == 0:

            frame_joint[0:h, w:w * 2] = frame_vis

        elif i == 1:

            frame_joint[h:h * 2, 0:w] = frame_vis

        elif i == 2:

            frame_joint[h:h * 2, w:w * 2] = frame_vis

        elif i == 3:

            frame_joint[0:h, 0:w] = frame_vis

    frame_joint = cv2.resize(frame_joint, (int(w * 0.8), int(h * 0.8)))

    cv2.imshow('seg_joint', frame_joint)


def pipeline_bb3d_joint_visualize(trackers, fvis):
    w = trackers[0].frame_width
    h = trackers[0].frame_height

    shape = (h * 2, w * 2, 3)
    frame_joint = np.zeros(shape, dtype=np.uint8)

    for i, tracker in enumerate(trackers):

        frame = tracker.frame_current
        frame_vis = frame.copy()

        fvis.draw_bb3ds_on_frame(frame_vis, tracker.instances)

        if i == 0:

            frame_joint[0:h, w:w * 2] = frame_vis

        elif i == 1:

            frame_joint[h:h * 2, 0:w] = frame_vis

        elif i == 2:

            frame_joint[h:h * 2, w:w * 2] = frame_vis

        elif i == 3:

            frame_joint[0:h, 0:w] = frame_vis

    frame_joint = cv2.resize(frame_joint, (int(w * 0.8), int(h * 0.8)))

    cv2.imshow('bb3d_joint', frame_joint)

    pass


def test_tracker(cam_id, track_id, folder, prefixes):
    # replay = True
    replay = False

    # save = False
    save = True

    # save_video = False
    save_video = True

    render = False

    # source video extension (works for .avi, .mpg and .mp4)
    extension = '.mp4'

    # fmt = 'npy'
    fmt = 'csv'

    #need_rectification = False
    need_rectification = True

    # Set up configuration parameters

    direction = cam_id
    postfix = ('_%d' % track_id)


    prefix = prefixes[direction]

    h_lines = [100, 100, 110, 90, 120]
    h_line = h_lines[cam_id]


    conf = TrackerConf(folder, prefix, postfix)
    conf.create_folders(folder, prefix, postfix, fmt)

    # Initialize the visualizer. 

    fvis = FrameVis()
    mvis = MapVis()
    # spvis = SpeedPlotVis(1)

    # Colors for drawing the FOV on the map.
    colors = [(255, 255, 0), (255, 0, 255), (0, 255, 255),
              (255, 0, 0), (0, 255, 0), (0, 0, 255)]

    # Initialize the tracker.

    init_frame_id = 0

    tracker, cap, csr, map_local, map_gmap, first_frame \
        = init_pipeline(conf, init_frame_id, need_rectification, replay, extension=extension)

    # Initialize the video writer, for the processed video
    # ToDO: processed video should be checked for
    if save_video:
        if not os.path.exists(folder + '/processed_video/'):
            os.mkdir(folder + '/processed_video/')
        fn_processed_video = folder + '/processed_video/' + prefix + postfix + '.mp4'
        writer_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #writer_size = (2260, 1270)
        writer_size = (2560, 1440)
        writer = cv2.VideoWriter(fn_processed_video, writer_fourcc, 30, writer_size)
    else:
        writer = None

    # Process the first frame.

    if replay:
        tracker.setup_first_frame_load_and_replay(init_frame_id, first_frame, fmt)
    else:
        tracker.setup_first_frame(init_frame_id, first_frame)
        if save:
            tracker.save_instances(conf, init_frame_id, tracker.instances, fmt)

    tracker.track_iteration_finish()

    stat = TrackerStats()

    # Now process each frame.

    for frame_id in tqdm(range(init_frame_id + 1, 1513)):
        print("FRAME ID: ", frame_id)
        ret, frame = cap.read()
        #frame = cv2.resize(frame, (635, 1130))
        if not ret:
            print('Unable to read a frame from the video!')
            break

        # Remove the camera shake.
        if need_rectification:
            frame = csr.rectify(frame)

        if replay:
            tracker.track_iteration_load_and_replay(conf, frame_id, frame, fmt)
        else:

            tracker.track_iteration_online(frame_id, frame)
            if save:
                tracker.save_all(conf, frame_id, fmt)

        # spvis.update_plot(frame_id, tracker.current_vehicles, tracker.removed_vehicles)

        if cam_id == 2 or cam_id == 3:
            rot_map = True
        else:
            rot_map = False

        # rot_map = False
        f = open(os.path.join(r"/scratch/rparikh4/egonet_old/intermediate_results/","{:06d}".format(frame_id)+'.txt'),"r")
        bbox_3d = []
        for line in f.readlines():
            point = line.split()
            bbox_3d.append(np.array([float(point[0]), float(point[1])]))
            
        pipeline_visualize(tracker, tracker.frame_current, frame_id, fvis,
                           mvis, rot_map, h_line, writer, bbox_3d, render=render)

        # print(frame_id)

        # map_gmap_vis = map_gmap.copy()
        # pipeline_map_gmap_visualize(tracker, mvis, map_gmap_vis, 
        #     direction, colors[direction])

        # if frame_id % 1000 == 0:
        #     print(frame_id, len(tracker.current_vehicles))

        tracker.track_iteration_finish()

        c = cv2.waitKey(1)

        # pipeline_statistics_one_iteration(tracker, stat, frame_id, c)

        if c & 0xFF == ord('q'):
            break

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()


def test_joint_tracker():
    replay = False
    # replay = True

    # save = False
    save = True

    # fmt = 'npy'
    fmt = 'csv'

    frame_width = 1280
    frame_height = 720

    postfix = ('_%d' % 0)

    prefixes = ['westbound', 'eastbound', 'northbound', 'southbound']

    folder = '../avacar_data'

    # Initialize the visualizer.

    fvis = FrameVis()
    mvis = MapVis()

    colors = [(255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 0, 0)]

    frame_ids = []
    nr_skip_frames = [0, 7, 58, 16]
    trackers = []
    caps = []
    csrs = []

    init_frame_id = 0

    for direction in range(4):

        prefix = prefixes[direction]

        conf = TrackerConf(folder, prefix, postfix)

        tracker, cap, csr, map_lmap, map_gmap, first_frame \
            = init_pipeline(conf, init_frame_id)

        # Process the first frame.
        # NOTE: The frame ID of the first frame is 0.

        if replay:
            tracker.setup_first_frame_load_and_replay(init_frame_id, first_frame, fmt)
        else:
            tracker.setup_first_frame(init_frame_id, first_frame)
            if save:
                tracker.save_instances(conf, init_frame_id, tracker.instances, fmt)

        tracker.track_iteration_finish()

        frame_ids.append(init_frame_id + 1)

        # timestamp sync
        for i in tqdm(range(nr_skip_frames[direction])):

            frame_id = frame_ids[direction]
            frame_ids[direction] += 1

            ret, frame = cap.read()
            if not ret:
                print('Unable to read a frame from the video!')
                return

            # Remove the camera shake.
            frame = csr.rectify(frame)

            if replay:
                tracker.track_iteration_load_and_replay(tracker.conf, frame_id, frame, fmt)
            else:

                tracker.track_iteration_online(frame_id, frame)
                if save:
                    tracker.save_all(tracker.conf, frame_id, fmt)

            tracker.track_iteration_finish()

        trackers.append(tracker)
        caps.append(cap)
        csrs.append(csr)

    print('Joint tracker initiation done.')

    for i in tqdm(range(1, 10000-100)):

        # print(i)

        map_gmap_vis = map_gmap.copy()
        map_gmap_vis_fuse = map_gmap.copy()

        for direction in range(4):

            frame_id = frame_ids[direction]
            frame_ids[direction] += 1

            # print('\t%d:%d, ' % (direction, frame_id))

            tracker = trackers[direction]
            cap = caps[direction]
            csr = csrs[direction]

            ret, frame = cap.read()
            if not ret:
                print('Unable to read a frame from the video!')
                return

            # Remove the camera shake.
            frame = csr.rectify(frame)

            if replay:
                tracker.track_iteration_load_and_replay(tracker.conf, frame_id, frame, fmt)
            else:

                tracker.track_iteration_online(frame_id, frame)
                if save:
                    tracker.save_all(tracker.conf, frame_id, fmt)

            tracker.track_iteration_finish()

        # print('\tjoint association')

        associations = pipeline_multiple_tracker_association(trackers)

        # if save:
        pipeline_save_multiple_tracker_association(trackers, associations, frame_ids, fmt)

        pipeline_multiple_tracker_fusion(trackers, associations)

        pipeline_segmentation_joint_visualize(trackers, fvis)
        pipeline_bb3d_joint_visualize(trackers, fvis)

        for direction in range(4):
            tracker = trackers[direction]
            pipeline_map_gmap_visualize(tracker, mvis, map_gmap_vis,
                                        direction, colors[direction])
            pipeline_map_gmap_fuse_visualize(tracker, mvis, map_gmap_vis_fuse,
                                             direction, colors[direction])

        c = cv2.waitKey(-1)

        if c & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
   
    prefixes = ['westbound', 'eastbound', 'northbound', 'southbound']
    
    ## Copy path containing the data (videos and calibration)
    folder = './data'
    
    ## For running tracking in a specific direction and a specific video number
    ## assumes the naming format for eg: southbound_12.mpg
    video_id = 0
    prefix_id = 1
    
    test_tracker(prefix_id, video_id, folder, prefixes)
    
    ## For running tracker in all directions and for all videos
    # for i in range(1, 4):
    #     for j in range(1, 13):
    #         test_tracker(i, j)


    # test_joint_tracker()

    pass
