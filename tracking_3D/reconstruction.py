'''
Created on Jul 1, 2020

@author: duolu
'''

import os
import warnings
warnings.filterwarnings("ignore")


import numpy as np
import cv2

import matplotlib
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from matplotlib.widgets import Slider
from matplotlib.widgets import Button

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    from skimage import measure

import pywavefront

from tracking_3d.calibration import Camera2DGroundModel

class VehicleStatesPerFrame(object):
    '''
    Vehicle states on a single frame.
    '''

    def __init__(self, vid, frame_id, instance_id, box, mask,
        pos, pos_flag, heading, dimension):

        self.vid = vid
        self.frame_id = frame_id
        self.instance_id = instance_id

        self.box = box
        self.mask = mask

        # NOTE: The following are all 3D states.
        self.pos = pos
        self.pos_flag = pos_flag
        self.heading = heading
        self.dimension = dimension

        self.bb3d_flag = None
        self.bb3d_situation = None
        self.bb3d_dim = None


def mask_local_to_global(box, mask_local, frame_size):

    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])

    mask_global = np.zeros((frame_size[1], frame_size[0]), dtype=bool)
    mask_global[y1:y2, x1:x2] = mask_local

    return mask_global


def save_data_file(fn, data, fmt, is_integer):

    if fmt == 'csv':
        if is_integer:
            np.savetxt(fn + '.csv', data, delimiter=',', fmt='%d')
        else:
            np.savetxt(fn + '.csv', data, delimiter=',', fmt='%.6f')
    elif fmt == 'npy':
        np.save(fn, data)
    else:
        raise ValueError('No such format: ' + fmt)



def load_data_file(fn, fmt, is_integer):

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






















def visualize_hist(ax, hist, pps_kp, voxel_size, dimension, dscaler, alpha=1, kp=False):

    ln = voxel_size[0]
    wn = voxel_size[1]
    hn = voxel_size[2]

    l = dimension[0] * dscaler[0]
    w = dimension[1] * dscaler[1]
    h = dimension[2] * dscaler[2]

    pps_kp = pps_kp.copy()
    pps_kp[:, 0] *= l / 2
    pps_kp[:, 1] *= w / 2
    pps_kp[:, 2] *= h

    lnm = ln - 1
    wnm = wn - 1
    #hnm = hn - 1

    qx = l / 2
    qy = w / 2

    qmx = -qx
    qmy = -qy

    xs = np.zeros((ln, wn))
    ys = np.zeros((ln, wn))
    zs = np.zeros((ln, wn))

    for i in range(ln):
        x = qx * (i / lnm) + qmx * (1 - i / lnm)
        for j in range(wn):
            y = qy * (j / wnm) + qmy * (1 - j / wnm)

            xs[i, j] = x
            ys[i, j] = y
            zs[i, j] = hist[i, j] * h
    
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    #voxels_vis = voxels > -1
    #ax.voxels(voxels, edgecolor="k")

    #ax.plot_wireframe(xs, ys, zs, rstride=1, cstride=1)
    ax.plot_surface(xs, ys, zs, alpha=alpha)

    if kp:
        colors = ['b', 'y', 'g', 'm', 'r', 'k']

        for i in range(6):

            j = i * 4
            pps_temp = pps_kp[j:j + 4]

            pps_vis = np.zeros((4, 3))
            pps_vis[0] = pps_temp[0]
            pps_vis[1] = pps_temp[1]
            pps_vis[2] = pps_temp[3]
            pps_vis[3] = pps_temp[2]
            #pps_vis[4] = pps_temp[0]

            ax.plot(pps_vis[:, 0], pps_vis[:, 1], pps_vis[:, 2], color=colors[i])
            ax.scatter(pps_vis[:, 0], pps_vis[:, 1], pps_vis[:, 2], s=20, color=colors[i])

    M = 1
    ax.set_xlim(-M, M)
    ax.set_ylim(-M, M)
    ax.set_zlim(-M, M)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_axis_off()

    if ax is None:
        plt.show()


def visualize_key_points(pps_kp, dimension, dscaler):


    l = dimension[0] * dscaler[0]
    w = dimension[1] * dscaler[1]
    h = dimension[2] * dscaler[2]

    pps_kp[:, 0] *= l / 2
    pps_kp[:, 1] *= w / 2
    pps_kp[:, 2] *= h / 2



    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')


    for i in range(4):

        j = i * 4
        pps_temp = pps_kp[j:j + 4]

        pps_vis = np.zeros((5, 3))
        pps_vis[0] = pps_temp[0]
        pps_vis[1] = pps_temp[1]
        pps_vis[2] = pps_temp[3]
        pps_vis[3] = pps_temp[2]
        pps_vis[4] = pps_temp[0]

        ax.plot(pps_vis[:, 0], pps_vis[:, 1], pps_vis[:, 2])

    lim = 2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')



    plt.show()


def visualize_mesh(voxels):

    voxels_vis = voxels

    # Use marching cubes to obtain the surface mesh of these ellipsoids
    verts, faces, _normals, _values = measure.marching_cubes(voxels_vis, 0, allow_degenerate=False)

    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ln = voxels_vis.shape[0]
    wn = voxels_vis.shape[1]
    hn = voxels_vis.shape[2]

    m = max(ln, max(wn, hn))

    ax.set_xlim(0, m)
    ax.set_ylim(0, m)
    ax.set_zlim(0, m)

    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_zlabel('z')

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_axis_off()


    plt.tight_layout()
    plt.show()


class TrackerRecordHelper(object):

    def __init__(self):

        pass


    def process_one_track(self, folder, camera_name, track_id, fmt):

        tracking_m = 81

        # Load per-frame vehicle and tracking files.

        folder_prefix = folder + '/tracking/'
        ct_str = camera_name + '_%d' % track_id

        folder_vehicle = folder_prefix + 'vehicle_' + fmt + '_' + ct_str
        fn_vehicle_prefix = folder_vehicle + '/' + ct_str + '_vehicle'
        fn_v2i_prefix = folder_vehicle + '/' + ct_str + '_v2i'

        folder_tracking = folder_prefix + 'tracking_' + fmt + '_' + ct_str
        fn_tracking_prefix = folder_tracking + '/' + ct_str + '_tracking'

        vehicle_dict = {}
        instance_dict = {}

        
        frame_id = 0

        while True:

            # NOTE: The actual "frame_id" with vehicles starts from 1.
            frame_id += 1

            #if frame_id > 4000:
            #    break

            frame_id_str = '_%d' % frame_id
            fn_vehicle = fn_vehicle_prefix + frame_id_str
            fn_v2i = fn_v2i_prefix + frame_id_str
            fn_tracking = fn_tracking_prefix + frame_id_str

            if not os.path.isfile(fn_vehicle + '.' + fmt):
                #print('No such file: ', fn_vehicle)
                break

            vehicle_array = load_data_file(fn_vehicle, fmt, is_integer=False)
            v2i_array = load_data_file(fn_v2i, fmt, is_integer=True)
            tracking_array = load_data_file(fn_tracking, fmt, is_integer=False)

            if vehicle_array is None:
                continue

            n = vehicle_array.shape[0]

            for i in range(n):

                vehicle_vector = vehicle_array[i]

                vid = v2i_array[i, 0]
                instance_id = v2i_array[i, 1]

                # NOTE: Link instance ID here. Originally the first element
                # is just a line number.
                vehicle_vector[0] = instance_id

                if instance_id >= 0:

                    instance_vector = tracking_array[instance_id]

                else:

                    instance_vector = np.zeros(tracking_m)
                    instance_vector[0] = -1
                
                # NOTE: Rewrite "frame_id" to avoid issues with early
                # versions of tracker output format.
                vehicle_vector[2] = frame_id
                instance_vector[2] = frame_id

                
                if vid in vehicle_dict:
                    vehicle_vector_list = vehicle_dict[vid]
                    instance_vector_list = instance_dict[vid]
                else:
                    vehicle_vector_list = []
                    instance_vector_list = []
                
                vehicle_vector_list.append(vehicle_vector)
                instance_vector_list.append(instance_vector)

                vehicle_dict[vid] = vehicle_vector_list
                instance_dict[vid] = instance_vector_list


        print(vehicle_dict.keys())
        print(instance_dict.keys())

        # Save per-track vehicle data.

        folder_prefix = folder + '/reconstruction/'
        ct_str = camera_name + '_%d' % track_id

        folder_track = folder_prefix + 'track_' + fmt + '_' + ct_str
        if not os.path.isdir(folder_track):
            os.mkdir(folder_track)


        fn_vtrack_prefix = folder_track + '/' + ct_str + '_vtrack'
        fn_itrack_prefix = folder_track + '/' + ct_str + '_itrack'

        for vid in vehicle_dict:

            vehicle_vector_list = vehicle_dict[vid]
            instance_vector_list = instance_dict[vid]

            vehicle_array = np.asarray(vehicle_vector_list)
            instance_array = np.asarray(instance_vector_list)

            #print(vehicle_array.shape)
            #print(instance_array.shape)

            vid_str = '_%d' % vid

            fn_vtrack = fn_vtrack_prefix + vid_str
            fn_itrack = fn_itrack_prefix + vid_str

            save_data_file(fn_vtrack, vehicle_array, fmt, is_integer=False)
            save_data_file(fn_itrack, instance_array, fmt, is_integer=False)





    def process_multiple_view_tracks(self, camera_names, track_id):

        # Load per-frame multiple view vehicle association files.




        # Save per-track vehicle association data.


        pass




class Reconstructor(object):

    M = 50

    def __init__(self, folder, camera_name, track_id, frame_size, camera_model):

        self.folder = folder
        self.camera_name = camera_name
        self.track_id = track_id

        self.frame_size = frame_size

        self.camera_model = camera_model

        self.voxel_size = (self.M, self.M, self.M)
        self.dscaler = np.asarray((1.5, 1.5, 1.5))


        self.folder_reconstruction = folder + '/reconstruction/'
        self.folder_tracking = folder + '/tracking/'



    def load_vehicle_track_data(self, fmt, vehicle_id):

        mask_size_th = 6000

        camera_name = self.camera_name
        track_id = self.track_id
        frame_size = self.frame_size

        folder_reconstruction = self.folder_reconstruction
        folder_tracking = self.folder_tracking

        ct_str = camera_name + '_%d' % track_id

        vid = vehicle_id
        vid_str = '_%d' % vid

        folder_track = folder_reconstruction + 'track_' + fmt + '_' + ct_str
        fn_prefix = folder_track + '/' + ct_str
        fn_vtrack = fn_prefix + '_vtrack' + vid_str
        fn_itrack = fn_prefix + '_itrack' + vid_str

        folder_instance = folder_tracking + 'instance_' + fmt + '_' + ct_str
        fn_mask_prefix = folder_instance + '/' + ct_str + '_mask'

        if not os.path.isfile(fn_vtrack + '.' + fmt) \
            or not os.path.isfile(fn_itrack + '.' + fmt):

            print('No such vehicle: ', fn_vtrack + '.' + fmt)
            return None
        
        vehicle_array = load_data_file(fn_vtrack, fmt, is_integer=False)
        instance_array = load_data_file(fn_itrack, fmt, is_integer=False)

        nr_frames = vehicle_array.shape[0]

        vstate_list = []

        for i in range(nr_frames):

            frame_id = vehicle_array[i, 2]
            pos = vehicle_array[i, 15:18]
            pos_flag = vehicle_array[i, 4]
            heading = vehicle_array[i, 6:9]
            dimension = vehicle_array[i, 9:12]

            instance_id = instance_array[i, 0]
            detection_id = instance_array[i, 1]
            occlusion_flag = int(round(instance_array[i, 12]))

            box = instance_array[i, 13:17]
            bb3d_flag = instance_array[i, 21]
            bb3d_situation = instance_array[i, 22]
            bb3d_dim = instance_array[i, 78:81]
            mask = None

            fid_str = '_%d' % frame_id
            iid_str = '_%d' % instance_id
            fn_mask = fn_mask_prefix + fid_str + iid_str

            if instance_id >= 0 and detection_id >= 0 and occlusion_flag == 0:
                mask_local = load_data_file(fn_mask, fmt, is_integer=True)
                mask_size = np.sum(mask_local)
                if mask_size > mask_size_th:
                    mask = mask_local_to_global(box, mask_local, frame_size)
            
            vstate = VehicleStatesPerFrame(vid, frame_id, instance_id, 
                box, mask, pos, pos_flag, heading, dimension)
            vstate.bb3d_flag = bb3d_flag
            vstate.bb3d_dim = bb3d_dim
            vstate.bb3d_situation = bb3d_situation

            vstate_list.append(vstate)

        return vstate_list




    def shape_from_masks(self, vstate_list, dimension):

        voxel_size = self.voxel_size

        

        voxels = np.zeros(voxel_size, dtype=np.int)

        nr_reconstruction = 0
        for vstate in vstate_list:

            if vstate.mask is not None and vstate.bb3d_flag > 0:
                self.carve_voxels_one_mask(vstate, dimension, voxels)
                nr_reconstruction += 1

        #print(nr_reconstruction)

        return voxels, nr_reconstruction

    def estimate_dimension(self, vstate_list):

        dimensions = []
        for vstate in vstate_list:
            if vstate.bb3d_flag > 0:
                dimensions.append(vstate.dimension)
                #print(vstate.dimension)

        dimensions_array = np.asarray(dimensions)
        dimension_mean = np.mean(dimensions_array, axis=0)

        return dimension_mean

    def carve_voxels_one_mask(self, vstate, dimension, voxel):

        center, nvtu = self.calculate_center(vstate, dimension)

        points_xyz = self.calculate_voxel_points(center, nvtu, dimension)

        points_uv = self.camera_model.project_points(points_xyz)

        mask = vstate.mask
        self.check_voxel_points_on_mask(voxel, points_uv, mask)

    def calculate_center(self, vstate, dimension_mean):

        pos = vstate.pos
        pos_flag = vstate.pos_flag
        heading = vstate.heading

        bb3d_situation = int(round(vstate.bb3d_situation))
        dimension = vstate.bb3d_dim

        if bb3d_situation == 1 or bb3d_situation == 2:
            dimension = dimension_mean

        nv = heading / np.linalg.norm(heading)
        nt = np.asarray((-nv[1], nv[0], 0))
        nu = np.cross(nv, nt)
        
        if pos_flag == 0:
            
            # pos is (qa + qb) / 2
            center = pos - nv * dimension[0] / 2
            
        elif pos_flag == 1:
            
            # pos is (qa + qc) / 2
            center = pos - nt * dimension[1] / 2
        
        elif pos_flag == 2:
            
            # pos is (qb + qd) / 2
            center = pos + nt * dimension[1] / 2

        elif pos_flag == 3:
            
            # pos is (qc + qd) / 2
            center = pos + nv * dimension[0] / 2

        else:
            
            # BUG
            print('BUG in get_center(), pos_flag = %d' % pos_flag)
            raise ValueError('Wrong value of pos_flag (%d)!' % pos_flag)

        return center, (nv, nt, nu)

    def calculate_voxel_points(self, center, nvtu, dimension):

        nv, nt, nu = nvtu

        R = np.zeros((3, 3))
        R[:, 0] = nv
        R[:, 1] = nt
        R[:, 2] = nu

        t = center.reshape((3, 1))

        ln, wn, hn = self.voxel_size

        lnm = ln - 1
        wnm = wn - 1
        hnm = hn - 1

        l = dimension[0] * self.dscaler[0]
        w = dimension[1] * self.dscaler[1]
        h = dimension[2] * self.dscaler[2]

        qx = l / 2
        qy = w / 2
        qz = h

        qmx = -qx
        qmy = -qy
        qmz = 0

        points = np.zeros((ln, wn, hn, 3))

        for i in range(ln):
            x = qx * (i / lnm) + qmx * (1 - i / lnm)
            for j in range(wn):
                y = qy * (j / wnm) + qmy * (1 - j / wnm)
                for k in range(hn):
                    z = qz * (k / hnm) + qmz * (1 - k / hnm)

                    points[i, j, k, 0] = x
                    points[i, j, k, 1] = y
                    points[i, j, k, 2] = z

        #print(points)

        points = points.reshape((ln * wn * hn, 3))
        points = points.T

        points = np.matmul(R, points) + t

        #print('calculate_voxel_points()')
        #print(R)
        #print(t)

        #print(points.T)

        return points


    def check_voxel_points_on_mask(self, voxel, points_uv, mask):

        frame_width = mask.shape[1]
        frame_height = mask.shape[0]

        ln, wn, hn = self.voxel_size

        points_uv = points_uv.T

        #print('check_voxel_points_on_mask')
        #print(points_uv.shape)

        points_uv = points_uv.reshape((ln, wn, hn, 2))

        

        image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        image[mask] = 255

        for i in range(ln):
            for j in range(wn):
                for k in range(hn):

                    u = int(round(points_uv[i, j, k, 0]))
                    v = int(round(points_uv[i, j, k, 1]))

                    #print(u, v)

                    if u >= 0 and u < frame_width \
                        and v >= 0 and v < frame_height \
                        and mask[v, u]:
                        #voxel[i, j, k] = 1
                        pass
                    else:
                        voxel[i, j, k] -= 1
                    
                    #cv2.circle(image, (u, v), 1, (0, 0, 255), 1)

        #cv2.imshow('mask', image)

        #c = cv2.waitKey(-1)

    def vis_voxels_one_mask(self, vstate, dimension, voxel):

        center, nvtu = self.calculate_center(vstate, dimension)

        points_xyz = self.calculate_voxel_points(center, nvtu, dimension)

        points_uv = self.camera_model.project_points(points_xyz)

        mask = vstate.mask
        self.vis_voxel_points_on_mask(voxel, points_uv, mask)

    def vis_voxel_points_on_mask(self, voxel, points_uv, mask):

        frame_width = mask.shape[1]
        frame_height = mask.shape[0]

        ln, wn, hn = self.voxel_size

        points_uv = points_uv.T

        #print('check_voxel_points_on_mask')
        #print(points_uv.shape)

        points_uv = points_uv.reshape((ln, wn, hn, 2))

        image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        image[mask] = 255

        for i in range(ln):
            for j in range(wn):
                for k in range(hn):

                    u = int(round(points_uv[i, j, k, 0]))
                    v = int(round(points_uv[i, j, k, 1]))

                    #print(u, v)

                    if voxel[i, j, k] > 0:
                        cv2.circle(image, (u, v), 1, (255, 0, 0), 1)
                    else:
                        cv2.circle(image, (u, v), 1, (0, 0, 255), 1)
                    
        cv2.imshow('mask', image)

        c = cv2.waitKey(-1)

    def voxel_to_histogram(self, voxel, cut):
        
        ln, wn, hn = self.voxel_size
        
        hist = np.zeros((ln, wn), dtype=np.int)

        for i in range(ln):
            for j in range(wn):
                for k in range(hn - 1, -1, -1):

                    if voxel[i, j, k] >= cut:
                        hist[i, j] = k
                        break
        
        


        return hist

    def histogram_postprocess(self, hist):

        edge_ratio = 0.5
        r = 0.8

        ln, wn, _hn = self.voxel_size

        for i in range(ln):
            for j in range(wn):
                jj = wn - j - 1
                if hist[i, jj] < hist[i, j]:
                    hist[i, j] = hist[i, jj]

        ln_cut = ln / self.dscaler[0]
        wn_cut = wn / self.dscaler[1]

        ln_max = int(round(ln / 2 + ln_cut / 2))
        ln_min = int(round(ln / 2 - ln_cut / 2))
        wn_max = int(round(wn / 2 + wn_cut / 2))
        wn_min = int(round(wn / 2 - wn_cut / 2))
        wn_max = self.M - (wn_min + 1)

        #print(wn_min, wn_max)

        wn_edge = wn_cut * edge_ratio
        wn_edge_max = int(round(wn / 2 + wn_edge / 2))
        wn_edge_min = int(round(wn / 2 - wn_edge / 2))

        for i in range(ln):

            h_edge = hist[i, wn_edge_max]

            for j in range(wn):

                if j >= wn_edge_min and j <= wn_edge_max:
                    hist[i, j] = h_edge * r + hist[i, j] * (1 - r)

                if i <= ln_min or i >= ln_max or j < wn_min or j > wn_max:

                    hist[i, j] = 0


        return hist
    
    def hist_to_voxel(self, hist):

        ln, wn, hn = self.voxel_size
        
        voxels = np.zeros((ln, wn, hn), dtype=np.int)

        for i in range(ln):
            for j in range(wn):

                h = hist[i, j]
                if h == 0:
                    continue

                for k in range(0, h + 1):
                    if k >= hn:
                        k = hn - 1
                    voxels[i, j, k] = 1

        return voxels

    def hist_to_metric_voxel(self, hist, dimension, step=0.08):

        l = dimension[0] * self.dscaler[0]
        w = dimension[1] * self.dscaler[1]
        h = dimension[2] * self.dscaler[2]

        ln = int(l / step)
        wn = int(w / step)
        hn = int(h / step)

        hist = hist.astype(np.float)
        hist = cv2.resize(hist, (wn, ln))
        scale = hn / self.M
        hist = hist * scale

        voxels = np.zeros((ln, wn, hn), dtype=np.int)

        for i in range(ln):
            for j in range(wn):

                h = int(hist[i, j])
                if h == 0:
                    continue

                for k in range(0, h + 1):
                    if k >= hn:
                        k = hn - 1
                    voxels[i, j, k] = 1

        print(dimension)

        return voxels


    def create_folders(self, fmt):

        folder = self.folder

        camera_name = self.camera_name
        track_id = self.track_id

        ct_str = camera_name + '_%d' % track_id

        folder_prefix = folder + '/reconstruction/'
        folder_voxels = folder_prefix + 'voxels_' + fmt + '_' + ct_str

        if not os.path.isdir(folder_voxels):
            os.mkdir(folder_voxels)


    def get_fn(self, module, subfolder, subtype, fmt, vehicle_id):

        folder = self.folder

        camera_name = self.camera_name
        track_id = self.track_id

        ct_str = camera_name + '_%d' % track_id

        vid = vehicle_id
        vid_str = '_%d' % vid

        fp = folder + '/' + module + '/' + subfolder + '_' + fmt + '_' + ct_str
        fn = fp + '/' + ct_str + '_' + subtype + vid_str

        return fn




    def save_voxels(self, voxels, hist, dimension, vehicle_id, fmt):

        fn_voxels = self.get_fn('reconstruction', 'voxels', 'voxels', fmt, vehicle_id)
        voxels_array = voxels.reshape((self.M, self.M * self.M))
        save_data_file(fn_voxels, voxels_array, fmt, is_integer=True)

        fn_hist = self.get_fn('reconstruction', 'voxels', 'hist', fmt, vehicle_id)
        save_data_file(fn_hist, hist, fmt, is_integer=True)

        fn_dimension = self.get_fn('reconstruction', 'voxels', 'dimension', fmt, vehicle_id)
        dimension_array = np.asarray(dimension).reshape((3, 1))
        save_data_file(fn_dimension, dimension_array, fmt, is_integer=False)


    def load_voxels(self, vehicle_id, fmt):

        fn_voxels = self.get_fn('reconstruction', 'voxels', 'voxels', fmt, vehicle_id)
        voxels_array = load_data_file(fn_voxels, fmt, is_integer=True)
        voxels = voxels_array.reshape((self.M, self.M, self.M))

        fn_hist = self.get_fn('reconstruction', 'voxels', 'hist', fmt, vehicle_id)
        hist = load_data_file(fn_hist, fmt, is_integer=True)

        fn_dimension = self.get_fn('reconstruction', 'voxels', 'dimension', fmt, vehicle_id)
        dimension_array = load_data_file(fn_dimension, fmt, is_integer=False)
        dimension = dimension_array.flatten()

        return voxels, hist, dimension





def test_track_helper():

    fmt = 'csv'

    camera_id = 3
    track_id = 0

    folder = '../avacar_data'

    camera_names = ['westbound', 'eastbound', 'northbound', 'southbound', 'osburn0723']
    camera_name = camera_names[camera_id]


    track_helper = TrackerRecordHelper()

    track_helper.process_one_track(folder, camera_name, track_id, fmt)


    pass


def test_reconstructor():

    replay = True
    #replay = False

    show = True
    #show = False

    camera_id = 0
    track_id = 0

    fmt = 'csv'

    folder = '../avacar_data'
    
    
    camera_names = ['westbound', 'eastbound', 'northbound', 'southbound', 'osburn0723']
    camera_name = camera_names[camera_id]

    camera_model = Camera2DGroundModel()
    calibration_folder = folder + '/calibration_2d/' + camera_name
    camera_model.load_calib_para(calibration_folder, camera_name)

    frame_size = (1280, 720)

    rc = Reconstructor(folder, camera_name, track_id, frame_size, camera_model)

    rc.create_folders(fmt)

    for vid in range(1, 1000):

        print(vid)

        if not replay:

            vstate_list = rc.load_vehicle_track_data(fmt, vid)

            if vstate_list is None:
                break
            
            dimension = rc.estimate_dimension(vstate_list)

            voxels, n = rc.shape_from_masks(vstate_list, dimension)

            n_th = n // 5
            if n_th > 5:
                n_th = 5

            print(n, n_th)

            hist = rc.voxel_to_histogram(voxels, -n_th)
            hist = rc.histogram_postprocess(hist)

            rc.save_voxels(voxels, hist, dimension, vid, fmt)

        if show:

            voxels_load, hist_load, dimension_load = rc.load_voxels(vid, fmt)

            voxels_normalized = rc.hist_to_voxel(hist_load)
            voxels_metric = rc.hist_to_metric_voxel(hist_load, dimension_load)
            #visualize_hist(hist_load, voxels_pro, rc.voxel_size, dimension_load, rc.dscaler)

            # for vstate in vstate_list:

            #     if vstate.mask is not None and vstate.bb3d_flag > 0:
            #         rc.vis_voxels_one_mask(vstate, dimension, voxels_normalized)


            visualize_mesh(voxels_metric)



    pass





def test_mesh():

    # Generate a level set about zero of two identical ellipsoids in 3D
    ellip_base = ellipsoid(6, 10, 16, levelset=True)
    ellip_double = np.concatenate((ellip_base[:-1, ...],
                                ellip_base[2:, ...]), axis=0)

    print(ellip_double.shape)

    # Use marching cubes to obtain the surface mesh of these ellipsoids
    verts, faces, normals, values = measure.marching_cubes(ellip_double, 0)

    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlabel("x-axis: a = 6 per ellipsoid")
    ax.set_ylabel("y-axis: b = 10")
    ax.set_zlabel("z-axis: c = 16")

    ax.set_xlim(0, 24)  # a = 6 (times two for 2nd ellipsoid)
    ax.set_ylim(0, 20)  # b = 10
    ax.set_zlim(0, 32)  # c = 16

    plt.tight_layout()
    plt.show()
    pass







def fill_one_mesh(hist, hist_box, voxels_size, mesh, vertices):

    x_min, x_max, y_min, y_max, z_min, z_max = hist_box
    ln, wn, hn = voxels_size

    scale_x = (x_max - x_min) / ln
    scale_y = (y_max - y_min) / wn

    for k, face in enumerate(mesh.faces):

        if k % 10000 == 0:
            print('processing face ', k)

        p0 = vertices[face[0]]
        p1 = vertices[face[1]]
        p2 = vertices[face[2]]

        xp_min = min((p0[0], p1[0], p2[0]))
        xp_max = max((p0[0], p1[0], p2[0]))
        yp_min = min((p0[1], p1[1], p2[1]))
        yp_max = max((p0[1], p1[1], p2[1]))

        nx = (xp_max - xp_min) / scale_x
        ny = (yp_max - yp_min) / scale_y

        n = int(max(nx, ny) * 4)
        if n == 1:
            n = 4

        for i in range(n):
            a = i / (n - 1)
            p3 = (1 - a) * p1 + a * p2
            for j in range(n):
                b = j / (n - 1)
                p = (1 - b) * p0 + b * p3

                # Fill one point
                x, y, z = p

                xb = (x - x_min) / (x_max - x_min) * (ln - 1)
                yb = (y - y_min) / (y_max - y_min) * (wn - 1)
                zb = (z - z_min) / (z_max - z_min) * (hn - 1)

                xb = int(xb)
                yb = int(yb)

                if zb > hist[xb, yb]:
                    hist[xb, yb] = zb

        pass

def calibrate_frame(pp_ref):

    pp_ref = pp_ref.T

    nx1 = pp_ref[:, 0] - pp_ref[:, 2]
    nx2 = pp_ref[:, 1] - pp_ref[:, 3]
    nx = (nx1 + nx2) / 2
    nx = nx / np.linalg.norm(nx)

    ny1 = pp_ref[:, 0] - pp_ref[:, 1]
    ny2 = pp_ref[:, 2] - pp_ref[:, 3]
    ny = (ny1 + ny2) / 2
    ny = ny / np.linalg.norm(ny)

    nz = np.cross(nx, ny)
    nz = nz / np.linalg.norm(nz)

    #print(nx)
    #print(ny)
    #print(nz)

    ny = np.cross(nz, nx)
    ny = ny / np.linalg.norm(ny)

    R = np.zeros((3, 3))
    R[:, 0] = nx
    R[:, 1] = ny
    R[:, 2] = nz

    R = R.T

    t = np.mean(pp_ref, axis=1)
    t = t.reshape((3, 1))

    t = -np.matmul(R, t)

    return R, t

def process_vertices(vertices, pp_kp, R, t, dscaler):

    v_array = np.asarray(vertices)
    n = v_array.shape[0]
    print(n)

    v_array = v_array.T
    v_array = np.matmul(R, v_array) + t
    v_array = v_array.T

    pp_kp = pp_kp.T
    pp_kp = np.matmul(R, pp_kp) + t
    pp_kp = pp_kp.T

    x_max = np.max(v_array[:, 0])
    x_min = np.min(v_array[:, 0])

    y_max = np.max(v_array[:, 1])
    y_min = np.min(v_array[:, 1])

    z_max = np.max(v_array[:, 2])
    z_min = np.min(v_array[:, 2])

    #print(x_min, x_max)
    #print(y_min, y_max)
    #print(z_min, z_max)

    l = x_max - x_min
    w = y_max - y_min
    h = z_max - z_min

    dimension = [l, w, h]
    

    xc = (x_min + x_max) / 2
    yc = (y_min + y_max) / 2
    zc = z_min

    pc = np.asarray((xc, yc, zc)).reshape((1, 3))

    v_array = v_array - pc
    pp_kp = pp_kp - pc


    m_kp = 10
    n_kp = 24

    pps_kp = np.zeros((n_kp, 3))
    for i in range(m_kp):
        
        i1 = 2 * i
        i2 = 2 * i + 1

        pps_kp[i1] = pp_kp[i]
        pps_kp[i2] = pp_kp[i]
        pps_kp[i2, 1] = - pps_kp[i2, 1]

    pps_kp[20:24] = pps_kp[16:20]
    pps_kp[20:24, 2] = 0

    x_min = -l / 2 * dscaler[0]
    x_max = l / 2 * dscaler[0]

    y_min = -w / 2 * dscaler[1]
    y_max = w / 2 * dscaler[1]

    z_min = 0
    z_max = h * dscaler[2]

    

    hist_box = (x_min, x_max, y_min, y_max, z_min, z_max)


    return v_array, pps_kp, dimension, hist_box


def normalize_key_points(pps_kp, hist_box):

    pps_kp[:, 0] /= hist_box[1]
    pps_kp[:, 1] /= hist_box[3]
    pps_kp[:, 2] /= hist_box[5]

    return pps_kp


def test_model(model_id):

    save = True
    #save = False

    #replay = True
    replay = False

    vis = True
    #vis = False

    model_id_str = '%d' % model_id

    print('processing model ' + model_id_str)

    folder = '../avacar_data'

    obj_folder = folder + '/models/obj'

    obj_fn = obj_folder + '/' + model_id_str + '.obj'
    ref_fn = obj_folder + '/' + model_id_str + '.txt'

    hist_data_fn = obj_folder + '/hist_data_' + model_id_str
    hist_meta_fn = obj_folder + '/hist_meta_' + model_id_str

    pp = np.loadtxt(ref_fn, delimiter=',')

    pp_ref = pp[0:4, :]

    pp_kp = pp[4:14, :]
    
    #print(pp_ref)


    # Construct histogram from mesh.
    
    

    if replay:

        hist = np.load(hist_data_fn + '.npy')
        meta = np.load(hist_meta_fn + '.npy')

        voxels_size = meta[0].astype(np.int)
        dimension = meta[1]
        dscaler = meta[2]
        pps_kp = meta[3:27]

    else:

        M = 100
        dscaler = np.asarray((1.2, 1.2, 1.2))
        voxels_size = (M, M, M)

        R, t = calibrate_frame(pp_ref)

        scene = pywavefront.Wavefront(obj_fn, collect_faces=True)

        mesh = scene.mesh_list[0]
        vertices = scene.vertices

        print(len(mesh.faces))

        #print(pp_kp)

        v_tup = process_vertices(vertices, pp_kp, R, t, dscaler)
        v_array, pps_kp, dimension, hist_box = v_tup

        print(dimension)

        hist = np.zeros((M, M))
        fill_one_mesh(hist, hist_box, voxels_size, mesh, v_array)
        pps_kp = normalize_key_points(pps_kp, hist_box)

        meta = np.zeros((3 + 24, 3))
        meta[0] = voxels_size
        meta[1] = dimension
        meta[2] = dscaler
        meta[3:27] = pps_kp

        if save:

            np.save(hist_data_fn, hist)
            np.save(hist_meta_fn, meta)


    if vis:

        # Normalize dimensions for visualization.
        dimension[0] /= dimension[1]
        dimension[2] /= dimension[1]
        dimension[1] = 1

        visualize_hist(None, hist, pps_kp, voxels_size, dimension, dscaler)

        #visualize_key_points(pps_kp, dimension, dscaler)



class JointPrior(object):


    def __init__(self):

        self.vm = None
        self.z = None
        self.wk = None

    def calculate_joint_prior(self, vectors, K):
        '''
        Run PCA on the vectors and calculate the joint prior model.

        "vectors" is a N-by-D matrix of N vectors of D dimension.
        '''

        N = vectors.shape[0]
        D = vectors.shape[1]

        vm = np.mean(vectors, axis=0).reshape((1, D))

        vc = vectors - vm

        vc_cov = np.cov(vc.T)

        ev, w = np.linalg.eigh(vc_cov)

        idx = ev.argsort()[::-1]
        ev = ev[idx]
        w = w[:, idx]

        #print(ev)

        evk = ev[:K]
        wk = w[:, :K]

        z = np.dot(vc, wk)

        vp = np.dot(z, wk.T)

        self.vm = vm
        self.z = z
        self.wk = wk


    def save_joint_prior(self, jp_folder):

        vm_fn = jp_folder + '/vm'
        z_fn = jp_folder + '/z'
        wk_fn = jp_folder + '/wk'

        np.save(vm_fn, self.vm)
        np.save(z_fn, self.z)
        np.save(wk_fn, self.wk)


    def load_joint_prior(self, jp_folder):

        vm_fn = jp_folder + '/vm' + '.npy'
        z_fn = jp_folder + '/z' + '.npy'
        wk_fn = jp_folder + '/wk' + '.npy'

        self.vm = np.load(vm_fn)
        self.z = np.load(z_fn)
        self.wk = np.load(wk_fn)

    def joint_prior_reconstruct(self, z):

        k = z.shape[0]
        z = z.reshape((1, k))
        k_max = self.wk.shape[1]

        if k < k_max:

            wk = self.wk[:, :k]
        
        else:

            wk = self.wk
            z = z[:, :k_max]


        vp = np.dot(z, wk.T) + self.vm

        return vp.flatten()

    def get_group_prior_coefficients(self, group):

        s_ids = [0, 20, 35, 45, 55, 60, 70]
        e_ids = [20, 35, 45, 55, 60, 70, 80]

        s_id = s_ids[group]
        e_id = e_ids[group]

        z_group = np.mean(self.z[s_id:e_id], axis=0)

        print(z_group)

        return z_group.flatten()

    def get_training_sample_coefficients(self, model_id):

        return self.z[model_id].flatten()

    def get_model_number(self):

        return self.z.shape[0]


class JointPriorVisualizer(object):


    def __init__(self, joint_prior, M, L):

        self.jp = joint_prior

        self.M = M
        self.L = L

        self.K = 5
        self.s = np.zeros(self.K)

        self.use_model = False
        self.model_id = 0
        self.model_number = joint_prior.get_model_number()

        self.setup_ui()

        self.update_plot()

    def setup_ui(self):

        fig = plt.figure(figsize=(16, 9), dpi=100, facecolor='w', edgecolor='k')

        ax1 = fig.add_axes([0.00, 0.3, 0.5, 0.6], projection='3d')
        ax2 = fig.add_axes([0.5, 0.3, 0.5, 0.6], projection='3d')

        self.ax1 = ax1
        self.ax2 = ax2

        self.fig = fig

        axcolor = 'lightgoldenrodyellow'

        ax_s0 = fig.add_axes([0.02, 0.2, 0.15, 0.05], facecolor=axcolor)
        ax_s1 = fig.add_axes([0.22, 0.2, 0.15, 0.05], facecolor=axcolor)
        ax_s2 = fig.add_axes([0.42, 0.2, 0.15, 0.05], facecolor=axcolor)
        ax_s3 = fig.add_axes([0.62, 0.2, 0.15, 0.05], facecolor=axcolor)
        ax_s4 = fig.add_axes([0.82, 0.2, 0.15, 0.05], facecolor=axcolor)

        ax_ss = [ax_s0, ax_s1, ax_s2, ax_s3, ax_s4]
        s_ss = []

        for i in range(self.K):

            s_s = Slider(ax_ss[i], 's%d' % i, -10, 10, valinit=0, valstep=0.1)
            s_s.on_changed(self.on_slider_update)
            s_ss.append(s_s)

        self.ax_ss = ax_ss
        self.s_ss = s_ss


        ax_b0 = fig.add_axes([0.01, 0.1, 0.09, 0.05])
        ax_b1 = fig.add_axes([0.11, 0.1, 0.09, 0.05])
        ax_b2 = fig.add_axes([0.21, 0.1, 0.09, 0.05])
        ax_b3 = fig.add_axes([0.31, 0.1, 0.09, 0.05])
        ax_b4 = fig.add_axes([0.41, 0.1, 0.09, 0.05])
        ax_b5 = fig.add_axes([0.51, 0.1, 0.09, 0.05])
        ax_b6 = fig.add_axes([0.61, 0.1, 0.09, 0.05])

        ax_b_prev = fig.add_axes([0.71, 0.1, 0.05, 0.05])
        ax_t_model_id = fig.add_axes([0.78, 0.1, 0.10, 0.05])
        ax_b_next = fig.add_axes([0.90, 0.1, 0.05, 0.05])

        b_0 = Button(ax_b0, 'SUV')
        b_1 = Button(ax_b1, 'hatchback')
        b_2 = Button(ax_b2, 'sedan')
        b_3 = Button(ax_b3, 'coupe')
        b_4 = Button(ax_b4, 'minivan')
        b_5 = Button(ax_b5, 'van')
        b_6 = Button(ax_b6, 'pickup')

        b_prev = Button(ax_b_prev, '<-')
        b_next = Button(ax_b_next, '->')

        b_0.on_clicked(self.on_button_0_clicked)
        b_1.on_clicked(self.on_button_1_clicked)
        b_2.on_clicked(self.on_button_2_clicked)
        b_3.on_clicked(self.on_button_3_clicked)
        b_4.on_clicked(self.on_button_4_clicked)
        b_5.on_clicked(self.on_button_5_clicked)
        b_6.on_clicked(self.on_button_6_clicked)

        b_prev.on_clicked(self.on_button_prev_clicked)
        b_next.on_clicked(self.on_button_next_clicked)

        ax_bs = [ax_b0, ax_b1, ax_b2, ax_b3, ax_b4, ax_b5, ax_b6]
        s_bs = [b_0, b_1, b_2, b_3, b_4, b_5, b_6]

        self.ax_bs = ax_bs
        self.s_bs = s_bs

        self.ax_b_prev = ax_b_prev
        self.ax_t_model_id = ax_t_model_id
        self.ax_b_next = ax_b_next

        self.b_prev = b_prev
        self.b_next = b_next



    def on_slider_update(self, val):

        self.use_model = False

        for i, s_s in enumerate(self.s_ss):

            self.s[i] = s_s.val
        
        self.update_plot()

    def on_button_0_clicked(self, event):

        self.process_model_button_click(0)

    def on_button_1_clicked(self, event):

        self.process_model_button_click(1)

    def on_button_2_clicked(self, event):

        self.process_model_button_click(2)

    def on_button_3_clicked(self, event):

        self.process_model_button_click(3)

    def on_button_4_clicked(self, event):

        self.process_model_button_click(4)

    def on_button_5_clicked(self, event):

        self.process_model_button_click(5)

    def on_button_6_clicked(self, event):

        self.process_model_button_click(6)

    def process_model_button_click(self, button_id):

        self.use_model = False
        self.s = self.jp.get_group_prior_coefficients(button_id)

        self.update_plot()


    def on_button_prev_clicked(self, event):

        self.use_model = True
        self.model_id = (self.model_id - 1) % self.model_number

        self.s = self.jp.get_training_sample_coefficients(self.model_id)

        self.update_plot()

    def on_button_next_clicked(self, event):

        self.use_model = True
        self.model_id = (self.model_id + 1) % self.model_number

        self.s = self.jp.get_training_sample_coefficients(self.model_id)

        self.update_plot()

    def update_plot(self):

        M = self.M
        L = self.L

        vector_p = self.jp.joint_prior_reconstruct(self.s)
        shape_v, wire_v, dimension_v = decouple_joint_vector(vector_p, M, L)

        voxels_size = (M, M, M)
        dscaler = np.asarray((1.2, 1.2, 1.2))

        self.ax1.clear()
        self.ax2.clear()
        visualize_hist(self.ax1, shape_v, wire_v, voxels_size, dimension_v, dscaler)
        visualize_hist(self.ax2, shape_v, wire_v, voxels_size, dimension_v, dscaler, 
            alpha=0.5, kp=True)

        self.ax_t_model_id.clear()
        self.ax_t_model_id.axis('off')
        if self.use_model:
            model_str = '%d' % self.model_id
            self.ax_t_model_id.text(0.5, 0.5, model_str,
                      horizontalalignment='center',
                      verticalalignment='center')

        for i in range(self.K):

            self.s_ss[i].val = self.s[i]

        self.fig.canvas.draw()


    def run(self):
        
        print(matplotlib.__version__)

        plt.show()


def smooth_shape(hist):

    M0 = hist.shape[0]
    M1 = hist.shape[1]

    for i in range(M0):

        h = 0
        for j in range(M1 // 2):

            if hist[i, j] > h:
                h = hist[i, j]
            else:
                hist[i, j] = h

        for j in range(M1 // 2):

            hist[i, M1 - j - 1] = hist[i, j]

    hist = hist / M0

    hist = cv2.GaussianBlur(hist, (3, 3), 0)
    hist = cv2.GaussianBlur(hist, (3, 3), 0)

    return hist


def load_joint_shape_wireframe_vectors(obj_folder, N, M, L):

    dscaler = np.asarray((1.2, 1.2, 1.2))
    voxels_size = (M, M, M)

    shape_v = np.zeros((N, M, M))
    wire_v = np.zeros((N, L, 3))
    dimension_v = np.zeros((N, 3))
    dscaler_v = np.zeros((N, 3))

    # Load histogram and key point data.

    for i in range(N):

        model_id_str = '%d' % i

        hist_data_fn = obj_folder + '/hist_data_' + model_id_str
        hist_meta_fn = obj_folder + '/hist_meta_' + model_id_str

        hist = np.load(hist_data_fn + '.npy')
        meta = np.load(hist_meta_fn + '.npy')

        dimension_v[i] = meta[1]
        dscaler_v[i] = meta[2]

        hist = smooth_shape(hist)
        hist = cv2.resize(hist, (M, M))

        shape_v[i] = hist
        wire_v[i] = meta[3:27]

        dimension_v[i, 0] /= dimension_v[i, 1]
        dimension_v[i, 2] /= dimension_v[i, 1]
        dimension_v[i, 1] = 1

        dimension = dimension_v[i]

        pps_kp = wire_v[i]

        #print(i, dimension)
        #print(pps_kp)

        #visualize_hist(hist, pps_kp, voxels_size, dimension, dscaler)


    # Prepare vectors.

    shape_vf = shape_v.reshape((N, M * M))
    wire_vf = wire_v.reshape((N, L * 3))
    dimension_vf = dimension_v

    vectors = np.concatenate((shape_vf, wire_vf, dimension_vf), axis=1)

    #vectors = wire_vf


    return vectors

def decouple_joint_vector(vector, M, L):

    c0 = 0
    c1 = M * M
    c2 = M * M + L * 3
    c3 = M * M + L * 3 + 3

    shape_vf = vector[c0:c1]
    wire_vf = vector[c1:c2]
    dimension_vf = vector[c2:c3]

    shape_v = shape_vf.reshape((M, M))
    wire_v = wire_vf.reshape((L, 3))
    dimension_v = dimension_vf

    return shape_v, wire_v, dimension_v

def check_reconstruction_error(self, v, vm, z, wk):



    pass



def test_joint_prior():

    save = True
    #save = False

    #replay = True
    replay = False


    # "L" is the number of wireframe points.
    L = 24
    # "M" is the number of grid ticks.
    M = 50
    # "N" is the number of models.
    N = 80

    K = 20


    folder = '../avacar_data'

    obj_folder = folder + '/models/obj'

    jp_folder = folder + '/models/prior'

    vectors = load_joint_shape_wireframe_vectors(obj_folder, N, M, L)

    print(vectors.shape)

    # Calculate the joint prior model.

    jp = JointPrior()

    if replay:

        jp.load_joint_prior(jp_folder)

    else:

        jp.calculate_joint_prior(vectors, K)

        if save:

            jp.save_joint_prior(jp_folder)

    print(jp.z.shape, jp.z[0])
    
    # Reconstruct surface.

    #vectors_p = jp.joint_prior_reconstruct(jp.z)

    # Check surface error.

    #shape_v, wire_v, dimension_v = decouple_joint_vectors(vectors_p, N, M, L)


    # Visualization.


    # dscaler = np.asarray((1.2, 1.2, 1.2))
    # voxels_size = (M, M, M)

    # for i in range(0, N):

    #     hist = shape_v[i]
    #     pps_kp = wire_v[i]
    #     dimension = dimension_v[i]

        #print(i, dimension)
        #print(jp.z[i])


        #visualize_hist(hist, pps_kp, voxels_size, dimension, dscaler)

    
    vis = JointPriorVisualizer(jp, M, L)

    vis.run()








if __name__ == '__main__':


    #test_reconstruction()

    #test_track_helper()

    test_reconstructor()

    #test_mesh()

    #test_model(76)

    #for i in range(80):

    #    test_model(i)


    #test_joint_prior()


    pass