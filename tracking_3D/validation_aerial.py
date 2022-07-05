'''
Created on Aug 10, 2020

@author: duolu
'''

import warnings

import numpy as np

import matplotlib.pyplot as plt
import cv2

from calibration import MapModel
from calibration import Camera2DGroundModel

from validation_gps import load_gps_rt_results

from visualization import MapVis


class VStateSeq(object):

    def __init__(self, vid):

        self.vid = vid
        self.pos_list = []
        self.vel_list = []
        self.itr_list = []
        self.fid_list = []

    def append(self, pos, vel, itr, fid):

        self.pos_list.append(pos)
        self.vel_list.append(vel)
        self.itr_list.append(itr)
        self.fid_list.append(fid)

    def __getitem__(self, key):

        return self.pos_list[key], self.vel_list[key], self.itr_list[key], self.fid_list[key]

    def resize(self, k):

        self.pos_list = self.pos_list[:k]
        self.vel_list = self.vel_list[:k]
        self.itr_list = self.itr_list[:k]
        self.fid_list = self.fid_list[:k]


def load_ground_data(folder, prefix, postfix, fmt, fid_start, fid_end):

    subfolder = folder + '/tracking/vehicle_' + fmt + '_' + prefix + postfix

    vstate_dict = {}

    for fid in range(fid_start, fid_end):

        fid_str = '_%d' % fid
        fn = subfolder + '/' + prefix + postfix + '_vehicle' + fid_str

        array = load_data_file(fn, fmt, is_integer=False)

        if array is not None:

            n = array.shape[0]
            for i in range(n):

                vid = int(round(array[i, 1]))
                itr = int(round(array[i, 3]))
                pos_flag = int(round(array[i, 4]))

                heading = array[i, 6:9]
                dim = array[i, 9:12]
                pos = array[i, 15:18]
                vel = array[i, 27:30]

                #if dim[0] < dim[1] * 2.2:
                #    dim[0] = dim[1] * 2.2

                #print(dim)

                pos = get_center_from_pos(pos_flag, pos, dim, heading)

                if vid in vstate_dict:
                    seq = vstate_dict[vid]
                else:
                    seq = VStateSeq(vid)
                    vstate_dict[vid] = seq

                seq.append(pos, vel, itr, fid)

        #print(fid)

    # Trim the sequence

    for vid, seq in vstate_dict.items():

        n = len(seq.fid_list)
        for i in range(1, n):
            assert seq.fid_list[i] - seq.fid_list[i - 1] == 1
        
        for k in range(n - 1, -1, -1):

            if seq.itr_list[k] < 1:
                break
        
        seq.resize(k)



    return vstate_dict


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



def get_center_from_pos(pos_flag, pos, dim, heading):

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

    elif pos_flag == 8:
        
        # pos is (qa + qb + qc + qd) / 4
        center = pos

    else:
        
        # BUG
        print('BUG in get_center(), pos_flag = %d' % pos_flag)

    return center


def load_aerial_data(folder, prefix, postfix, fmt, fid_start, fid_end):

    subfolder = folder + '/aerial_tracking/vehicle_' + fmt + '_' + prefix + postfix

    vstate_dict = {}

    for fid in range(fid_start, fid_end):

        fid_str = '_%d' % fid
        fn = subfolder + '/' + prefix + postfix + '_vehicle' + fid_str

        array = load_data_file(fn, fmt, is_integer=False)

        if array is not None:

            n = array.shape[0]
            for i in range(n):

                vid = int(round(array[i, 1]))
                itr = int(round(array[i, 3]))

                dim = array[i, 12:15]
                pos = array[i, 18:21]
                vel = array[i, 30:33]

                if vid in vstate_dict:
                    seq = vstate_dict[vid]
                else:
                    seq = VStateSeq(vid)
                    vstate_dict[vid] = seq
                
                seq.append(pos, vel, itr, fid)

    # Trim the sequence


    for vid, seq in vstate_dict.items():

        n = len(seq.fid_list)
        for i in range(1, n):
            assert seq.fid_list[i] - seq.fid_list[i - 1] == 1


    return vstate_dict






def align_data(vstate_g, vstate_a, vid_ass, offset_g2a):

    vstate_g_new = {}
    vstate_a_new = {}

    # m is the number of vehicles.
    m = vid_ass.shape[0]

    for j in range(m):

        vid_g = vid_ass[j, 0]
        vid_a = vid_ass[j, 1]

        seq_g = vstate_g[vid_g]
        seq_a = vstate_a[vid_a]

        seq_g_new, seq_a_new = align_one_seq_pair(seq_g, seq_a, offset_g2a)

        vstate_g_new[vid_g] = seq_g_new
        vstate_a_new[vid_a] = seq_a_new

    return vstate_g_new, vstate_a_new



def align_one_seq_pair(seq_g, seq_a, offset_g2a):

    ng = len(seq_g.fid_list)
    na = len(seq_a.fid_list)

    if ng == 0 or na == 0:
        return None

    kg = 0
    ka = 0

    fid_g = seq_g.fid_list[0]
    fid_a = seq_a.fid_list[0]

    while fid_g + offset_g2a < fid_a:
        kg += 1
        fid_g += 1
    
    while fid_a < fid_g + offset_g2a:
        ka += 1
        fid_a += 1

    seq_g_new = VStateSeq(seq_g.vid)
    seq_a_new = VStateSeq(seq_a.vid)

    while kg < ng and ka < na:

        seq_g_new.append(*seq_g[kg])
        seq_a_new.append(*seq_a[ka])

        kg += 1
        ka += 1

    return seq_g_new, seq_a_new


def validate_data(vstate_g, vstate_a, vid_ass):


    m = vid_ass.shape[0]

    dist_avg_all = np.zeros(m)
    dists_all = []

    for j in range(m):

        vid_g = vid_ass[j, 0]
        vid_a = vid_ass[j, 1]

        seq_g = vstate_g[vid_g]
        seq_a = vstate_a[vid_a]

        ng = len(seq_g.fid_list)
        na = len(seq_a.fid_list)

        assert ng == na

        dists = np.zeros(ng)

        for i in range(ng):

            #pos_g = seq_g.vel_list[i]
            #pos_a = seq_a.vel_list[i]

            pos_g = seq_g.pos_list[i]
            pos_a = seq_a.pos_list[i]

            dists[i] = np.linalg.norm(pos_g - pos_a)
            
        dist_avg_all[j] = np.mean(dists)
        dists_all.append(dists)

    dist_avg_avg = np.mean(dist_avg_all)
    print(dist_avg_avg)

    for j in range(m):

        dists = dists_all[j]
        n = dists.shape[0]
        for i in range(n):
            print('%.2f\t' % dists[i], end='')
        print()


def analyze_error(vstate_g, vstate_a, vid_ass, camera_model):

    _R, t = camera_model.get_camera_pose()

    dist_pairs_list = []

    m = vid_ass.shape[0]

    ss = 0
    nn = 0

    for j in range(m):

        vid_g = vid_ass[j, 0]
        vid_a = vid_ass[j, 1]

        seq_g = vstate_g[vid_g]
        seq_a = vstate_a[vid_a]

        ng = len(seq_g.fid_list)
        na = len(seq_a.fid_list)

        assert ng == na

        dist_pairs = np.zeros((ng, 2))

        #print(vid_g)

        for i in range(ng):

            #pos_g = seq_g.vel_list[i]
            #pos_a = seq_a.vel_list[i]

            pos_g = seq_g.pos_list[i]
            pos_a = seq_a.pos_list[i]

            lon_vec = pos_g - t
            lon_vec[2] = 0
            lat_vec = np.array((-lon_vec[1], lon_vec[0], 0))

            lon_vec /= np.linalg.norm(lon_vec)
            lat_vec /= np.linalg.norm(lat_vec)

            dist_vec = pos_g - pos_a

            lon_dist = np.dot(dist_vec, lon_vec)
            lat_dist = np.dot(dist_vec, lat_vec)

            depth = np.linalg.norm(pos_g - t)
            dist = np.linalg.norm(pos_g - pos_a)

            dist_pairs[i, 0] = depth
            dist_pairs[i, 1] = dist

            #print(lon_dist, lat_dist)

            if depth < 120:
                ss += dist
                nn += 1

        #print()

        dist_pairs_list.append(dist_pairs)

    print(ss / nn)


def plot_gps_trajectory(map_local, gps_lmap):

    m = gps_lmap.shape[0]

    for i in range(m):

        if i % 3 != 0:
            continue

        tx = int(gps_lmap[i, 0])
        ty = int(gps_lmap[i, 1])

        cv2.circle(map_local, (tx, ty), 1, (0, 0, 255), 3)

def plot_aerial_trajectory(map_local, aerial_lmap):

    m = aerial_lmap.shape[0]

    for i in range(m):

        if i % 3 != 0:
            continue

        tx = int(aerial_lmap[i, 0])
        ty = int(aerial_lmap[i, 1])

        cv2.circle(map_local, (tx, ty), 1, (0, 140, 255), 3)


def plot_tracking_trajectory(map_local, tracking_lmap):

    m = tracking_lmap.shape[0]

    for i in range(m):

        if i % 3 != 0:
            continue

        cx = int(tracking_lmap[i, 0])
        cy = int(tracking_lmap[i, 1])

        cv2.circle(map_local, (cx, cy), 1, (255, 0, 0), 3)


def test_validate_aerial():

    folder = '../avacar_data'

    

    g_prefix = 'southbound'
    g_postfix = '_103'

    a_prefix = '1000'
    a_postfix = '_2'

    g_fmt = 'npy'
    a_fmt = 'npy'

    g_fid_start = 7602
    g_fid_end = 7942

    a_fid_start = 9730
    a_fid_end = 10070


    offset_g2a = a_fid_start - g_fid_start

    # Load vid association

    #vid_ass_fn = folder + '/validation_aerial/vid_association'
    #vid_ass = np.loadtxt(vid_ass_fn, dtype=np.int, delimiter=',')

    #print(vid_ass)

    vid_ass = np.array((105, 147))
    vid_ass = vid_ass.reshape((1, 2))

    g_paras = (g_prefix, g_postfix, g_fmt, g_fid_start, g_fid_end)
    vstate_g = load_ground_data(folder, *g_paras)

    a_paras = (a_prefix, a_postfix, a_fmt, a_fid_start, a_fid_end)
    vstate_a = load_aerial_data(folder, *a_paras)

    vstate_g_new, vstate_a_new = align_data(vstate_g, vstate_a, vid_ass, offset_g2a)

    #validate_data(vstate_g_new, vstate_a_new, vid_ass)

    calibration_folder = folder + '/calibration_2d/' + g_prefix

    camera_model = Camera2DGroundModel()
    camera_model.load_calib_para(calibration_folder, g_prefix)

    analyze_error(vstate_g_new, vstate_a_new, vid_ass, camera_model)


    
    map_global_fn = calibration_folder + '/' + g_prefix + '_map_global.png'
    map_local_fn = calibration_folder + '/' + g_prefix + '_map_local.png'

    map_local = cv2.imread(map_local_fn)
    map_global = cv2.imread(map_global_fn)

    map_model = MapModel(map_local, map_global)
    map_model.load_map_para(calibration_folder, g_prefix)

    factor = 1.5
    map_local = map_local.astype(np.float)
    map_local /= factor
    map_local += (255 - 255 / factor)
    map_local = map_local.astype(np.uint8)


    vid_g = 105
    vid_a = 147

    vseq_g = vstate_g_new[vid_g]
    g_pos_xyz = np.asarray(vseq_g.pos_list)
    g_vel_xyz = np.asarray(vseq_g.vel_list)
    g_pos_lmap = map_model.transform_points_xyz_to_lmap(g_pos_xyz.T).T


    vseq_a = vstate_a_new[vid_a]
    a_pos_xyz = np.asarray(vseq_a.pos_list)
    a_vel_xyz = np.asarray(vseq_a.vel_list)
    a_pos_lmap = map_model.transform_points_xyz_to_lmap(a_pos_xyz.T).T


    n = g_vel_xyz.shape[0]
    m = a_vel_xyz.shape[0]

    print(n, m)

    dist_xyz = np.zeros((n, 3))
    dist = np.zeros(n)

    for i in range(n):

        print(g_pos_xyz[i, 0], g_pos_xyz[i, 1], a_pos_xyz[i, 0], a_pos_xyz[i, 1])

        dist_vec = g_pos_xyz[i] - a_pos_xyz[i]
        dist_xyz[i] = dist_vec
        dist[i] = np.linalg.norm(dist_vec)

        g_vel = np.linalg.norm(g_vel_xyz[i])
        a_vel = np.linalg.norm(a_vel_xyz[i])

        #print(g_vel, a_vel)

    print(np.mean(dist))


    # validate GPS data and drone data

    validation_folder = folder + '/gps_rt_validation'
    gps_fn = validation_folder + '/GPS_1006_Run_15.csv'

    gps_pos_xyz, gps_pos_lmap, gps_vel = load_gps_rt_results(gps_fn, map_model)

    gps_pos_x = gps_pos_xyz[:, 0]
    gps_pos_y = gps_pos_xyz[:, 1]

    g_pos_x = g_pos_xyz[:, 0]
    g_pos_y = g_pos_xyz[:, 1]
    g_pos_y_re = np.interp(gps_pos_x, g_pos_x, g_pos_y)

    a_pos_x = a_pos_xyz[:, 0]
    a_pos_y = a_pos_xyz[:, 1]
    a_pos_y_re = np.interp(gps_pos_x, a_pos_x, a_pos_y)


    m = gps_pos_xyz.shape[0]

    for i in range(m):

        #print(gps_pos_x[i], gps_pos_y[i], g_pos_x[i], a_pos_x[i])
        pass

    plot_aerial_trajectory(map_local, a_pos_lmap)
    plot_tracking_trajectory(map_local, g_pos_lmap)
    plot_gps_trajectory(map_local, gps_pos_lmap)

    mvis = MapVis()

    #FOV_ROI = (0, 93, 1280, 720)
    #mvis.draw_camera_FOV_on_map(map_local, camera_model, map_model, FOV_ROI)

    map_local = cv2.resize(map_local, (720, 1280))
    cv2.imshow('map_local', map_local)

    c = cv2.waitKey(-1)


    pass









if __name__ == '__main__':
    

    test_validate_aerial()















