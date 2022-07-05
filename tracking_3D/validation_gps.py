'''
Created on Aug 10, 2020

@author: duolu
'''

import warnings

import numpy as np
import cv2

from calibration import MapModel


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




def plot_gps_trajectory(map_local, gps_lmap):

    m = gps_lmap.shape[0]

    for i in range(m):

        if i % 3 != 0:
            continue

        tx = int(gps_lmap[i, 0])
        ty = int(gps_lmap[i, 1])

        cv2.circle(map_local, (tx, ty), 3, (0, 0, 255), 3)


def plot_tracking_trajectory(map_local, tracking_lmap):

    m = tracking_lmap.shape[0]

    for i in range(m):

        if i % 3 != 0:
            continue

        cx = int(tracking_lmap[i, 0])
        cy = int(tracking_lmap[i, 1])

        cv2.circle(map_local, (cx, cy), 3, (255, 0, 0), 3)



def plot_trajectory(map_local, telemetry_lmap, camera_lmap):

    m = telemetry_lmap.shape[0]

    for i in range(m):

        tx = int(telemetry_lmap[i, 0])
        ty = int(telemetry_lmap[i, 1])

        cx = int(camera_lmap[i, 0])
        cy = int(camera_lmap[i, 1])

        cv2.circle(map_local, (cx, cy), 3, (255, 0, 0), 3)
        cv2.circle(map_local, (tx, ty), 3, (0, 0, 255), 3)


def print_pos(pos):

    n = pos.shape[0]

    for i in range(n):

        print(pos[i, 0], pos[i, 2], pos[i, 3])



def test_valid_gps_cmt():


    camera_id = 3
    track_id = 103

    vid_list = [101]
    
    folder = '../avacar_data'
    
    
    prefixes = ['westbound', 'eastbound', 'northbound', 'southbound', 'osburn0723']
    prefix = prefixes[camera_id]
    postfix = ('_%d' % track_id)


    calibration_folder = folder + '/calibration_2d/' + prefix
    
    map_global_fn = calibration_folder + '/' + prefix + '_map_global.png'
    map_local_fn = calibration_folder + '/' + prefix + '_map_local.png'

    map_local = cv2.imread(map_local_fn)
    map_global = cv2.imread(map_global_fn)

    map_model = MapModel(map_local, map_global)
    map_model.load_map_para(calibration_folder, prefix)


    validation_folder = folder + '/validation'

    validation_fn = validation_folder + '/' + prefix + postfix + '.csv'
    telemetry_fn = validation_folder + '/' + prefix + postfix + '_telemetry.csv'

    array = np.loadtxt(validation_fn, delimiter=',')
    telemetry = np.loadtxt(telemetry_fn, delimiter=',')

    m = array.shape[0]

    # for i in range(m):

    #     if i % 2 != 0:
    #         continue

    #     print('%d\t%f\t%f' % (array[i, 0], array[i, 3], array[i, 4]))


    

    n = telemetry.shape[0]
    
    telemetry_esd = np.zeros((n, 3))
    telemetry_lmap = np.zeros((n, 3))

    for i in range(n):

        llh = np.asarray((telemetry[i, 0], telemetry[i, 1], 539))
        llh = llh.reshape((3, 1))
        esd = map_model.transform_points_llh_to_esd(llh)
        xyz = map_model.transform_points_esd_to_xyz(esd)
        lmap = map_model.transform_points_xyz_to_lmap(xyz)

        esd = esd.flatten()
        lmap = lmap.flatten()

        telemetry_esd[i] = esd
        telemetry_lmap[i] = lmap

    camera_esd = np.zeros((n, 3))
    camera_lmap = np.zeros((n, 3))

    for i in range(m):

        if i % 2 != 0:
            continue

        j = i // 2

        frame_id = array[i, 0]

        xyz = np.asarray((array[i, 1], array[i, 2], 0))
        xyz = xyz.reshape((3, 1))
        esd = map_model.transform_points_xyz_to_esd(xyz)
        lmap = map_model.transform_points_xyz_to_lmap(xyz)

        esd = esd.flatten()
        lmap = lmap.flatten()

        if j < m:
            camera_esd[j] = esd
            camera_lmap[j] = lmap
        
        e_cam = esd[0]
        e_tel = telemetry_esd[j, 0]
        e_diff = e_cam - e_tel

        s_cam = esd[1]
        s_tel = telemetry_esd[j, 1]
        s_diff = s_cam - s_tel

        print('%d\t%f\t%f\t%f\t%f\t%f\t%f\t' 
            % (frame_id, e_cam, e_tel, e_diff, 
                s_cam, s_tel, s_diff))



    plot_trajectory(map_local, telemetry_lmap, camera_lmap)

    #map_local = cv2.resize(map_local, (1280, 720))
    map_local = cv2.resize(map_local, (720, 1280))
    cv2.imshow('map_local', map_local)

    c = cv2.waitKey(-1)


def load_tracing_results(folder, prefix, postfix, fmt, fid_start, fid_end, vid_list):

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

                if not vid in vid_list:
                    continue

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


def load_gps_rt_results(gps_fn, map_model):

    start = 10
    n = 1127


    gps_array = np.loadtxt(gps_fn, delimiter=',', comments='#')

    gps_array = gps_array[start:start+n, :]

    lat = gps_array[:, 0]
    lon = gps_array[:, 1]
    # convert foot to meter
    height = gps_array[:, 2] * 0.3048

    print(height)

    # convert to m/s
    vn = gps_array[:, 3] * 1.60934 / 3.6
    ve = gps_array[:, 4] * 1.60934 / 3.6


    t = np.zeros(n)

    pos_xyz = np.zeros((n, 3))
    pos_lmap = np.zeros((n, 3))
    vel = np.zeros(n)

    for i in range(n):

        t[i] = i * 0.01

        llh = np.asarray((lat[i], lon[i], height[i]))
        llh = llh.reshape((3, 1))
        esd = map_model.transform_points_llh_to_esd(llh)
        xyz = map_model.transform_points_esd_to_xyz(esd)
        lmap = map_model.transform_points_xyz_to_lmap(xyz)

        xyz = xyz.flatten()
        lmap = lmap.flatten()

        pos_xyz[i] = xyz
        pos_lmap[i] = lmap
        
        v = np.asarray((vn[i], ve[i]))
        vel[i] = np.linalg.norm(v)

    m = int(n / (10 / 3))
    xt = np.zeros(m)
    for i in range(m):
        xt[i] = i * 1 / 30

    pos_xyz_re = np.zeros((m, 3))
    pos_lmap_re = np.zeros((m, 3))

    for j in range(3):

        pos_xyz_re[:, j] = np.interp(xt, t, pos_xyz[:, j])
        pos_lmap_re[:, j] = np.interp(xt, t, pos_lmap[:, j])

    vel_re = np.interp(xt, t, vel)


        



    return pos_xyz_re, pos_lmap_re, vel_re




def test_valid_gps_rt():


    camera_id = 3
    track_id = 103

    vid = 105
    vid_list = [vid]
    
    folder = '../avacar_data'
    
    
    prefixes = ['westbound', 'eastbound', 'northbound', 'southbound', 'osburn0723']
    prefix = prefixes[camera_id]
    postfix = ('_%d' % track_id)

    fmt = 'npy'

    calibration_folder = folder + '/calibration_2d/' + prefix
    
    map_global_fn = calibration_folder + '/' + prefix + '_map_global.png'
    map_local_fn = calibration_folder + '/' + prefix + '_map_local.png'

    map_local = cv2.imread(map_local_fn)
    map_global = cv2.imread(map_global_fn)

    map_model = MapModel(map_local, map_global)
    map_model.load_map_para(calibration_folder, prefix)


    validation_folder = folder + '/gps_rt_validation'

    gps_fn = validation_folder + '/GPS_1006_Run_15.csv'


    gps_pos_xyz, gps_pos_lmap, gps_vel = load_gps_rt_results(gps_fn, map_model)


    vstate_dict = load_tracing_results(folder, prefix, postfix, fmt, 7602, 7950, vid_list)

    vseq = vstate_dict[vid]
    tracking_pos_xyz = np.asarray(vseq.pos_list)
    tracking_vel_xyz = np.asarray(vseq.vel_list)
    
    tracking_pos_lmap = map_model.transform_points_xyz_to_lmap(tracking_pos_xyz.T).T

    n = tracking_pos_xyz.shape[0]




    print(n)
    print(gps_pos_xyz.shape[0])


    m = gps_vel.shape[0]

    for i in range(m - 4):

        delta1 = tracking_pos_xyz[i + 1] - tracking_pos_xyz[i]
        delta2 = tracking_pos_xyz[i + 2] - tracking_pos_xyz[i + 1]
        delta3 = tracking_pos_xyz[i + 3] - tracking_pos_xyz[i + 2]
        delta4 = tracking_pos_xyz[i + 4] - tracking_pos_xyz[i + 3]
        delta = (delta1 + delta2 + delta3 + delta4) / 4
        tracking_vel = np.linalg.norm(delta) / (1 / 30)
        #tracking_vel = np.linalg.norm(tracking_vel_xyz[i])
        #print(tracking_vel_xyz[i])
        #print(i * (1 / 30), tracking_vel, gps_vel[i])

    pos_xyz_delta = np.zeros((m, 3))
    pos_delta = np.zeros(m)
    for i in range(m):

        #print(tracking_pos_xyz[i, 0], tracking_pos_xyz[i, 1], gps_pos_xyz[i, 0], gps_pos_xyz[i, 1])

        xyz_delta = tracking_pos_xyz[i] - gps_pos_xyz[i]

        pos_xyz_delta[i] = xyz_delta
        pos_delta[i] = np.linalg.norm(xyz_delta)

    diff_x = np.mean(np.abs(pos_xyz_delta[:, 0]))
    diff_y = np.mean(np.abs(pos_xyz_delta[:, 1]))

    diff = np.mean(pos_delta)

    print(diff_x, diff_y, diff)

    factor = 1.5
    map_local = map_local.astype(np.float)
    map_local /= factor
    map_local += (255 - 255 / factor)
    map_local = map_local.astype(np.uint8)

    plot_gps_trajectory(map_local, gps_pos_lmap)
    plot_tracking_trajectory(map_local, tracking_pos_lmap)

    map_local = cv2.resize(map_local, (720, 1280))
    cv2.imshow('map_local', map_local)

    c = cv2.waitKey(-1)




if __name__ == '__main__':
    

    #test_valid_gps_cmt()

    test_valid_gps_rt()

