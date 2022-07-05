'''
Created on Aug 18, 2020

@author: duolu
'''

import os

import warnings

import numpy as np
import cv2

from background import CameraShakeRectifier
from tracking import TrackerConf


def calculate_box_size(box):
    '''
    Calculate the size of a 2D bounding box.
    '''
    
    p1x = box[0]
    p1y = box[1]
    p2x = box[2]
    p2y = box[3]
    
    w = p2x - p1x
    h = p2y - p1y
    
    return abs(w * h)

class AnnotatedVehicle(object):
    '''The class for a single annotated vehicle.
    '''

    def __init__(self, vid, type_code):

        self.vid = vid
        self.type_code = type_code

        self.box = None
        self.patch = None

        self.frame_id = -1
        self.instance_id = -1

    def set_patch(self, box, patch):

        if self.box is not None and self.patch is not None:

            box_size_self = calculate_box_size(self.box)
            box_size = calculate_box_size(box)

            if box_size > box_size_self:

                self.box = box
                self.patch = patch

        else:

            self.box = box
            self.patch = patch



def load_labels(labels_fn):


    labels = np.loadtxt(labels_fn, dtype=np.int, delimiter=',')
    n = labels.shape[0]
    ann_dict = {}
    for i in range(n):
        vid = labels[i, 0]
        # NOTE: There are labels mistakes in these two fields,
        # i.e., "ann_frame_id" and "instance_id". Do not rely on them.
        # ann_frame_id = labels[i, 1]
        # instance_id = labels[i, 2]
        type_code = labels[i, 3]

        ann_vehicle = AnnotatedVehicle(vid, type_code)
        ann_vehicle.frame_id = labels[i, 1]
        ann_vehicle.instance_id = labels[i, 2]

        ann_dict[vid] = ann_vehicle
        
    return ann_dict

def save_dataset(file_name_prefix, ann_dict):

    for vid, v in ann_dict.items():

        if v.patch is None:
            continue

       
        fn_suffix = '_%d_%d_%d.png' \
            % (vid, v.frame_id, v.instance_id)
        file_name = file_name_prefix + fn_suffix
        #print('saving: ', file_name)
        cv2.imwrite(file_name, v.patch)

def save_labels(file_name, ann_dict):

    labels = []

    for vid, v in ann_dict.items():

        if v.patch is None:
            continue

        v_tup = (vid, v.frame_id, v.instance_id, v.type_code)

        labels.append(v_tup)
    
    labels_array = np.asarray(labels)
    np.savetxt(file_name + '.csv', labels_array, fmt='%d', delimiter=',')




def check_box_outside_view(box, w=1280, h=720, 
    h_margin=20, v_margin=20, h1=0, h2=0, v1=100, v2=0):
    '''
    Check whether a 2D bounding box is outside of view.
    '''
    
    p1x = box[0]
    p1y = box[1]
    p2x = box[2]
    p2y = box[3]
    
    
    return p1x < (h_margin + h1) or p2x > (w - h_margin - h2) \
        or p1y < (v_margin + v1) or p2y > (h - v_margin - v2)


def load_data_file(file_name, fmt, is_integer):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if fmt == 'csv':
            if is_integer:
                array = np.loadtxt(file_name + '.csv', delimiter=',', dtype=np.int)
            else:
                array = np.loadtxt(file_name + '.csv', delimiter=',')
        elif fmt == 'npy':
            array = np.load(file_name + '.npy')
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


def prepare_dataset(camera_id, track_id):

    need_csr = True
    #need_csr = False

    has_labels = True
    #has_labels = False

    #fmt = 'npy'
    fmt = 'csv'

    folder = '../avacar_data'
    
    
    prefixes = ['westbound', 'eastbound', 'northbound', 'southbound', 'osburn0723']
    prefix = prefixes[camera_id]
    postfix = ('_%d' % track_id)

    calibration_folder = folder + '/calibration_2d/' + prefix
    
    bg_fn = calibration_folder + '/'  + prefix + '_bg_ref.png'
    mask_fn = calibration_folder + '/'  + prefix + '_bg_mask.png'
    
    frame_bg = cv2.imread(bg_fn)
    mask_bg = cv2.imread(mask_fn)

    if has_labels:
        labels_folder = folder + '/vehicle_annotation/'
        labels_fn = labels_folder + prefix + postfix + '_annotation.' + fmt
        ann_dict = load_labels(labels_fn)
    else:
        ann_dict = {}


    print('processing ' + prefix + postfix)

    video_fn = folder + '/video/' + prefix + postfix + '.mpg'
    
    cap = cv2.VideoCapture(video_fn)
    assert cap.isOpened()


    csr = CameraShakeRectifier(frame_bg, mask_bg)

    tconf = TrackerConf(folder, prefix, postfix)

    frame_id = 0

    # NOTE: Skip the intial frame.
    _ret, frame = cap.read()
    if frame is None:
        return

    for frame_id in range(1, 18000 - 500):

        #print(frame_id)
        
        _ret, frame = cap.read()
        if frame is None:
            break
        
        if need_csr:
            frame_rect = csr.rectify(frame)
        else:
            frame_rect = frame

        if frame_id % 1000 == 0:
            print(frame_id)

        tracking_fn = tconf.get_fn_per_frame('tracking', fmt, 'tracking', frame_id)
        tracking_data = load_data_file(tracking_fn, fmt, is_integer=False)

        i2v_fn = tconf.get_fn_per_frame('vehicle', fmt, 'i2v', frame_id)
        i2v_data = load_data_file(i2v_fn, fmt, is_integer=True)

        if tracking_data is None or i2v_data is None:
            continue

        # Update the patch of the annotated vehicle.
        n = tracking_data.shape[0]
        for i in range(n):

            vid_i = i2v_data[i, 1]
            if vid_i == -1:
                continue

            if has_labels:
                if vid_i not in ann_dict:
                    continue
                
                ann_vehicle = ann_dict[vid_i]
            else:
                if vid_i not in ann_dict:
                    ann_vehicle = AnnotatedVehicle(vid_i, 0)
                    ann_dict[vid_i] = ann_vehicle
                else:
                    ann_vehicle = ann_dict[vid_i]
            

            box = tracking_data[i, 13:17].astype(np.int)

            if check_box_outside_view(box):
                continue

            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            w = x2 - x1
            h = y2 - y1

            if w > h:

                y1 = cy - w // 2
                if y1 < 0:
                    y1 = 0
                y2 = y1 + w

            else:

                x1 = cx - h // 2
                if x1 < 0:
                    x1 = 0
                x2 = x1 + h

            patch = frame_rect[y1:y2, x1:x2]

            
            ann_vehicle.set_patch(box, patch)
            ann_vehicle.frame_id = frame_id
            ann_vehicle.instance_id = i


    dataset_folder = folder + '/vehicle_type_dataset/' + prefix + postfix
    if not os.path.isdir(dataset_folder):
        os.mkdir(dataset_folder)
    file_name_prefix = dataset_folder + '/' + prefix + postfix
    save_dataset(file_name_prefix, ann_dict)
    
    labels_file_name = folder + '/vehicle_type_labels/' + prefix + postfix + '_labels'
    save_labels(labels_file_name, ann_dict)

def annotate_dataset(camera_id, track_id):


    folder = '../avacar_data'
    
    
    prefixes = ['westbound', 'eastbound', 'northbound', 'southbound', 'osburn0723']
    prefix = prefixes[camera_id]
    postfix = ('_%d' % track_id)

    labels_folder = folder + '/vehicle_type_labels_todo/'
    labels_file_name = labels_folder + prefix + postfix + '_labels'
    ann_dict = load_labels(labels_file_name + '.csv')

    dataset_folder = folder + '/vehicle_type_dataset_todo/' + prefix + postfix

    for vid, v in ann_dict.items():

        fn_suffix = '_%d_%d_%d_%d.png' \
            % (vid, v.frame_id, v.instance_id, v.type_code)
        image_fn = dataset_folder + '/' + prefix + postfix + fn_suffix
        #print(image_fn)
        print(vid, v.frame_id, v.instance_id, v.type_code)

        image = cv2.imread(image_fn, cv2.IMREAD_COLOR)
        v.patch = image

        image_vis = cv2.resize(image, (512, 512))
        cv2.imshow('image', image_vis)

        c = cv2.waitKey(-1)
        
        if c & 0xFF == ord('q'):
            break

        code = c & 0xFF - ord('0')
        if code >= 0 and code <= 9:

            if code == 0:
                code = 10

            print('set vehicle %d to type %d.' % (vid, code))
            v.type_code = code

    dataset_folder = folder + '/vehicle_type_dataset/' + prefix + postfix
    if not os.path.isdir(dataset_folder):
        os.mkdir(dataset_folder)
    file_name_prefix = dataset_folder + '/' + prefix + postfix
    save_dataset(file_name_prefix, ann_dict)

    labels_file_name = folder + '/vehicle_type_labels/' + prefix + postfix + '_labels'
    save_labels(labels_file_name, ann_dict)

    print('done', labels_file_name)


if __name__ == '__main__':
    
    
    #prepare_dataset(3, 3)


    # for cid in range(0, 4):
    #     for tid in range(0, 4):

    #         prepare_dataset(cid, tid)


    annotate_dataset(2, 4)









