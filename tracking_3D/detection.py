'''
Created on May 13, 2020

@author: duolu
'''

import colorsys
import os
import random
import time

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
# from detectron2 import model_zoo
# from detectron2.config import get_cfg
# from detectron2.engine import DefaultPredictor

import transforms as TT
import utils as utils
from background import CameraShakeRectifier
from engine import train_one_epoch


def mask_local_to_global(box, mask_local, frame_width, frame_height):
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])

    mask_global = np.zeros((frame_height, frame_width), dtype=bool)
    mask_global[y1:y2, x1:x2] = mask_local

    return mask_global


def compute_center_from_box(box):
    '''
    Compute the center of a 2D bounding box.
    '''

    p1x = box[0]
    p1y = box[1]
    p2x = box[2]
    p2y = box[3]

    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2

    center = np.zeros(2)
    center[0] = cx
    center[1] = cy

    return center


def compute_contour_from_mask(mask):
    '''
    Compute the contour from a segmentation mask.
    '''

    mask_image = np.zeros([mask.shape[0], mask.shape[1], 1], dtype=np.uint8)
    mask_image[mask] = 255

    contours, hierarchy = cv2.findContours(
        mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    n = len(contours)

    # print('contours: ', n)
    if n == 1:

        contour = contours[0]
        contour = contour.reshape((-1, 2))

    elif n > 1:

        max_k = 0
        max_area = 0
        for k, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_k = k

        contour = contours[max_k]
        contour = contour.reshape((-1, 2))

    else:

        contour = np.zeros((0, 2))

    return contour


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


def check_box_overlap(box1, box2):
    '''
    Check whether two 2D bounding boxes overlap.
    '''

    x11 = box1[0]
    y11 = box1[1]
    x12 = box1[2]
    y12 = box1[3]

    x21 = box2[0]
    y21 = box2[1]
    x22 = box2[2]
    y22 = box2[3]

    x_overlap = x12 >= x21 and x22 >= x11
    y_overlap = y12 >= y21 and y22 >= y11

    return x_overlap and y_overlap


def check_box_bottom_occlusion(box1, box2, margin=5):
    '''
    Check whether box1 is occluded by box2 on the bottom.
    '''

    x11 = box1[0]
    y11 = box1[1]
    x12 = box1[2]
    y12 = box1[3]

    x21 = box2[0]
    y21 = box2[1]
    x22 = box2[2]
    y22 = box2[3]

    xm = (x11 + x12) / 2
    xl = (x11 + xm) / 2
    xr = (x12 + xm) / 2
    xlc = x11
    xrc = x12

    flag_m = xm > x21 and xm < x22 and y12 > y21 - margin and y12 < y22 + margin
    flag_l = xl > x21 and xl < x22 and y12 > y21 - margin and y12 < y22 + margin
    flag_lc = xlc > x21 and xlc < x22 and y12 > y21 - margin and y12 < y22 + margin
    flag_r = xr > x21 and xr < x22 and y12 > y21 - margin and y12 < y22 + margin
    flag_rc = xrc > x21 and xrc < x22 and y12 > y21 - margin and y12 < y22 + margin

    # print('\t', flag_l, flag_m, flag_r)

    return flag_l and flag_lc and flag_m and flag_r and flag_rc


def check_box_on_mask_bg(box, mask_bg):
    w = mask_bg.shape[1]
    h = mask_bg.shape[0]

    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])

    if x1 < 0:
        x1 = 0
    if x2 >= w:
        x2 = w - 1
    if y1 < 0:
        y1 = 0
    if y2 >= h:
        y2 = h - 1

    tl_flag = mask_bg[y1, x1]
    tr_flag = mask_bg[y1, x2]
    bl_flag = mask_bg[y2, x1]
    br_flag = mask_bg[y2, x2]

    return tl_flag or tr_flag or bl_flag or br_flag


def merge_boxes(box1, box2):
    '''
    Merge two 2D bounding box.
    '''

    x11 = box1[0]
    y11 = box1[1]
    x12 = box1[2]
    y12 = box1[3]

    x21 = box2[0]
    y21 = box2[1]
    x22 = box2[2]
    y22 = box2[3]

    x_min = min(x11, x21)
    x_max = max(x12, x22)
    y_min = min(y11, y21)
    y_max = max(y12, y22)

    box1[0] = x_min
    box1[1] = y_min
    box1[2] = x_max
    box1[3] = y_max

    return box1


def calculate_mask_overlap(mask1, mask2):
    '''
    Calculate the overlap of two segmentation masks.
    '''

    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    area_i = np.sum(intersection)
    area_u = np.sum(union)
    area_1 = np.sum(mask1)
    area_2 = np.sum(mask2)

    iou = area_i / area_u
    io1 = area_i / area_1
    io2 = area_i / area_2

    return iou, io1, io2


class Detector(object):
    '''
    The detector base class.
    '''

    def __init__(self, folder, prefix, postfix):
        self.folder = folder
        self.prefix = prefix
        self.postfix = postfix

    def get_file_names(self, frame_id):
        frame_id_str = '_%d' % frame_id

        common_prefix = self.folder + '/' + self.prefix + self.postfix

        fn_boxes = common_prefix + '_boxes' + frame_id_str
        fn_masks = common_prefix + '_masks' + frame_id_str
        fn_labels = common_prefix + '_classes' + frame_id_str
        fn_scores = common_prefix + '_scores' + frame_id_str

        return fn_boxes, fn_masks, fn_labels, fn_scores

    def save_results(self, frame_id, results):
        '''
        Save detection results to files.

        NOTE: "centers" and "contours" are not saved. They are always
        computed from "boxes" and "masks".
        '''

        boxes, centers, masks, contours, labels, scores = results

        file_names = self.get_file_names(frame_id)

        fn_boxes, fn_masks, fn_labels, fn_scores = file_names

        np.save(fn_boxes, boxes)

        n = boxes.shape[0]

        # NOTE: Only the mask within the bounding box is saved.
        for i in range(n):
            i_str = '_%d' % i

            box = boxes[i]
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            mask_i = masks[i][y1:y2, x1:x2]
            fn_mask_i = fn_masks + i_str

            np.save(fn_mask_i, mask_i)

        np.save(fn_labels, labels)
        np.save(fn_scores, scores)


class DetectorTorchvisionMaskRCNN(Detector):

    def __init__(self, folder, prefix, postfix, pretrained=False):

        Detector.__init__(self, folder, prefix, postfix)

        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        if not pretrained:
            model.load_state_dict(torch.load('retrained_net.pth'))

        model.eval()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)

        self.model = model

    def detect(self, frame_id, frame):

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        tf = T.Compose([T.ToTensor()])
        frame_torch = tf(frame)

        frame_torch = frame_torch.to(device)
        pred = self.model([frame_torch])

        labels_raw = pred[0]['labels'].detach().cpu().numpy()
        scores_raw = pred[0]['scores'].detach().cpu().numpy()
        boxes_raw = pred[0]['boxes'].detach().cpu().numpy()
        masks_raw = (pred[0]['masks']).squeeze().detach().cpu().numpy()

        masks_raw = masks_raw > 0.5

        # print(scores_raw, labels_raw)
        # print(boxes_raw.shape, masks_raw.shape)

        n_raw = labels_raw.shape[0]
        indices = []

        for i in range(n_raw):

            if scores_raw[i] > 0.1:
                indices.append(i)

        n = len(indices)
        h = frame.shape[0]
        w = frame.shape[1]

        boxes = np.zeros((n, 4))
        masks = np.zeros((n, h, w), dtype=np.bool)

        labels = np.zeros(n, dtype=np.int)
        scores = np.zeros(n)

        for j in range(n):
            i = indices[j]

            boxes[j] = boxes_raw[i]
            masks[j] = masks_raw[i]

            labels[j] = labels_raw[i]
            scores[j] = scores_raw[i]

        centers = np.zeros((n, 2))
        contours = []

        for i in range(n):
            centers[i] = compute_center_from_box(boxes[i])
            contour = compute_contour_from_mask(masks[i])
            contours.append(contour)
        
        

        return boxes, centers, masks, contours, labels, scores


class DetectorMaskRCNN(Detector):
    '''
    The detector that uses Mask RCNN implemented in FAIR Detectron2.
    '''

    def __init__(self, folder, prefix, postfix):
        Detector.__init__(self, folder, prefix, postfix)

        # config_fn = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
        # model_fn = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
        config_fn = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        model_fn = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

        # get the mask using RCNN
        config = get_cfg()
        config.merge_from_file(model_zoo.get_config_file(config_fn))
        config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
        # cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 200
        # cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 50

        config.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_fn)

        predictor = DefaultPredictor(config)

        self.config = config
        self.predictor = predictor

    def detect(self, frame_id, frame):
        outputs = self.predictor(frame)

        predictions = outputs["instances"].to("cpu")

        boxes = np.asarray(predictions.pred_boxes.tensor)
        masks = np.asarray(predictions.pred_masks)
        labels = np.asarray(predictions.pred_classes)
        scores = np.asarray(predictions.scores)

        n = boxes.shape[0]

        centers = np.zeros((n, 2))
        contours = []

        for i in range(n):
            centers[i] = compute_center_from_box(boxes[i])
            contour = compute_contour_from_mask(masks[i])
            contours.append(contour)

        # print(labels)

        return boxes, centers, masks, contours, labels, scores


class DetectorWithSavedResults(Detector):
    '''
    The detector that uses saved detection results.
    '''

    def __init__(self, folder, prefix, postfix):

        Detector.__init__(self, folder, prefix, postfix)

    def detect(self, frame_id, frame):

        file_names = self.get_file_names(frame_id)
        fn_boxes, fn_masks, fn_labels, fn_scores = file_names

        boxes = np.load(fn_boxes + '.npy')

        n = boxes.shape[0]

        # print(n, frame.shape)

        masks = np.zeros((n, frame.shape[0], frame.shape[1]), dtype=bool)

        # NOTE: Only the mask within the bounding box is saved.
        for i in range(n):
            i_str = '_%d' % i

            box = boxes[i]
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            # print(i, y1, y2, x1, x2, masks[i].shape)

            fn_mask_i = fn_masks + i_str

            mask_i = np.load(fn_mask_i + '.npy')
            masks[i, y1:y2, x1:x2] = mask_i

        labels = np.load(fn_labels + '.npy')
        scores = np.load(fn_scores + '.npy')

        centers = np.zeros((n, 2))
        contours = []

        for i in range(n):
            centers[i] = compute_center_from_box(boxes[i])
            contour = compute_contour_from_mask(masks[i])
            contours.append(contour)

        return boxes, centers, masks, contours, labels, scores


def visualize_mask(boxes, masks, labels, scores, mask_vis, frame_vis, colors):
    areas = []

    margin = 0

    mask_vis[0:margin, :] = 0
    frame_vis[0:margin, :] = 0

    for k, (box, mask, label, score) in enumerate(zip(boxes, masks, labels, scores)):

        if score < 0.9:
            continue

        if label != 3 and label != 8:
            continue

        p1x = int(box[0])
        p1y = int(box[1])
        p2x = int(box[2])
        p2y = int(box[3])

        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)

        if cy < margin:
            continue

        c = k % colors.shape[0]
        color = colors[c]
        color_tuple = (int(color[0]), int(color[1]), int(color[2]))

        mask_vis[masks[k]] = color

        cv2.rectangle(frame_vis, (p1x, p1y), (p2x, p2y), color_tuple, 2)

        cv2.putText(mask_vis, '%2d' % (score * 100), (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        areas.append(np.sum(mask))

    # print(scores, labels)


def save_for_annotation(frame, frame_with_masks, results, prefix, postfix, fid):
    margin = 0

    boxes, centers, masks, contours, labels, scores = results

    folder = '/home/duolu/project_avacar/video_annotation/'
    folder = folder + prefix + postfix + '_%d' % fid

    frame[0:margin, :] = 0

    if not os.path.isdir(folder):
        os.mkdir(folder)

    fn_frame = folder + '/frame.png'
    cv2.imwrite(fn_frame, frame)

    fn_frame = folder + '/frame_with_masks.png'
    cv2.imwrite(fn_frame, frame_with_masks)

    n = boxes.shape[0]

    for i in range(n):

        box = boxes[i]
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)

        w = x2 - x1
        h = y2 - y1

        size = w * h

        if cy < margin:
            continue

        if scores[i] < 0.9:
            continue

        label = labels[i]
        if label != 3 and label != 8:
            continue

        fn_mask_i = folder + '/mask_%d.png' % i

        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        mask_img = np.zeros((frame_height, frame_width, 3), np.uint8)
        mask_img[masks[i]] = 255
        mask_img[0:margin, :] = 0
        cv2.imwrite(fn_mask_i, mask_img)

        fn_check_i = folder + '/check_%d.png' % i

        check_img = frame.copy()
        check_img[masks[i]] = np.asarray((255, 0, 0), dtype=np.uint8)
        check_img[0:margin, :] = 0
        cv2.imwrite(fn_check_i, check_img)


def test_detection(camera_id, track_id, model=None):
    need_csr = True
    # need_csr = False

    # use_saved_results = True
    use_saved_results = False

    save = True
    # save = False

    vis = True
    # vis = False

    # NOTE: Frames will be resized to the size of this "frame_resize".
    frame_resize = (1280, 720)
    # frame_resize = (1920, 1080)

    folder = '../avacar_data'

    prefixes = ['westbound', 'eastbound', 'northbound', 'southbound', 'osburn1004']
    prefix = prefixes[camera_id]
    postfix = ('_%d' % track_id)

    if need_csr:
        calibration_folder = folder + '/calibration_2d/' + prefix

        bg_fn = calibration_folder + '/' + prefix + '_bg_ref.png'
        mask_fn = calibration_folder + '/' + prefix + '_bg_mask.png'

        frame_bg = cv2.imread(bg_fn)
        mask_bg = cv2.imread(mask_fn)

        frame_bg = cv2.resize(frame_bg, frame_resize)
        mask_bg = cv2.resize(mask_bg, frame_resize)

        csr = CameraShakeRectifier(frame_bg, mask_bg)

    video_fn = folder + '/video/' + prefix + postfix + '.mpg'

    print(video_fn)

    cap = cv2.VideoCapture(video_fn)
    assert cap.isOpened()

    detection_folder = folder + '/detection/' + prefix + postfix

    if use_saved_results:
        detector = DetectorWithSavedResults(detection_folder, prefix, postfix)
    else:
        # detector = DetectorMaskRCNN(detection_folder, prefix, postfix)
        detector = DetectorTorchvisionMaskRCNN(detection_folder, prefix, postfix)
        if model is not None:
            detector.model = model

    if not os.path.isdir(detection_folder):
        os.mkdir(detection_folder)

    nr_colors = 15

    colors = np.zeros((nr_colors, 3), np.uint8)

    for i in range(nr_colors):
        color_tuple = colorsys.hsv_to_rgb(i / nr_colors, 1, 1)
        color = np.asarray(color_tuple)
        color *= 255
        colors[i] = color.astype(np.uint8)

    # Count FPS.
    ts_last = time.monotonic()
    counter = 0
    fps = 0

    frame_id = 0

    for frame_id in range(180000):

        if not cap.isOpened():
            break

        ret, frame = cap.read()
        if frame is None:
            break

        # print(frame_id)

        # if frame_id < 15000:
        #    continue

        if frame_id % 1000 == 0:
            print(frame_id)

        frame = cv2.resize(frame, frame_resize)

        if need_csr:
            frame_rect = csr.rectify(frame)
        else:
            frame_rect = frame

        results = detector.detect(frame_id, frame_rect)

        boxes, centers, masks, contours, labels, scores = results

        if save and not use_saved_results:
            detector.save_results(frame_id, results)

        ts = time.monotonic()
        counter += 1

        # if ts - ts_last > 1:

        #     fps = counter / (ts - ts_last)
        #     ts_last = ts
        #     counter = 0

        #     print(fps)

        # print(frame_id, boxes.shape[0])

        # continue

        # print(masks.shape, masks.dtype)

        frame_id += 1

        if vis:
            mask_vis = frame_rect.copy()
            frame_vis = frame_rect.copy()
            visualize_mask(boxes, masks, labels, scores, mask_vis, frame_vis, colors)

            cv2.imshow('masks', mask_vis)
            cv2.imshow('frame', frame_vis)

            c = cv2.waitKey(1)

            if c & 0xFF == ord('q'):
                break

            if c & 0xFF == ord('s'):
                save_for_annotation(frame_rect, mask_vis, results, prefix, postfix, frame_id)

    cap.release()
    cv2.destroyAllWindows()


class ImageDatasetForDetector(object):

    def __init__(self, folder, transforms):

        self.folder = folder
        self.transforms = transforms

    def rotate_image(self, image, angle):

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def load_metadata(self):

        folder = self.folder

        subfolders = os.listdir(folder)

        subfolders.sort()

        fns_list = []
        frame_fns = []

        for i, subfolder in enumerate(subfolders):
            subfolder = folder + '/' + subfolder
            fns = os.listdir(subfolder)

            frame_i_fn = subfolder + '/frame.png'

            fns_list.append(fns)
            frame_fns.append(frame_i_fn)

        self.subfolders = subfolders
        self.fns_list = fns_list
        self.frame_fns = frame_fns

    def load_images_and_masks(self, augment=False):
        '''
        Load all images and masks into the memory.

        NOTE: This is only for checking the annotated images.
        '''

        augment_multiplier = 0

        folder = self.folder

        subfolders = os.listdir(folder)

        subfolders.sort()

        # print(subfolders)

        frames = []
        boxes = []
        masks = []
        names = []

        fns_list = []
        frame_fns = []

        for i, subfolder in enumerate(subfolders):

            subfolder = folder + '/' + subfolder
            fns = os.listdir(subfolder)

            frame_i_fn = subfolder + '/frame.png'

            fns_list.append(fns)
            frame_fns.append(frame_i_fn)

            frame_i, masks_i, boxes_i = \
                self.load_one_image(subfolder, frame_i_fn, fns, False)
            if len(masks_i) > 0:
                frames.append(frame_i)
                boxes.append(np.array(boxes_i))
                masks.append(np.array(masks_i))
                names.append(subfolder)

            # NOTE: Random rotation for data augmentation.
            for j in range(augment_multiplier):

                frame_i, masks_i, boxes_i = \
                    self.load_one_image(subfolder, frame_i_fn, fns, True)

                if len(masks_i) > 0:
                    frames.append(frame_i)
                    boxes.append(np.array(boxes_i))
                    masks.append(np.array(masks_i))
                    names.append(subfolder)

            print('load image', i)
            # if i > 20:
            #     break

        self.frames = frames
        self.boxes = boxes
        self.masks = masks
        self.names = names

        self.subfolders = subfolders
        self.fns_list = fns_list
        self.frame_fns = frame_fns

        return frames, boxes, masks, subfolders

    def load_one_image(self, subfolder, frame_i_fn, fns, augment):

        # print(frame_i_fn)

        frame_i = cv2.imread(frame_i_fn)

        width = frame_i.shape[1]
        height = frame_i.shape[0]

        angle = random.randint(0, 180)
        ratio = random.randint(25, 120) / 50
        augment_width = int(width * ratio)
        augment_height = int(height * ratio)
        augment_size = (augment_width, augment_height)
        # augment_size = (1920, 1080)

        if augment:
            frame_i = cv2.resize(frame_i, augment_size)
            # frame_i = self.rotate_image(frame_i, angle)

        masks_i = []
        boxes_i = []

        for fn in fns:

            if fn.startswith('mask'):

                mask_fn = subfolder + '/' + fn

                mask_i = cv2.imread(mask_fn)
                # print(mask_fn)
                if augment:
                    mask_i = cv2.resize(mask_i, augment_size)
                    # mask_i = self.rotate_image(mask_i, angle)
                mask_i = cv2.cvtColor(mask_i, cv2.COLOR_BGR2GRAY)
                mask_i_th = mask_i > 127

                mask_size = np.sum(mask_i_th)
                # print(mask_size)

                if mask_size > 200:
                    pos = np.where(mask_i_th)
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])
                    box = [xmin, ymin, xmax, ymax]

                    masks_i.append(mask_i_th)
                    boxes_i.append(box)

        return frame_i, np.array(masks_i), np.array(boxes_i)

    def check_images_and_masks(self):

        frames = self.frames
        boxes = self.boxes
        masks = self.masks
        names = self.names

        nr_colors = 15

        colors = np.zeros((nr_colors, 3), np.uint8)

        for i in range(nr_colors):
            color_tuple = colorsys.hsv_to_rgb(i / nr_colors, 1, 1)
            color = np.asarray(color_tuple)
            color *= 255
            colors[i] = color.astype(np.uint8)

        for i, (frame, boxes_i, masks_i, name) in enumerate(zip(frames, boxes, masks, names)):

            print('check image', i, name)

            for k, (box, mask) in enumerate(zip(boxes_i, masks_i)):
                c = k % colors.shape[0]
                color = colors[c]
                color_tuple = (int(color[0]), int(color[1]), int(color[2]))

                frame[mask] = color

                p1x = int(box[0])
                p1y = int(box[1])
                p2x = int(box[2])
                p2y = int(box[3])

                # cv2.rectangle(frame, (p1x, p1y), (p2x, p2y), color_tuple, 2)

            cv2.imshow('frame', frame)

            c = cv2.waitKey(-1)

            if c & 0xFF == ord('q'):
                break

    def __getitem__(self, i):

        subfolder = self.subfolders[i]
        frame_i_fn = self.frame_fns[i]
        fns = self.fns_list[i]

        subfolder = self.folder + '/' + subfolder

        n = 0
        while n == 0:
            # NOTE: The images cannot be rotated arbitrily.
            frame_i, masks_i, boxes_i = \
                self.load_one_image(subfolder, frame_i_fn, fns, True)
            n = len(boxes_i)

        # CAUTION: all objects have the same label!!!
        labels = np.zeros((n,), dtype=np.int)
        labels[:] = 3

        boxes_torch = torch.as_tensor(boxes_i, dtype=torch.float32)
        labels_torch = torch.as_tensor(labels, dtype=torch.int64)
        masks_torch = torch.as_tensor(masks_i, dtype=torch.uint8)

        image_id_torch = torch.tensor([i])
        area_torch = (boxes_torch[:, 3] - boxes_torch[:, 1]) \
                     * (boxes_torch[:, 2] - boxes_torch[:, 0])
        iscrowd_torch = torch.zeros((n,), dtype=torch.int64)

        # print('converting tensor done', i)

        target = {}
        target["boxes"] = boxes_torch
        target["labels"] = labels_torch
        target["masks"] = masks_torch
        target["image_id"] = image_id_torch
        target["area"] = area_torch
        target["iscrowd"] = iscrowd_torch

        img, target = self.transforms(frame_i, target)

        return img, target

    def __len__(self):

        return len(self.subfolders)


def get_transform(train):
    transforms = []
    transforms.append(TT.ToTensor())
    if train:
        transforms.append(TT.RandomHorizontalFlip(0.5))
    return TT.Compose(transforms)


def test_detector_fine_tuning():
    folder = '/home/varun/PycharmProjects/analysis_suite/tracking_3d/retrain/video_annotation'

    ds = ImageDatasetForDetector(folder, get_transform(train=True))

    ds.load_metadata()
    # ds.load_images_and_masks()
    # ds.check_images_and_masks()
    # ds.prepare_torchvision_dataset()

    # return

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # split the dataset in train and test set
    indices = torch.randperm(len(ds)).tolist()
    ds_train = torch.utils.data.Subset(ds, indices[:])
    ds_test = torch.utils.data.Subset(ds, indices[-20:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        ds_train, batch_size=4, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        ds_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # # our dataset has two classes only - background and object
    # num_classes = 2

    # # get number of input features for the classifier
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # # replace the pre-trained head with a new one
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # # now get the number of input features for the mask classifier
    # in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # hidden_layer = 256
    # # and replace the mask predictor with a new one
    # model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
    #                                                    hidden_layer,
    #                                                    num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        # evaluate(model, data_loader_test, device=device)

    torch.save(model.state_dict(), 'retrained_net.pth')
    model.eval()
    # model._save_to_state_dict('retrained_net.pth')

    # for cam_id in range(2, 4):
    #     for track_id in range(0, 3):
    #     #for track_id in range(201, 202):
    #         test_detection(cam_id, track_id, model=model)

    print("That's it!")

    pass


def load_test_detector_fine_tuning():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.to(device)

    # torch.save(model.state_dict(), 'retrained_net.pth')
    model.load_state_dict(torch.load('retrained_net.pth'))
    model.eval()
    # model._save_to_state_dict('retrained_net.pth')

    for cam_id in range(1):
        # for track_id in range(0, 3):
        for track_id in range(201, 202):
            test_detection(cam_id, track_id, model=model)
    pass


if __name__ == '__main__':
    # test_detection(4, 1)

    # test_detection(1, 201)

    # test_detection(0, 100)

    # test_detection(3, 103)
    # test_detection(3, 105)

    # for track_id in range(104, 107):
    #     for  cam_id in range(0, 4): 

    #         print('camera %d, track %d' % (cam_id, track_id))

    #         test_detection(cam_id, track_id)

    # now = datetime.datetime.now()
    # print (now.strftime("%Y-%m-%d %H:%M:%S"))

    load_test_detector_fine_tuning()

    pass
