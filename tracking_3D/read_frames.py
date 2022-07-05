'''
Created on May 13, 2020

@author: duolu
'''

import os

import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


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

    else:

        max_k = 0
        max_area = 0
        for k, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_k = k

        contour = contours[max_k]

    contour = contour.reshape((-1, 2))

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


class DetectorMaskRCNN(Detector):
    '''
    The detector that uses Mask RCNN implemented in FAIR Detectron2.
    '''

    def __init__(self, folder, prefix, postfix):
        Detector.__init__(self, folder, prefix, postfix)

        config_fn = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
        model_fn = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
        #        config_fn = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        #        model_fn = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

        # get the mask using RCNN
        config = get_cfg()
        config.merge_from_file(model_zoo.get_config_file(config_fn))
        config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
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

        masks = np.zeros((n, frame.shape[0], frame.shape[1]), dtype=bool)

        # NOTE: Only the mask within the bounding box is saved.
        for i in range(n):
            i_str = '_%d' % i

            box = boxes[i]
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            fn_mask_i = fn_masks + i_str

            mask_i = np.load(fn_mask_i + '.npy')
            masks[i][y1:y2, x1:x2] = mask_i

        labels = np.load(fn_labels + '.npy')
        scores = np.load(fn_scores + '.npy')

        centers = np.zeros((n, 2))
        contours = []

        for i in range(n):
            centers[i] = compute_center_from_box(boxes[i])
            contour = compute_contour_from_mask(masks[i])
            contours.append(contour)

        return boxes, centers, masks, contours, labels, scores


def visualize_mask(boxes, masks, mask_vis, frame_vis, colors):
    for k, (box, mask) in enumerate(zip(boxes, masks)):
        c = k % colors.shape[0]
        color = colors[c]
        color_tuple = (int(color[0]), int(color[1]), int(color[2]))

        mask_vis[masks[k]] = color

        p1x = int(box[0])
        p1y = int(box[1])
        p2x = int(box[2])
        p2y = int(box[3])

        # cv2.rectangle(frame_vis, (p1x, p1y), (p2x, p2y), color_tuple, 2)


def test_detection(camera_id, track_id):
    # need_csr = True

    # NOTE: Frames will be resized to the size of this "frame_resize".
    frame_resize = (1280, 720)
    # frame_resize = (1920, 1080)

    folder = './avacar_data'

    prefixes = ['westbound', 'eastbound', 'northbound', 'southbound', 'osburn0723', 'aerial_osburn0723']
    prefix = prefixes[camera_id]
    postfix = ('_%d' % track_id)

    video_fn = folder + '/video/' + prefix + postfix + '_frame.mp4'
    print(video_fn)

    cap = cv2.VideoCapture(video_fn)
    assert cap.isOpened()

    cap_mask = cv2.VideoCapture(folder + '/video/' + prefix + postfix + '_detectron.mp4')

    frame_id = 0
    DATA_ROOT = './Dataset'
    IMG_PATH = DATA_ROOT + '/Images/'
    MASK_PATH = DATA_ROOT + '/Masks/'
    # import os
    if not os.path.exists(DATA_ROOT):
        os.makedirs(DATA_ROOT)
        os.makedirs(IMG_PATH)
        os.makedirs(MASK_PATH)

    readCorrect = True
    while (readCorrect):
        frame_id += 1
        readCorrect, frame = cap.read()
        ret, maskFrame = cap_mask.read()
        readCorrect = readCorrect and ret

        if readCorrect:
            cv2.imshow('Masks', maskFrame)
            r = cv2.selectROI(frame, showCrosshair=False)
            # print('r :{}'.format(r))

            if r[0] != 0 and r[1] != 0 and r[2] != 0 and r[3] != 0:
                # Crop image, frame number
                imCrop = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

                mask_crop = maskFrame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

                fname = prefix + postfix + '_' + str(frame_id) + '.png'
                print(fname)
                cv2.imwrite(IMG_PATH + fname, imCrop)
                cv2.imwrite(MASK_PATH + fname, mask_crop)

        c = cv2.waitKey(1)

        # add options here
        # if key == ord('s'):

        if c & 0xFF == ord('q'):
            print("Did this")
            break

    cap.release()
    cap_mask.release()
    # outvideo_frame.release()
    # outvideo_detectron.release()
    cv2.destroyAllWindows()


def test_detector_fine_tuning():
    pass


if __name__ == '__main__':
    test_detection(0, 0)

    # for cam_id in range(0, 4):
    #     for track_id in range(0, 13):
    #
    #         print('camera %d, track %d' % (cam_id, track_id))
    #
    #         test_detection(cam_id, track_id)

    pass
