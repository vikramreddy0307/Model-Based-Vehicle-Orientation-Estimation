'''
Created on May 12, 2020

@author: duolu
'''

import os

import math
import numpy as np
import cv2


class CameraShakeRectifier():
    
    
    def __init__(self, frame_bg_ref, mask_bg_ref):
        
        
        
        # Compute sparse optical flow (Lucas-Kanade) on cornner features (Shi-Tomasi)
        # We use sparse optical flow to estimate camera shake and remove it.
        
        corner_params = dict( maxCorners = 200,
                           qualityLevel = 0.01,
                           minDistance = 10,
                           blockSize = 7 )
        
        bg_gray = cv2.cvtColor(frame_bg_ref, cv2.COLOR_BGR2GRAY)
        #bg_gray = cv2.GaussianBlur(bg_gray, (3, 3), cv2.BORDER_DEFAULT)
        #bg_gray = cv2.Canny(bg_gray, 50,150)
        mask_gray = cv2.cvtColor(mask_bg_ref, cv2.COLOR_BGR2GRAY)
        corners_bg = cv2.goodFeaturesToTrack(bg_gray, mask=mask_gray, **corner_params)
    
        #cv2.imshow('canny', bg_gray)
    
        of_params = dict( winSize  = (21,21),
                          maxLevel = 3,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        
        self.corner_params = corner_params
        self.mask_gray = mask_gray
        self.of_params = of_params
        
        self.corners_bg = corners_bg
        self.bg_gray = bg_gray
        
        self.alpha = 0.3
        
        self.first_frame = True
        self.tx = 0.0
        self.ty = 0.0
        self.theta = 0.0
        
        pass
    
    
    
    def rectify(self, frame):
        
        sof_check_th = 2.0

        corners_bg = self.corners_bg
        bg_gray = self.bg_gray
        
        of_params = self.of_params
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame_gray = cv2.GaussianBlur(frame_gray, (3, 3), cv2.BORDER_DEFAULT)
        #frame_gray = cv2.Canny(frame_gray, 50,150)
        corners_frame, st, err = cv2.calcOpticalFlowPyrLK(
            bg_gray, frame_gray, corners_bg, None, **of_params)
        corners_bg_check, st, err = cv2.calcOpticalFlowPyrLK(
            frame_gray, bg_gray, corners_frame, None, **self.of_params)
        
        d = abs(corners_bg - corners_bg).reshape(-1, 2).max(-1)
        status = d < sof_check_th

        corners_new = corners_frame[st == 1]
        corners_old = corners_bg[st == 1]
        
        n = corners_new.shape[0]
        
        frame_rect = self.rectify_shake_homography(frame, corners_new, corners_old, n)
        
        # draw the tracks
        for i, (new, old) in enumerate(zip(corners_new, corners_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.circle(frame, (a, b), 1, (255, 0, 0), 2)
            cv2.line(frame, (a, b), (c, d), (0, 0, 255), 2)
        
        frame_rect_gray = cv2.cvtColor(frame_rect, cv2.COLOR_BGR2GRAY)
        corners_rect = cv2.goodFeaturesToTrack(
            frame_rect_gray, mask=self.mask_gray, **self.corner_params)
        
        # for corner in corners_rect:
            
        #     a, b = corner.ravel()
        #     cv2.circle(frame_rect,(a,b), 1, (255, 0, 0), 2)
        
        return frame_rect

    def rectify_shake_translation(self, frame, delta, ps_new, ps_old, n):
        
        mean_delta = np.mean(delta, axis=0)
        
        tx = mean_delta[0]
        ty = mean_delta[1]
        
        if not self.first_frame:
            
            tx = self.alpha * self.tx + (1 - self.alpha) * tx
            ty = self.alpha * self.ty + (1 - self.alpha) * ty
        
        self.tx = tx
        self.ty = ty
        
        A = np.zeros((2, 3))
        A[0, 0] = 1
        A[1, 1] = 1
        A[0, 2] = -tx
        A[1, 2] = -ty
        
         
        frame_rect = cv2.warpAffine(frame, A, (frame.shape[1], frame.shape[0]))
        
        return frame_rect
    
    def rectify_shake_euclidean(self, frame, delta, ps_new, ps_old, n):
        
        A = np.zeros((2 * n, 4))
        b = np.zeros((2 * n, 1))
        
        for i in range(n):
            
            x = ps_new[i, 0]
            y = ps_new[i, 1]
            
            u = ps_old[i, 0]
            v = ps_old[i, 1]
            
            A[2 * i, 0] = x
            A[2 * i, 1] = -y
            A[2 * i, 2] = 1
            
            A[2 * i + 1, 0] = y
            A[2 * i + 1, 1] = x
            A[2 * i + 1, 3] = 1
            
            b[2 * i] = u
            b[2 * i + 1] = v
        
        p_inv = np.linalg.inv(np.matmul(A.T, A))
        atb = np.matmul(A.T, b)
        
        x = np.matmul(p_inv, atb)
        
        # cos(theta)
        c = x[0, 0]
        # sin(theta)
        s = x[1, 0]
        
        # CAUTION: c^2 + s^2 should be one but we did not check this
        # constraints here.
        
        theta = math.atan2(s, c)
        
        tx = x[2, 0]
        ty = x[3, 0]
        
        if not self.first_frame:
            
            tx = self.alpha * self.tx + (1 - self.alpha) * tx
            ty = self.alpha * self.ty + (1 - self.alpha) * ty
            
            theta = self.alpha * self.theta + (1 - self.alpha) * theta
        
        self.tx = tx
        self.ty = ty
        self.theta = theta
        
        c = math.cos(theta)
        s = math.sin(theta)
        
        
        T = np.zeros((2, 3))
        T[0, 0] = c
        T[0, 1] = -s
        T[0, 2] = tx
        T[1, 0] = s
        T[1, 1] = c
        T[1, 2] = ty
        
        #print(T)
        
        #H, mask = cv2.findHomography(ps_src, ps_dst)
        
        #print(H)
        
        frame_rect = cv2.warpAffine(frame, T, (frame.shape[1], frame.shape[0]))
    
        
        return frame_rect
    
    
    def rectify_shake_affine(self, frame, delta, ps_new, ps_old, n):
        
        A = np.zeros((2 * n, 6))
        b = np.zeros((2 * n, 1))
        
        for i in range(n):
            
            x = ps_new[i, 0]
            y = ps_new[i, 1]
            
            u = ps_old[i, 0]
            v = ps_old[i, 1]
            
            A[2 * i, 0] = x
            A[2 * i, 1] = y
            A[2 * i, 2] = 1
            
            A[2 * i + 1, 3] = x
            A[2 * i + 1, 4] = y
            A[2 * i + 1, 5] = 1
            
            b[2 * i] = u
            b[2 * i + 1] = v
        
        p_inv = np.linalg.inv(np.matmul(A.T, A))
        atb = np.matmul(A.T, b)
        
        x = np.matmul(p_inv, atb)
        
        T = np.zeros((2, 3))
        T[0, :] = x[0:3, 0]
        T[1, :] = x[3:6, 0]
        
        #print(T)
        
        #H, mask = cv2.findHomography(ps_src, ps_dst)
        
        #print(H)
        
        frame_rect = cv2.warpAffine(frame, T, (frame.shape[1], frame.shape[0]))
        
        #frame_rect = cv2.warpPerspective(frame, H, (frame.shape[1], frame.shape[0]))
        
        return frame_rect

    def rectify_shake_homography(self, frame, corners_new, corners_old, n):
        
        H, mask = cv2.findHomography(corners_new, corners_old, method=cv2.RANSAC, 
            ransacReprojThreshold=3)

        if self.first_frame:
            self.H = H
            self.first_frame = False
        else:
            H = self.alpha * H + (1 - self.alpha) * self.H
            self.H = H
        
        frame_rect = cv2.warpPerspective(frame, H, (frame.shape[1], frame.shape[0]))
        
        return frame_rect
 


class BackgroundGaussian():

    def __init__(self, n, dtype, height, width, channel):
        
        self.n = n
        self.height = height
        self.width = width
        self.i = 0
        
        self.frames = np.zeros((height, width, channel))
        
        self.ready = False
        #print(frames.shape)
        
        pass
    
    
    
    def update(self, frame):
        
        self.frames += frame
        
        self.i += 1
        
    
    
    
    
    
    def get_background(self):
        
        frame_bg = self.frames / self.i
        
        return frame_bg.astype(np.uint8)
        



class KLTOpticalFlowTracker():
    
    
    def __init__(self, mask_bg):
        
        mask_gray = cv2.cvtColor(mask_bg, cv2.COLOR_BGR2GRAY)
        
        self.mask_gray = mask_gray
        
        pass
    
    
    
    def track_iteration_online(self, prev_frame, this_frame):
        
        mask_gray = self.mask_gray
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        this_gray = cv2.cvtColor(this_frame, cv2.COLOR_BGR2GRAY)
        
        corner_params = dict( maxCorners = 2000,
                           qualityLevel = 0.02,
                           minDistance = 5,
                           blockSize = 3 )
        
        corners_this = cv2.goodFeaturesToTrack(this_gray, mask = mask_gray, **corner_params)
        #corners_this = cv2.goodFeaturesToTrack(this_gray, **corner_params)

        of_params = dict( winSize  = (7,7),
                          maxLevel = 9,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.01))

        corners_prev, st, err = cv2.calcOpticalFlowPyrLK(
            this_gray, prev_gray, corners_this, None, **of_params)

        #print(corners_this.shape)
        #print(corners_prev.shape)
        #print(st.shape)


        # draw the tracks
#         k = 2
#         for new, old, s in zip(corners_this, corners_prev, st):
#              
#             a, b = new.ravel()
#             c, d = old.ravel()
#  
#             #print(s, end=' ')
#             #print()
#              
#             if s != 1:
#                  
#                 cv2.circle(this_frame_vis,(a,b), 1, (0, 0, 255), 2)
#                  
#             else:
#                  
#                 #cv2.circle(this_frame_vis,(c, d), 1, (0, 0, 255), 2)
#                 cv2.circle(this_frame_vis,(a,b), 1, (0, 0, 255), 2)
#                 cc = int(a + k * (c - a))
#                 dd = int(b + k * (d - b))
#                 cv2.line(this_frame_vis, (a, b), (cc, dd), (0, 0, 255), 2)


        
        corners_this = corners_this[st==1]
        corners_prev = corners_prev[st==1]


        return corners_this, corners_prev









def test_shake_elimination(camera_id, track_id):

    #save = True
    save = False

    show = True
    #show = False

    frame_resize = (1280, 720)
    #frame_resize = (1920, 1080)

    folder = '../avacar_data'
    
    
    prefixes = ['westbound', 'eastbound', 'northbound', 'southbound', 'osburn1004']
    prefix = prefixes[camera_id]
    postfix = ('_%d' % track_id)

    calibration_folder = folder + '/calibration_2d/' + prefix
    
    bg_fn = calibration_folder + '/'  + prefix + '_bg_ref.png'
    mask_fn = calibration_folder + '/'  + prefix + '_bg_mask.png'
    
    frame_bg = cv2.imread(bg_fn)
    mask_bg = cv2.imread(mask_fn)

    frame_bg = cv2.resize(frame_bg, frame_resize)
    mask_bg = cv2.resize(mask_bg, frame_resize)

    
    
    video_fn = folder + '/video/' + prefix + postfix + '.mpg'
    #video_fn = folder + '/aerial_video/' + prefix + '.MP4'
    
    cap = cv2.VideoCapture(video_fn)
    assert cap.isOpened()


    csr = CameraShakeRectifier(frame_bg, mask_bg)
    
    rectified_folder = folder + '/rectified_frames/' + prefix + postfix
    if not os.path.isdir(rectified_folder):
        os.mkdir(rectified_folder)

    rectified_prefix = rectified_folder + '/' + prefix + postfix

    print(frame_bg.shape)
    
    frame_id = 0
    
    while True:
        
        if not cap.isOpened():
            break
        
        ret, frame = cap.read()
        if frame is None:
            break
        
        frame = cv2.resize(frame, frame_resize)

        frame_rect = csr.rectify(frame)
        
        print(frame_id)
        frame_id += 1
        
        if save:

            rectified_fn = rectified_prefix + ('_%d' % frame_id) + '.png'
            cv2.imwrite(rectified_fn, frame_rect)

        if show:

            cv2.imshow('rect',frame_rect)
            cv2.imshow('frame',frame)
            
            c = cv2.waitKey(-1)
            
            if c & 0xFF == ord('q'):
                break

    
    cap.release()
    cv2.destroyAllWindows()



def test_background_subtraction():

    cam_id = 3
    track_id = 0
    
    folder = '../avacar_data'
    
    
    prefixes = ['westbound', 'eastbound', 'northbound', 'southbound', 'osburn0723']
    prefix = prefixes[cam_id]
    postfix = ('_%d' % track_id)

    calibration_folder = folder + '/calibration_2d/' + prefix
    
    bg_fn = calibration_folder + '/'  + prefix + '_bg_ref.png'
    mask_fn = calibration_folder + '/'  + prefix + '_bg_mask.png'
    ofmask_fn = mask_fn


    frame_bg = cv2.imread(bg_fn)
    mask_bg = cv2.imread(mask_fn)
    mask_of = cv2.imread(ofmask_fn)

    
    frame_resize = (1280, 720)
    
    frame_bg = cv2.resize(frame_bg, frame_resize)
    mask_bg = cv2.resize(mask_bg, frame_resize)
    mask_of = cv2.resize(mask_of, frame_resize)


    video_fn = folder + '/video/'  + prefix + postfix + '.mpg'
    
    cap = cv2.VideoCapture(video_fn)
    assert cap.isOpened()


    csr = CameraShakeRectifier(frame_bg, mask_bg)
    
    print(frame_bg.shape)
    
    nr_bg_frames = 10
    
    #bgm = BackgroundGaussian(nr_bg_frames, frame_bg.dtype, *frame_bg.shape)
    
    bgm = cv2.createBackgroundSubtractorMOG2(history=100)
    
    bgm.setShadowThreshold(0.6)
    
    print(bgm.getShadowThreshold())
    print(bgm.getVarThresholdGen())
    
    
    oftraker = KLTOpticalFlowTracker(mask_of)
    
    
    for frame_id in range(nr_bg_frames):
    
        print(frame_id)
    
        ret, frame_prev = cap.read()
        frame_prev = cv2.resize(frame_prev, frame_resize)

        frame_prev = csr.rectify(frame_prev)
        
        #frame_prev = cv2.GaussianBlur(frame_prev,(3,3),cv2.BORDER_DEFAULT)
        fgmask = bgm.apply(frame_prev)
        
        
        
    
    while True:
        
        if not cap.isOpened():
            break
        
        ret, frame = cap.read()
        frame = cv2.resize(frame, frame_resize)
        
        frame_rect = csr.rectify(frame)

        frame_prev_vis = frame_prev.copy()
        frame_rect_vis = frame_rect.copy()
        
        #frame_rect = cv2.GaussianBlur(frame_rect,(3,3),cv2.BORDER_DEFAULT)
        fgmask = bgm.apply(frame_rect)
        fgmask = cv2.bitwise_not(fgmask)
        
        
        print(frame_id)
        frame_id += 1
        

        oftraker.track_iteration_online(frame_prev, frame_rect)
  
  
        frame_diff = cv2.absdiff(frame_rect, frame_prev)
        frame_diff = cv2.bitwise_not(frame_diff)
        
        frame_prev = frame_rect
        
  
        frame_fg = cv2.absdiff(frame_rect, frame_bg)
         
        frame_fg = cv2.GaussianBlur(frame_fg,(9,9),cv2.BORDER_DEFAULT)
         
        ret, frame_fg_th = cv2.threshold(frame_fg, 30, 255, cv2.THRESH_BINARY)
        frame_fg = cv2.bitwise_not(frame_fg)
        frame_fg_th = cv2.bitwise_not(frame_fg_th)

        kernel1 = np.ones((3,3),np.uint8)
        kernel2 = np.ones((3,3),np.uint8)
        frame_fg_th = cv2.dilate(frame_fg_th, kernel1,iterations = 1)
        frame_fg_th = cv2.erode(frame_fg_th, kernel2,iterations = 1)

        #fgmask = cv2.dilate(fgmask, kernel1,iterations = 1)
        #fgmask = cv2.erode(fgmask, kernel2,iterations = 1)

    
        cv2.imshow('diff',frame_diff)
        cv2.imshow('fg',fgmask)
        cv2.imshow('th',frame_fg_th)
        cv2.imshow('rect',frame_rect)#[40:-10, 10:-10])
        cv2.imshow('vis',frame_rect_vis)
        #cv2.imshow('frame',frame)
          
        c = cv2.waitKey(-1)
           
        if c & 0xFF == ord('q'):
            break

        
        
#     frame_bg = bgm.get_background()
#      
#     cv2.imwrite('./car_data/westbound_bg.png', frame_bg)
#      
#      
#     cv2.imshow('background',frame_bg)
#      
#     c = cv2.waitKey(-1)
    
        
    
        
    
    cap.release()
    cv2.destroyAllWindows()







if __name__ == '__main__':
    
    
    test_shake_elimination(2, 100)

    # for camera_id in range(0, 4):
    #     for track_id in range(0, 1):
    #         test_shake_elimination(camera_id, track_id)

    #test_background_subtraction()

    pass