

import numpy as np
import cv2





def process_demo_videos(cam_id, track_id):

    fid_start = 15000
    nr_frames = 600



    prefixes = ['westbound', 'eastbound', 'northbound', 'southbound', 'osburn1004']
    prefix = prefixes[cam_id]
    postfix = ('_%d' % track_id)
     
    folder = '../avacar_data'

    fn_original = folder + '/video/' + prefix + postfix + '.mpg'
    fn_in = folder + '/processed_video/' + prefix + postfix + '.mp4'

    fn_out = folder + '/processed_video/' + prefix + postfix + '_demo' + '.mp4'


    writer_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer_size = (2560, 1440)
    writer = cv2.VideoWriter(fn_out, writer_fourcc, 30, writer_size)


    cap = cv2.VideoCapture(fn_in)
    assert cap.isOpened()


    for i in range(0, fid_start):

        cap.read()
    
    for i in range(nr_frames):

        ret, frame = cap.read()
        assert ret

        frame = cv2.resize(frame, writer_size)

        writer.write(frame)

        print(i)

    writer.release()


    pass



def process_drone_demo_videos():


    g_prefix = 'osburn1004'
    g_postfix = '_1'

    a_prefix = '36'
    a_postfix = '_0'


    folder = '../avacar_data'

    fn_in_g = folder + '/processed_video/' + g_prefix + g_postfix + '.mp4'

    fn_in_a = folder + '/processed_aerial_video/' + a_prefix + a_postfix + '.mp4'


    fn_out = folder + '/processed_video/' + g_prefix + g_postfix + '_aerial_demo' + '.mp4'

    g_fid_start = 367
    g_fid_end = 9073

    a_fid_start = 0
    a_fid_end = 8706

    nr_frames = 8700


    writer_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer_size = (2560, 1440)
    writer = cv2.VideoWriter(fn_out, writer_fourcc, 30, writer_size)


    cap_g = cv2.VideoCapture(fn_in_g)
    assert cap_g.isOpened()

    cap_a = cv2.VideoCapture(fn_in_a)
    assert cap_a.isOpened()


    for i in range(0, g_fid_start):

        cap_g.read()

    for i in range(0, a_fid_start):

        cap_a.read()


    for i in range(nr_frames):

        ret, frame_g = cap_g.read()
        assert ret

        ret, frame_a = cap_a.read()
        assert ret

        frame_a[0:1440, 0:1280] = frame_g[0:1440, 1280:]

        cv2.putText(frame_a, 'ground camera', (40, 760),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 4)

        cv2.putText(frame_a, 'drone camera', (1320, 760),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 4)

        writer.write(frame_a)

        print(i)

    writer.release()



    pass



if __name__ == '__main__':

    #process_demo_videos(4, 1)

    process_drone_demo_videos()

    pass

