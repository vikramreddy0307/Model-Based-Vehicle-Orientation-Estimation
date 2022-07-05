'''
Created on May 30, 2020

@author: duolu
'''

import numpy as np
import cv2

import colorsys
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

#import open3d as o3d

class Visualizer(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''

        # Initialize color palette.

        nr_colors = 19
        
        colors = np.zeros((nr_colors, 3), np.uint8)
        colors_list = []
        
        for i in range(nr_colors):
            
            color_tuple = colorsys.hsv_to_rgb((4 * i) % nr_colors / nr_colors, 1, 1)
            color_array = np.asarray(color_tuple)
            color_array *= 255
            colors[i] = color_array.astype(np.uint8)
            
            color_tuple_reverse = (color_tuple[2], color_tuple[1], color_tuple[0])
            colors_list.append(color_tuple_reverse)


        self.nr_colors = nr_colors
        self.colors = colors
        self.colors_list = colors_list




    def draw_target(self, img, x, y, r, color, thickness, c, dx, dy):
    
        cv2.circle(img, (x, y), r, color, thickness)
        d = int(r / 1.414)
        cv2.line(img, (x - d, y - d), (x + d, y + d), color, thickness)
        cv2.line(img, (x + d, y - d), (x - d, y + d), color, thickness)
    
        cv2.putText(img, c, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, r / 10, color, thickness)

    def draw_anchor_pp(self, img_map, pp_anchor_lmap):
        
        chars = ['A', 'B', 'C', 'D']

        dxs = [30, 30, -60, 20]
        dys = [-20, -20, 60, 60]
        
        dx = dxs[direction]
        dy = dys[direction]

        
        for i, pp in enumerate(pp_anchor_lmap.T):
            
            e = int(pp[0])
            s = int(pp[1])
            
        
            self.draw_target(img_map, e, s, 20, (0, 0, 255), 3, chars[i], dx, dy)
        
        pass

    def draw_pp(self, img_map, img_bg, pp_calib_lmap, pp_calib_uv, direction):
    
        
        
        dxs = [30, 30, -60, 20, 20]
        dys = [-20, -20, 60, 60, 20]
        
        dx = dxs[direction]
        dy = dys[direction]
        

        for i, (pp_lmap, pp_uv) in enumerate(zip(pp_calib_lmap.T, pp_calib_uv.T)):
            
            e = int(pp_lmap[0])
            s = int(pp_lmap[1])
            
            u = int(pp_uv[0])
            v = int(pp_uv[1])
        
            self.draw_target(img_map, e, s, 20, (255, 0, 0), 3, '%d' % (i), dx, dy)
        
            self.draw_target(img_bg, u, v, 10, (255, 0, 0), 2, '%d' % (i), dx // 2, dy // 2)
    






class FrameVis(Visualizer):

    def __init__(self):
    
        Visualizer.__init__(self)
        pass

    def draw_ground_grid_on_image(self, img, M, N, grid_uv):
        
        # lines along u
        for i in range(N):
             
            p1 = (int(grid_uv[0, i * M]), int(grid_uv[1, i * M]))
            p2 = (int(grid_uv[0, i * M + M - 1]), int(grid_uv[1, i * M + M - 1]))
             
            cv2.line(img, p1, p2, (255, 0, 0), 2)
     
        # lines along v
        for i in range(M):
             
            p1 = (int(grid_uv[0, i]), int(grid_uv[1, i]))
            p2 = (int(grid_uv[0, M * N - M + i]), int(grid_uv[1, M * N - M + i]))
             
            cv2.line(img, p1, p2, (255, 0, 0), 2)
    
    def draw_ground_xyz_axes_on_image(self, img, P, M, N, grid_xyz):
    
        # plot axes on the image
        
        for i in range(M * N):
            
            o = np.ones((4, 1))
            o[0:3, 0] = grid_xyz[:, i]
            qx = o + np.asarray([[1], [0], [0], [0]]) * 2
            qy = o + np.asarray([[0], [1], [0], [0]]) * 1
            qz = o + np.asarray([[0], [0], [1], [0]]) * 1
            
            # Project to the image
            po = np.matmul(P, o)
            px = np.matmul(P, qx)
            py = np.matmul(P, qy)
            pz = np.matmul(P, qz)
    
            po = po / po[2, 0]
            px = px / px[2, 0]
            py = py / py[2, 0]
            pz = pz / pz[2, 0]
    
            
            p_o = (int(po[0, 0]), int(po[1, 0]))
            p_x = (int(px[0, 0]), int(px[1, 0]))
            p_y = (int(py[0, 0]), int(py[1, 0]))
            p_z = (int(pz[0, 0]), int(pz[1, 0]))
            
            cv2.line(img, p_o, p_x, (0, 0, 255), 2)
            cv2.line(img, p_o, p_y, (0, 255, 0), 2)
            cv2.line(img, p_o, p_z, (255, 0, 0), 2)


    def draw_sof_whole_frame(self, frame, corners_this, corners_prev, c1, c2):
    
    
        k = 2
        
        for corner, corners_prev in zip(corners_this, corners_prev):
            
            x, y = corner.ravel()
            vx, vy = corners_prev.ravel()
    
            x = int(x)
            y = int(y)
            vx = int(vx)
            vy = int(vy)
            
            cv2.circle(frame, (x, y), 1, c1, 2)
            cv2.line(frame, (x, y), (vx, vy), c2, 2)

    def draw_sof_instance(self, frame, instance, c1, c2):

        d = 2

        corners_inlier = instance.corners_to_prev_inliers
        sofs_inlier = instance.sofs_to_prev_inliers
        
        if corners_inlier is None:
            return

        for corner, sof in zip(corners_inlier, sofs_inlier):
            
            x, y = corner.ravel()
            vx, vy = sof.ravel()
    
            x = int(x)
            y = int(y)
            xp = int(x + d * vx)
            yp = int(y + d * vy)
            
            cv2.circle(frame, (x, y), 1, c1, 2)
            cv2.line(frame, (x, y), (xp, yp), c2, 2)

    def draw_instance_segmentation(self, frame, instances):

        for k, instance in enumerate(instances):

            c = k % colors.shape[0]
            color = self.colors[c]
            color_tuple = (int(color[0]), int(color[1]), int(color[2]))
            
            frame[instance.mask] = color
            
            p1x = int(box[0])
            p1y = int(box[1])
            p2x = int(box[2])
            p2y = int(box[3])




    def get_vehicle_color(self, vehicle):

        if vehicle is not None:
            vid = vehicle.vid
        else:
            vid = -1
            
        if vid < 0:
            color = (127, 127, 127)
            color_array = np.asarray(color)
        else:
            color_array = self.colors[vid % self.nr_colors]
            color = (int(color_array[0]), int(color_array[1]), int(color_array[2]))

        return color, color_array

    def draw_vehicle_detection(self, frame, instances):
        
        colors = self.colors
        nr_colors = self.nr_colors
        
        c1 = (0, 0, 255) # red
        c2 = (255, 0, 0) # blue
        
        for k, instance in enumerate(instances):

            color, color_array = self.get_vehicle_color(instance.vehicle)

            box = instance.box

            p1x = int(box[0])
            p1y = int(box[1])
            p2x = int(box[2])
            p2y = int(box[3])


            cv2.rectangle(frame, (p1x, p1y), (p2x, p2y), color, 2)
            #cv2.line(frame, (p1x, p1y), (p2x, p2y), color, 2)


        for k, instance in enumerate(instances):

            self.draw_sof_instance(frame, instance, c1, c2)
            

        pass

    


    def draw_vehicle_segmentation(self, frame, instances):
        
        d = 10
        colors = self.colors
        nr_colors = self.nr_colors
        
        
        for k, instance in enumerate(instances):

            color, color_array = self.get_vehicle_color(instance.vehicle)

            center = instance.center
            mask = instance.mask
            motion = instance.motion

            frame[mask] = color_array

            cx = int(center[0])
            cy = int(center[1])

            vx = int(cx + motion[0] * d)
            vy = int(cy + motion[1] * d)

            #vx = int(cx + momentum[0] * d)
            #vy = int(cy + momentum[1] * d)

            cv2.circle(frame, (cx, cy), 3, (255, 255, 255), 2)
            cv2.line(frame, (cx, cy), (vx, vy), (0, 0, 0), 2)
            
        for k, instance in enumerate(instances):

            vehicle = instance.vehicle

            if vehicle is not None:
                vid = vehicle.vid
            else:
                vid = -1

            tracking_id = instance.tracking_id

            center = instance.center

            cx = int(center[0])
            cy = int(center[1])

            cv2.putText(frame, '%d' % (vid), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        



    def draw_tangent_lines(self, frame, p_x0, p_x1, p_y0, p_y1, p_z0, p_z1, vpx, vpy, vpz):
        '''
        Draw tangent lines using the tangent point on the contour and the
        vanishing point.
        '''
        
        
        vpx_t = tuple(vpx.astype(np.int32))
        vpy_t = tuple(vpy.astype(np.int32))
        vpz_t = tuple(vpz.astype(np.int32))
    
        cv2.line(frame, vpx_t, tuple(p_x0.astype(np.int32)), (0, 0, 255), 2)
        cv2.line(frame, vpx_t, tuple(p_x1.astype(np.int32)), (0, 0, 127), 2)
        cv2.line(frame, vpy_t, tuple(p_y0.astype(np.int32)), (0, 255, 0), 2)
        cv2.line(frame, vpy_t, tuple(p_y1.astype(np.int32)), (0, 127, 0), 2)
        cv2.line(frame, vpz_t, tuple(p_z0.astype(np.int32)), (255, 0, 0), 2)
        cv2.line(frame, vpz_t, tuple(p_z1.astype(np.int32)), (127, 0, 0), 2)
        
    
    
    def draw_bb3d_edge(self, frame, p1, p2, v1, v2, color):
        '''
        Draw the edge of the 3D bounding box.
        '''
        
        if v1 > 0 and v2 > 0:
            
            tp1 = tuple(p1.astype(np.int))
            tp2 = tuple(p2.astype(np.int))
            
            cv2.line(frame, tp1, tp2, color, 4)


        
    
    def draw_bb3d(self, frame, pbox, visibility):
        '''
        Draw the 3D bounding box.
        '''

        pa = pbox[0]
        pb = pbox[1]
        pc = pbox[2]
        pd = pbox[3]
        pe = pbox[4]
        pf = pbox[5]
        pg = pbox[6]
        ph = pbox[7]
    
        va = visibility[0]
        vb = visibility[1]
        vc = visibility[2]
        vd = visibility[3]
        ve = visibility[4]
        vf = visibility[5]
        vg = visibility[6]
        vh = visibility[7]
        
        red = (0, 0, 255)
        green = (0, 255, 0)
        blue = (255, 0, 0)
    
        self.draw_bb3d_edge(frame, pa, pe, va, ve, blue)
        self.draw_bb3d_edge(frame, pb, pf, vb, vf, blue)
        self.draw_bb3d_edge(frame, pc, pg, vc, vg, blue)
        self.draw_bb3d_edge(frame, pd, ph, vd, vh, blue)

#         cv2.putText(frame, 'A', tuple(pa.astype(np.int)), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
#         cv2.putText(frame, 'B', tuple(pb.astype(np.int)), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
#         cv2.putText(frame, 'C', tuple(pc.astype(np.int)), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
#         cv2.putText(frame, 'D', tuple(pd.astype(np.int)), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        self.draw_bb3d_edge(frame, pa, pb, va, vb, green)
        self.draw_bb3d_edge(frame, pc, pd, vc, vd, green)
        self.draw_bb3d_edge(frame, pe, pf, ve, vf, green)
        self.draw_bb3d_edge(frame, pg, ph, vg, vh, green)
        
        self.draw_bb3d_edge(frame, pa, pc, va, vc, red)
        self.draw_bb3d_edge(frame, pb, pd, vb, vd, red)
        self.draw_bb3d_edge(frame, pe, pg, ve, vg, red)
        self.draw_bb3d_edge(frame, pf, ph, vf, vh, red)
    

    def draw_bb3ds_on_frame(self, frame, instances):
        
        
        for instance in instances:

            contour = instance.contour
            bb3d = instance.bb3d

            if bb3d is None:
                continue

            pbox = bb3d.pbox
            visibility = bb3d.visibility

            #color, color_array = self.get_instance_color(instance)

            #cv2.drawContours(frame, [contour], -1, color, 3)

            cv2.drawContours(frame, [contour], -1, (255, 255, 255), 3)

            self.draw_bb3d(frame, pbox, visibility)
        
        
        
        pass







class MapVis(Visualizer):

    def __init__(self):
    
        Visualizer.__init__(self)
        pass


    def draw_camera_FOV_on_map(self, map, camera_model, map_model, roi,
                               color=(255, 0, 0), thickness=4):
        
        u1, v1, u2, v2 = roi
        
        p1 = np.asarray((u1, v1))
        p2 = np.asarray((u2, v1))
        p3 = np.asarray((u2, v2))
        p4 = np.asarray((u1, v2))
        
        q1 = camera_model.transform_point_image_to_ground(p1)
        q2 = camera_model.transform_point_image_to_ground(p2)
        q3 = camera_model.transform_point_image_to_ground(p3)
        q4 = camera_model.transform_point_image_to_ground(p4)
        
        qs = np.zeros((3, 4))
        qs[:, 0] = q1
        qs[:, 1] = q2
        qs[:, 2] = q3
        qs[:, 3] = q4
        
        rs = map_model.transform_points_xyz_to_lmap(qs)
        
        r1 = rs[0:2, 0]
        r2 = rs[0:2, 1]
        r3 = rs[0:2, 2]
        r4 = rs[0:2, 3]
        
        cv2.line(map, tuple(r1.astype(np.int)), tuple(r2.astype(np.int)), color, thickness)
        cv2.line(map, tuple(r2.astype(np.int)), tuple(r3.astype(np.int)), color, thickness)
        cv2.line(map, tuple(r3.astype(np.int)), tuple(r4.astype(np.int)), color, thickness)
        cv2.line(map, tuple(r4.astype(np.int)), tuple(r1.astype(np.int)), color, thickness)
        
        pass
    
    
    def draw_camera_FOV_on_global_map(self, map, camera_model, map_model, roi,
                                      color=(0, 0, 0), thickness=2):
        
        u1, v1, u2, v2 = roi
        
        p1 = np.asarray((u1, v1))
        p2 = np.asarray((u2, v1))
        p3 = np.asarray((u2, v2))
        p4 = np.asarray((u1, v2))
        
        q1 = camera_model.transform_point_image_to_ground(p1)
        q2 = camera_model.transform_point_image_to_ground(p2)
        q3 = camera_model.transform_point_image_to_ground(p3)
        q4 = camera_model.transform_point_image_to_ground(p4)
        
        qs = np.zeros((3, 4))
        qs[:, 0] = q1
        qs[:, 1] = q2
        qs[:, 2] = q3
        qs[:, 3] = q4
        
        rs = map_model.transform_points_xyz_to_lmap(qs)
        rs = map_model.transform_points_lmap_to_gmap(rs)
        
        r1 = rs[0:2, 0]
        r2 = rs[0:2, 1]
        r3 = rs[0:2, 2]
        r4 = rs[0:2, 3]

        
        cv2.line(map, tuple(r1.astype(np.int)), tuple(r2.astype(np.int)), color, thickness)
        cv2.line(map, tuple(r2.astype(np.int)), tuple(r3.astype(np.int)), color, thickness)
        cv2.line(map, tuple(r3.astype(np.int)), tuple(r4.astype(np.int)), color, thickness)
        cv2.line(map, tuple(r4.astype(np.int)), tuple(r1.astype(np.int)), color, thickness)
        
        pass


    def draw_bb3ds_on_map(self, map, map_model, instances, instances_3d_states, frame_id):
        
        for instance, instance_3d_states in zip(instances, instances_3d_states):

            vehicle = instance.vehicle
            bb3d = instance.bb3d
            
            vel = instance_3d_states.vel

            if bb3d is None:
                continue

            if vehicle is not None:
                vid = vehicle.vid
            else:
                vid = -1
                
            if vid < 0:
                color = (127, 127, 127)
                color_array = np.asarray(color)
            else:
                color_array = self.colors[vid % self.nr_colors]
                color = (int(color_array[0]), int(color_array[1]), int(color_array[2]))

            
            qbox = bb3d.qbox
            
            qam = qbox[0].reshape((3, 1))
            qbm = qbox[1].reshape((3, 1))
            qcm = qbox[2].reshape((3, 1))
            qdm = qbox[3].reshape((3, 1))
            
            
            
            qa_lmap = map_model.transform_points_xyz_to_lmap(qam)
            qb_lmap = map_model.transform_points_xyz_to_lmap(qbm)
            qc_lmap = map_model.transform_points_xyz_to_lmap(qcm)
            qd_lmap = map_model.transform_points_xyz_to_lmap(qdm)
            
            #print(k, qa, qa_lmap)
            
            tqa_lmap = (int(qa_lmap[0]), int(qa_lmap[1]))
            tqb_lmap = (int(qb_lmap[0]), int(qb_lmap[1]))
            tqc_lmap = (int(qc_lmap[0]), int(qc_lmap[1]))
            tqd_lmap = (int(qd_lmap[0]), int(qd_lmap[1]))
            
            thickness = 4
            
            cv2.line(map, tqa_lmap, tqb_lmap, color, thickness)
            cv2.line(map, tqb_lmap, tqd_lmap, color, thickness)
            cv2.line(map, tqd_lmap, tqc_lmap, color, thickness)
            cv2.line(map, tqc_lmap, tqa_lmap, color, thickness)
        
            offset = (qb_lmap - qa_lmap) * 1.2
            tx = int(qb_lmap[0] + offset[0])
            ty = int(qb_lmap[1] + offset[1])
            
            # Convert the speed to km/h.
            # speed = np.linalg.norm(v3d) * 3.6
            # Convert the speed to mile/h.
            speed = np.linalg.norm(vel) * 3.6 / 1.609344
        
            cv2.putText(map, '%.1f' % (speed), (tx, ty), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, thickness)

#             if vid == 3:
#                 
#                 speed = np.linalg.norm(vel)
#                 print(frame_id, np.linalg.norm(vel))



    def draw_observed_states_on_map(self, map, map_model, 
        instances, instances_3d_states, frame_inverval):
        
        mm = map_model
        
        for instance, states in zip(instances, instances_3d_states):
            
            vehicle = instance.vehicle
            bb3d = instance.bb3d

            if bb3d is None:
                continue

            
            pos = states.pos.copy()
            pos_uc_point = states.pos.copy()
            pos_uc_point[0] += states.pos_uc
            
            vel = states.vel
            vel_uc = states.vel_uc

            pos_vel_point = pos + vel * frame_inverval * 3

            if vehicle is not None:
                vid = vehicle.vid
            else:
                vid = -1
                
            if vid < 0:
                color = (127, 127, 127)
                color_array = np.asarray(color)
            else:
                color_array = self.colors[vid % self.nr_colors]
                color = (int(color_array[0]), int(color_array[1]), int(color_array[2]))
            
        
            pos_lmap = mm.transform_points_xyz_to_lmap(pos.reshape((3, 1)))
            pos_uc_point_lmap = mm.transform_points_xyz_to_lmap(pos_uc_point.reshape((3, 1)))
            pos_uc_lmap = np.linalg.norm(pos_uc_point_lmap - pos_lmap)
        
            pos_vel_point_lmap = mm.transform_points_xyz_to_lmap(pos_vel_point.reshape((3, 1)))
            
            cv2.circle(map, (int(pos_lmap[0]), int(pos_lmap[1])), int(pos_uc_lmap), color, 3)
            cv2.line(map, (int(pos_lmap[0]), int(pos_lmap[1])), 
                     (int(pos_vel_point_lmap[0]), int(pos_vel_point_lmap[1])), color, 3)
    
    
    def draw_vehicle_states_on_map(self, map_local, map_model, vehicles, frame_inverval, pair=None, lane=None, all_df=None, thickness=8):
        
        # This is a scalar to make the velocity vector more easy to observe
        # when plotted on the map.
        vel_d = 10
        
        mm = map_model
        
        for vid, vehicle in vehicles.items():
            # df_id = all_df[all_df['temporaryId'] == vid]
            
            # if vid == pair[0] or vid == pair[1]:
            color_array = self.colors[vid % self.nr_colors]
            color = (int(color_array[0]), int(color_array[1]), int(color_array[2]))

            heading = vehicle.heading
            pos, vel, pos_flag = vehicle.get_pos_vel()

            pos_uc, vel_uc = vehicle.get_pos_vel_uc()

            pos_uc_point = pos.copy()
            pos_uc_point[0] += pos_uc
            pos_vel_point = pos + vel * frame_inverval * vel_d


            dim = vehicle.dim
            l = dim[0]
            w = dim[1]

            nv = heading / np.linalg.norm(heading)
            nt = np.asarray((-nv[1], nv[0], 0))

            if pos_flag == 0:

                qa = pos + 0.5 * w * nt
                qb = pos - 0.5 * w * nt
                qc = qa - l * nv
                qd = qb - l * nv

            elif pos_flag == 1:

                qa = pos + 0.5 * l * nv
                qc = pos - 0.5 * l * nv
                qb = qa - w * nt
                qd = qc - w * nt

            elif pos_flag == 2:

                qb = pos + 0.5 * l * nv
                qd = pos - 0.5 * l * nv
                qa = qb + w * nt
                qc = qd + w * nt

            elif pos_flag == 3:

                qc = pos + 0.5 * w * nt
                qd = pos - 0.5 * w * nt
                qa = qc + l * nv
                qb = qd + l * nv

            elif pos_flag == 8:

                qa = pos + 0.5 * w * nt + 0.5 * l * nv
                qb = pos - 0.5 * w * nt + 0.5 * l * nv
                qc = pos + 0.5 * w * nt - 0.5 * l * nv
                qd = pos - 0.5 * w * nt - 0.5 * l * nv


            else:

                # The pos_flag is buggy!
                continue


            # Transform points to the map.

            pos_lmap = mm.transform_points_xyz_to_lmap(pos.reshape((3, 1)))
            pos_uc_point_lmap = mm.transform_points_xyz_to_lmap(pos_uc_point.reshape((3, 1)))
            pos_uc_lmap = np.linalg.norm(pos_uc_point_lmap - pos_lmap)

            pos_vel_point_lmap = mm.transform_points_xyz_to_lmap(pos_vel_point.reshape((3, 1)))


            # Draw the bounding box of the vehicle projected on the map.

            qa_lmap = mm.transform_points_xyz_to_lmap(qa.reshape((3, 1)))
            qb_lmap = mm.transform_points_xyz_to_lmap(qb.reshape((3, 1)))
            qc_lmap = mm.transform_points_xyz_to_lmap(qc.reshape((3, 1)))
            qd_lmap = mm.transform_points_xyz_to_lmap(qd.reshape((3, 1)))

            center_lmap = (qa_lmap + qb_lmap + qc_lmap + qd_lmap) / 4

            ra = (int(qa_lmap[0]), int(qa_lmap[1]))
            rb = (int(qb_lmap[0]), int(qb_lmap[1]))
            rc = (int(qc_lmap[0]), int(qc_lmap[1]))
            rd = (int(qd_lmap[0]), int(qd_lmap[1]))

            cv2.line(map_local, ra, rb, color, thickness)
            cv2.line(map_local, rb, rd, color, thickness)
            cv2.line(map_local, rd, rc, color, thickness)
            cv2.line(map_local, rc, ra, color, thickness)

            # Draw the uncertainty circle and the velocity vector on the map.

            rp = (int(center_lmap[0]), int(center_lmap[1]))
            #rp = (int(pos_lmap[0]), int(pos_lmap[1]))
            rp_uc = int(pos_uc_lmap)

            rpv = (int(pos_vel_point_lmap[0]), int(pos_vel_point_lmap[1]))

            cv2.circle(map_local, rp, rp_uc, color, 4)
            #cv2.line(map_local, rp, rpv, color, 4)


            offset = (qc_lmap - qa_lmap) * 1.0
            tx = int(qc_lmap[0] + offset[0]/2)
            ty = int(qc_lmap[1] + offset[1]/2)

            # Convert the speed to km/h.
            # speed = np.linalg.norm(v3d) * 3.6
            # Convert the speed to mile/h.
            speed = np.linalg.norm(vel) #* 3.6 #/ 1.609344
            try:
                line_id = df_id.lane.values[0]
                cv2.putText(map_local, f'({vid}, {round(speed, 1)}, {line_id})', (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 4)
            except:
                cv2.putText(map_local, f'({vid}, {round(speed, 1)})', (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 4)

        pass
    
    def draw_vehicle_states_on_global_map(self, map, map_model, vehicles, frame_inverval):
        
        # This is a scalar to make the velocity vector more easy to observe
        # when plotted on the map.
        vel_d = 3
        thickness = 2
        
        mm = map_model
        
        for vid, vehicle in vehicles.items():

            color_array = self.colors[vid % self.nr_colors]
            color = (int(color_array[0]), int(color_array[1]), int(color_array[2]))
            
            heading = vehicle.heading
            pos, vel, pos_flag = vehicle.get_pos_vel()
            
            pos_uc, vel_uc = vehicle.get_pos_vel_uc()

            pos_uc_point = pos.copy()
            pos_uc_point[0] += pos_uc
            pos_vel_point = pos + vel * frame_inverval * vel_d

            
        
            dim = vehicle.dim
            l = dim[0]
            w = dim[1]
            
            nv = heading / np.linalg.norm(heading)
            nt = np.asarray((-nv[1], nv[0], 0))
            
            if pos_flag == 0:
                
                qa = pos + 0.5 * w * nt
                qb = pos - 0.5 * w * nt
                qc = qa - l * nv
                qd = qb - l * nv
            
            elif pos_flag == 1:
                
                qa = pos + 0.5 * l * nv
                qc = pos - 0.5 * l * nv
                qb = qa - w * nt
                qd = qc - w * nt
            
            elif pos_flag == 2:
                
                qb = pos + 0.5 * l * nv
                qd = pos - 0.5 * l * nv
                qa = qb + w * nt
                qc = qd + w * nt
            
            elif pos_flag == 3:
                
                qc = pos + 0.5 * w * nt
                qd = pos - 0.5 * w * nt
                qa = qc + l * nv
                qb = qd + l * nv

            else:
                
                # The pos_flag is buggy!
                continue
            
            
            
            # Draw the bounding box of the vehicle projected on the map.
            
            qs = np.zeros((3, 7))
            qs[:, 0] = qa
            qs[:, 1] = qb
            qs[:, 2] = qc
            qs[:, 3] = qd
            
            qs[:, 4] = pos
            qs[:, 5] = pos_uc_point
            qs[:, 6] = pos_vel_point
            
            rs = mm.transform_points_xyz_to_lmap(qs)
            rs = mm.transform_points_lmap_to_gmap(rs)
            
            r1 = rs[0:2, 0]
            r2 = rs[0:2, 1]
            r3 = rs[0:2, 2]
            r4 = rs[0:2, 3]
    
            pos_gmap = rs[0:2, 4]
            pos_uc_point_gmap = rs[0:2, 5]
            pos_vel_point_gmap = rs[0:2, 6]
            
            pos_uc_gmap = np.linalg.norm(pos_uc_point_gmap - pos_gmap)
            
            cv2.line(map, tuple(r1.astype(np.int)), tuple(r2.astype(np.int)), color, thickness)
            cv2.line(map, tuple(r1.astype(np.int)), tuple(r3.astype(np.int)), color, thickness)
            cv2.line(map, tuple(r2.astype(np.int)), tuple(r4.astype(np.int)), color, thickness)
            cv2.line(map, tuple(r3.astype(np.int)), tuple(r4.astype(np.int)), color, thickness)
            
            # Draw the uncertainty circle and the velocity vector on the map.
            
            #cv2.circle(map, (int(pos_gmap[0]), int(pos_gmap[1])), int(pos_uc_gmap), color, thickness)
            #cv2.line(map, (int(pos_gmap[0]), int(pos_gmap[1])), 
            #         (int(pos_vel_point_gmap[0]), int(pos_vel_point_gmap[1])), color, thickness)

            
            offset = (r2 - r1) * 1.2
            tx = int(r2[0] + offset[0])
            ty = int(r2[1] + offset[1])
            
            # Convert the speed to km/h.
            # speed = np.linalg.norm(v3d) * 3.6
            # Convert the speed to mile/h.
            speed = np.linalg.norm(vel) * 3.6 / 1.609344
        
            #cv2.putText(map, '%.1f' % (speed), (tx, ty), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
            
            cv2.putText(map, '%d' % (vid), (tx, ty), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
            
            
            
            
    def draw_fused_states_on_global_map(self, map, map_model, vehicles, frame_inverval):
        
        # This is a scalar to make the velocity vector more easy to observe
        # when plotted on the map.
        vel_d = 3
        thickness = 2
        
        mm = map_model
        
        for vid, vehicle in vehicles.items():

            color_array = self.colors[vid % self.nr_colors]
            color = (int(color_array[0]), int(color_array[1]), int(color_array[2]))
            
            center = vehicle.center_fuse
            dx = vehicle.dx_fuse
            dy = vehicle.dy_fuse

            #print(dx, dy)

            r1 = center + dx + dy
            r2 = center + dx - dy
            r3 = center - dx + dy
            r4 = center - dx - dy

            r1 = r1[0:2].flatten()
            r2 = r2[0:2].flatten()
            r3 = r3[0:2].flatten()
            r4 = r4[0:2].flatten()
            
            
            
            cv2.line(map, tuple(r1.astype(np.int)), tuple(r2.astype(np.int)), color, thickness)
            cv2.line(map, tuple(r1.astype(np.int)), tuple(r3.astype(np.int)), color, thickness)
            cv2.line(map, tuple(r2.astype(np.int)), tuple(r4.astype(np.int)), color, thickness)
            cv2.line(map, tuple(r3.astype(np.int)), tuple(r4.astype(np.int)), color, thickness)
        
        cv2.putText(map, 'ASU Active Perception Group', (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (127, 127, 127), thickness)
            

class SpeedPlotVis(Visualizer):



    def __init__(self, init_frame_id):

        Visualizer.__init__(self)
    
        plt.interactive(True)
    
        fig = plt.figure()
    
        ax_speed = fig.add_axes([0.1, 0.1, 0.85, 0.85])
        
        
        self.fig = fig
        self.ax_speed = ax_speed
        
        self.i = 0
        self.d = 600
        
        self.vstates = {}
        
        plt.show()
        
        
        
        pass


    def update_plot(self, frame_id, current_vehicles, removed_vehicles):
        
        
        
#         for vehicle in removed_vehicles:
#             
#             vid = vehicle.vid
#             
#             if vid in self.vstates:
#                 
#                 states, start, end = self.vstates[vid]
#                 
#                 self.vstates[vid] = (states, start, frame_id)
                
        
        for vid, vehicle in current_vehicles.items():
            
            if vid not in self.vstates:
                
                self.vstates[vid] = ([], frame_id, frame_id)
        
            states, start, end = self.vstates[vid]
            
            pos, vel, pos_flag = vehicle.get_pos_vel()
            
            speed = np.linalg.norm(vel) * 3.6 / 1.609344
            states.append(speed)
            
            self.vstates[vid] = (states, start, frame_id + 1)
        
        
        i = self.i
        d = self.d
        
        self.ax_speed.clear()
        
        for vid, (states, start, end) in self.vstates.items():

            if end >= i or start < i + d:
                
                l = end - start
                xs = np.arange(start, end)
                ys = states[:l]
                
                n = len(xs)
                m = len(ys)
                
                if n != m:
                    
                    print(n, m, start, end, current_vehicles[vid].prediction_iteration)
                
                color = self.colors_list[vid % self.nr_colors]
                
                self.ax_speed.plot(xs, ys, color=color)
        
        self.ax_speed.set_xlim(i, i + d)
        self.ax_speed.set_ylim(0, 80)
        
        self.ax_speed.xaxis.set_major_locator(MultipleLocator(30))
        
        self.ax_speed.set_xlabel('frame')
        self.ax_speed.set_ylabel('speed (mph)')
        self.ax_speed.grid(True)
        
        if frame_id > i + d:
            self.i = i + 1



if __name__ == '__main__':


    pass