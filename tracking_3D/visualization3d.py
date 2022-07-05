import numpy as np
import cv2

class MapVis3D(object):

    def __init__(self):
    
        pass


    def plot_img(self, ax, image):
        
        height = image.shape[0]
        width = image.shape[1]
        
        xa = np.arange(0, height, 1)
        ya = np.arange(0, width, 1)
        xs, ys = np.meshgrid(xa, ya)
        
        zs = np.zeros(xs.shape)
        
        
        facecolors = np.ones((height, width, 4))
        facecolors[:, :, 0:3] = image / 256
        
        print(xs.shape, ys.shape, zs.shape, facecolors.shape)
        
        ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, facecolors=facecolors)
        
        pass



    def draw_camera_3d(self, R, t, points_xyz, grid_x, grid_y):
        
        cam = Camera(-1, width, height)
        cam.set_pose(R.T, -np.matmul(R.T, t))
         
        vis = CameraVisualizer3D()
         
        vis.plot_camera(cam, f=1, scale=2)
        vis.plot_points(points_xyz)
         
        wireframe_z = np.zeros(grid_x.shape)
        fvis.ax.plot_wireframe(grid_x, grid_y, wireframe_z)
         
        vis.set_axes_label()
        vis.set_axes_limit(xlim=40, ylim=40, zlim=40)
        vis.show()
        
        
        
        pass





from numpy import pi, sin, cos, mgrid
from mayavi import mlab
import pylab as pl

def test_map3d():
    
#     direction = 0
#     postfix = ('_%d' % 1)
#     
#     prefixes = ['westbound', 'eastbound', 'northbound', 'southbound']
#     prefix = prefixes[direction]
#     
#     folder = './car_data'
#     
#     map_fn = folder + '/calibration/' + prefix + '_map.png'
#     map = cv2.imread(map_fn)
#     
#     
#     mlab.imshow(map[:, :, 0])
#     
#     
#     mlab.show()
    
    x, y = np.mgrid[0:3:1,0:3:1]
    s = mlab.surf(x, y, np.asarray(x*0.1, 'd'))
    
    @mlab.animate
    def anim():
        for i in range(10):
            s.mlab_source.scalars = np.asarray(x*0.1*(i+1), 'd')
            yield
    
    anim()
    mlab.show()    
    
    pass


def euler_zyx_to_rotation_matrix(z, y, x):
    '''
    Converting a rotation represented by three Euler angles (z-y'-x") to
    rotation matrix represenation, i.e., using the following,
    
        R = R_z * R_y * R_x
    
    where,
    
        R_z             = [ cos(z)     -sin(z)        0       ]
                          [ sin(z)     cos(z)         0       ]
                          [ 0            0            1       ]
    
        R_y             = [ cos(y)       0            sin(y)  ]
                          [ 0            1            0       ]
                          [ -sin(y)      0            cos(y)  ]
    
        R_x             = [ 1            0            0       ]
                          [ 0            cos(x)       -sin(x) ]
                          [ 0            sin(x)       cos(x)  ]
    
    Also, the angles are named as following,
    
        z - yaw (psi)
        y - pitch (theta)
        x - roll (phi)
    
    These angles are also called Tait-Bryan angles, and we use the z-y'-x"
    intrinsic convention. See this for the conventions:
    
        https://en.wikipedia.org/wiki/Euler_angles#Tait–Bryan_angles
    
    Also see this for the conversion between different representations:
    
        https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimension
    
    Caution: The three input angles are in radian!
    
    '''
    

    sz = np.sin(z)
    cz = np.cos(z)
    sy = np.sin(y)
    cy = np.cos(y)
    sx = np.sin(x)
    cx = np.cos(x)

    a11 = cz * cy
    a12 = cz * sy * sx - cx * sz
    a13 = sz * sx + cz * cx * sy
    a21 = cy * sz
    a22 = cz * cx + sz * sy * sx
    a23 = cx * sz * sy - cz * sx
    a31 = -sy
    a32 = cy * sx
    a33 = cy * cx

    R = np.asarray([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])

    return R

def calculate_homography_from_cam_pose(R, t, f, u0, v0):
    
    Rp = R.T
    tp = -np.matmul(R.T, t.reshape((3, 1)))
    
    K = np.identity(3)
    K[0, 0] = f
    K[1, 1] = f
    K[0, 2] = u0
    K[1, 2] = v0
    
    T = np.zeros((3, 3))
    T[0, 0] = Rp[0, 0]
    T[0, 1] = Rp[0, 1]
    T[0, 2] = tp[0]
    T[1, 0] = Rp[1, 0]
    T[1, 1] = Rp[1, 1]
    T[1, 2] = tp[1]
    T[2, 0] = Rp[2, 0]
    T[2, 1] = Rp[2, 1]
    T[2, 2] = tp[2]

    H = np.matmul(K, T)
    
    return H

def calculate_projection_from_cam_pos(R, t, f, u0, v0):


    pass

def normalize_rotation_matrix(R):
    '''
    Normalize a rotation matrix with SVD, i.e., using the following step.
    
        u, s, vh = np.linalg.svd(R)
        
        return np.matmul(u, vh)
    
    '''
    
    u, s, vh = np.linalg.svd(R)
    del s
    
    return np.matmul(u, vh)

def test_custom_3d():
    
    
    direction = 0
    postfix = ('_%d' % 1)
    
    prefixes = ['westbound', 'eastbound', 'northbound', 'southbound']
    prefix = prefixes[direction]
    
    folder = './car_data'
    
    map_fn = folder + '/calibration/' + prefix + '_map.png'
    map = cv2.imread(map_fn)
    
    
    f = 400
    u0 = 640
    v0 = 320
    
    
    d_yaw = 0.0
    d_pitch = 0.0
    d_roll = 0.0
    
    R = euler_zyx_to_rotation_matrix(0, 0, 0)
    
    x = 0
    y = 0
    z = -2000

    t = np.zeros(3)
    t[0] = x
    t[1] = y
    t[2] = z

    dx = 0
    dy = 0
    dz = 0
    
    dt = np.zeros(3)
    
    
    while True:


        
        #print(yaw, pitch, roll, x, y, z, f)
        
        dR = euler_zyx_to_rotation_matrix(
            d_yaw / 180 * np.pi, d_pitch / 180 * np.pi, d_roll / 180 * np.pi)
        
        dt[0] = dx
        dt[1] = dy
        dt[2] = dz
        
        
        R = np.matmul(R, dR)
        R = normalize_rotation_matrix(R)
        
        t = np.matmul(R, dt.reshape((3, 1))) + t.reshape((3, 1))
        
        H = calculate_homography_from_cam_pose(R, t, f, u0, v0)
        
        H_inv = np.linalg.inv(H)
        
        white_im = np.zeros((720, 1280, 3), dtype=np.uint8)
        white_im.fill(255)
        map_warp = cv2.warpPerspective(map, H, (1280, 720), white_im)
        
        cv2.imshow('map_warp', map_warp)
        
        dx = 0
        dy = 0
        dz = 0

        d_yaw = 0.0
        d_pitch = 0.0
        d_roll = 0.0
        
        
        c = cv2.waitKey(-1)
           
        if c & 0xFF == ord('x'):
            break
        
        if c & 0xFF == ord('d'):
            d_pitch += 2
        if c & 0xFF == ord('a'):
            d_pitch -= 2
        if c & 0xFF == ord('s'):
            d_roll += 2
        if c & 0xFF == ord('w'):
            d_roll -= 2
        if c & 0xFF == ord('q'):
            d_yaw += 2
        if c & 0xFF == ord('e'):
            d_yaw -= 2

        if c & 0xFF == ord('z'):
            f += 5
        if c & 0xFF == ord('c'):
            f -= 5
            if f < 1:
                f = 1


        if c & 0xFF == ord('t'):
            dy += 5
        if c & 0xFF == ord('u'):
            dy -= 5
        if c & 0xFF == ord('y'):
            dz += 5
        if c & 0xFF == ord('h'):
            dz -= 5
        if c & 0xFF == ord('j'):
            dx += 5
        if c & 0xFF == ord('g'):
            dx -= 5

    
        if c & 0xFF == ord('x'):
            break
    
    
    cv2.destroyAllWindows()
    
    pass



def test_validation():

    fn = './car_data/validation/vis_2.csv'

    data = np.loadtxt(fn, delimiter=',')
    
    n = data.shape[0]
    
    for i in range(n):
        
        if i % 2 == 0:
            continue
        
        print('%d\t%f' % (data[i, 0], data[i, 1]))


    pass
