'''
Created on Aug 10, 2020

@author: duolu

This module contains methods for working with coordinates in WGS84.

Terminologies:

    "llh" or "LLH" means (latitude, longitude, height)
    "latitude" and "longitude" are always in degrees.
    "lat" and "lon" are always in radians (only used internally).
    "height" is the height above the geodetic ellipsoid, in meters.
    "llh_ref" is a tuple of LLH for specifying a reference point.

    "ecef" or "ECEF" means "Earth-Centered-Earth-Fixed".
    "xyz" or "XYZ" means coordinates in the ECEF reference frame.
    "enu" or "ENU" means (east, north, up).
    "ned" or "NED" means (north, east, down).
    "esd" or "ESD" means (east, south, down).
    

NOTE: All Euclidean reference frames are right handed.

NOTE: All Euclidean coordinates are in meters.

NOTE: "height" is not "altitude".


'''


import math
import numpy as np



# These are constants used in WGS84, all related to the geodetic ellipsoid.
#
# "F" is the flattening constant
# "A" is the semi-major axis length, i.e., equatorial radius.
# "B" is the semi-minor axis length, i.e., polar radius.
# "E2" is the first eccentricity squared
# "EP2" is the second eccentricity squared

F = 1/298.257223563
A = 6378137.0
A2 = A * A
B = A * (1 - F)
B2 = B * B
E2 = (A2 - B2) / A2
EP2 = (A2 - B2) / B2

def llh_to_ecef(latitude, longitude, height):
    '''Convert (latitude, longitude, height) in LLH to (x, y, z) in ECEF.
    '''

    lat = latitude / 180 * math.pi
    lon = longitude / 180 * math.pi

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    roc = A2 / math.sqrt(A2 * cos_lat * cos_lat + B2 * sin_lat * sin_lat)

    x = (roc + height) * cos_lat * cos_lon
    y = (roc + height) * cos_lat * sin_lon
    z = (B2 / A2 * roc + height) * sin_lat

    return (x, y, z)

def ecef_to_llh(x, y, z):
    '''Convert (x, y, z) in ECEF to (latitude, longitude, height) in LLH.
    '''

    r2 = x * x + y * y
    r = math.sqrt(r2)

    theta = math.atan2(A * z, B * r)

    lat_y = z + EP2 * B * (math.sin(theta) ** 3)
    lat_x = r - E2 * A * (math.cos(theta) ** 3)

    lon = math.atan2(y, x)
    lat = math.atan2(lat_y, lat_x)

    sin_lat = math.sin(lat)
    roc = A / math.sqrt(1 - E2 * sin_lat * sin_lat)

    height = r / math.cos(lat) - roc

    latitude = lat / math.pi * 180
    longitude = lon / math.pi * 180

    return latitude, longitude, height


def ecef_to_enu_rotation_matrix(lat, lon):
    '''Obtain the rotation matrix from the ECEF frame to ENU frame.

    This ENU frame is on the tagent plane at the point (lat, lon).

    NOTE: "lat" and "lon" are in radian.
    '''
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    R = np.zeros((3, 3))

    R[0, 0] = -sin_lon
    R[0, 1] = cos_lon
    R[0, 2] = 0
    R[1, 0] = -sin_lat * cos_lon
    R[1, 1] = -sin_lat * sin_lon
    R[1, 2] = cos_lat
    R[2, 0] = cos_lat * cos_lon
    R[2, 1] = cos_lat * sin_lon
    R[2, 2] = sin_lat

    return R


def ecef_to_enu(x, y, z, llh_ref):
    '''Convert (x, y, z) in ECEF to (e, n, u) in ENU.

    This ENU frame is on the tagent plane at the point "llh_ref"
    '''

    x_ref, y_ref, z_ref = llh_to_ecef(*llh_ref)

    # CAUTION: angles must be in radian before doing sin() and cos().
    lat = llh_ref[0] / 180 * np.pi
    lon = llh_ref[1] / 180 * np.pi

    R = ecef_to_enu_rotation_matrix(lat, lon)

    p = np.asarray((x, y, z))
    p = p.reshape((3, 1))

    p_ref = np.asarray((x_ref, y_ref, z_ref))
    p_ref = p_ref.reshape((3, 1))

    enu = np.matmul(R, p - p_ref)

    return enu[0], enu[1], enu[2]

def enu_to_ecef(e, n, u, llh_ref):
    '''Convert (e, n, u) in ENU to (x, y, z) in ECEF.

    This ENU frame is on the tagent plane at the point "llh_ref".

    "llh_ref" is a tuple of (latitude, longitude, height).
    '''

    x_ref, y_ref, z_ref = llh_to_ecef(*llh_ref)

    # CAUTION: angles must be in radian before doing sin() and cos().
    lat = llh_ref[0] / 180 * np.pi
    lon = llh_ref[1] / 180 * np.pi

    R = ecef_to_enu_rotation_matrix(lat, lon)

    p = np.asarray((e, n, u))
    p = p.reshape((3, 1))

    p_ref = np.asarray((x_ref, y_ref, z_ref))
    p_ref = p_ref.reshape((3, 1))

    xyz = np.matmul(R.T, p) + p_ref

    return xyz[0], xyz[1], xyz[2]

def llh_to_enu(latitude, longitude, height, llh_ref):
    '''Convert (latitude, longitude, height) in LLH to (e, n, u) in ENU.
    '''

    x, y, z = llh_to_ecef(latitude, longitude, height)

    e, n, u = ecef_to_enu(x, y, z, llh_ref)

    return e, n, u

def enu_to_llh(e, n, u, llh_ref):
    '''Convert (e, n, u) in ENU to (latitude, longitude, height) in LLH.
    '''

    x, y, z = enu_to_ecef(e, n, u, llh_ref)

    latitude, longitude, height = ecef_to_llh(x, y, z)

    return latitude, longitude, height


def llh_to_ned(latitude, longitude, height, llh_ref):
    '''Convert (latitude, longitude, height) in LLH to (n, e, d) in NED.
    '''

    e, n, u = llh_to_enu(latitude, longitude, height, llh_ref)

    return n, e, -u

def ned_to_llh(n, e, d, llh_ref):
    '''Convert (n, e, d) in NED to (latitude, longitude, height) in LLH.
    '''

    latitude, longitude, height = enu_to_llh(e, n, -d, llh_ref)

    return latitude, longitude, height

def llh_to_esd(latitude, longitude, height, llh_ref):
    '''Convert (latitude, longitude, height) in LLH to (e, s, d) in ESD.
    '''

    e, n, u = llh_to_enu(latitude, longitude, height, llh_ref)

    return e, -n, -u

def esd_to_llh(e, s, d, llh_ref):
    '''Convert (e, s, d) in ESD to (latitude, longitude, height) in LLH.
    '''

    latitude, longitude, height = enu_to_llh(e, -s, -d, llh_ref)

    return latitude, longitude, height

def test_wgs84():
    '''Test code for this module.

    NOTE: Currently it can pass all tests at atol = 1e-8.
    '''

    atol = 1e-8

    # for latitude in range(-90, 90, 1):
    #     for longitude in range(-180, 180, 1):
    #         for height in range(-10000, 100000, 10000):

    #             x, y, z = llh_to_ecef(latitude, longitude, height)
    #             latitude_p, longitude_p, height_p = ecef_to_llh(x, y, z)

    #             assert math.isclose(latitude, latitude_p, abs_tol=atol), \
    #                 '%.10f <-> %.10f' % (latitude, latitude_p)
    #             assert math.isclose(longitude, longitude_p, abs_tol=atol), \
    #                 '%.10f <-> %.10f' % (longitude, longitude_p)
    #             assert math.isclose(height, height_p, abs_tol=atol), \
    #                 '%.10f <-> %.10f' % (height, height_p)


    llh_ref = (33.843004, -112.134953, 539)

    for e in range (-200, 200, 10):
        for n in range(-200, 200, 10):
            for u in range(-200, 200, 10):

                llh = enu_to_llh(e, n, u, llh_ref)
                e_p, n_p, u_p = llh_to_enu(*llh, llh_ref)

                assert math.isclose(e, e_p, abs_tol=atol), \
                    '%.10f <-> %.10f' % (e, e_p)
                assert math.isclose(n, n_p, abs_tol=atol), \
                    '%.10f <-> %.10f' % (n, n_p)
                assert math.isclose(u, u_p, abs_tol=atol), \
                    '%.10f <-> %.10f' % (u, u_p)





    print('Pass all tests!')

    #print(llh_to_ecef(12, 23, 56))
    #print(ecef_to_llh(5743689.84, 2438051.69, 1317414.17))


if __name__ == '__main__':

    test_wgs84()
