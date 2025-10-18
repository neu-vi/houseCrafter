import csv
from typing import Union, Dict, List
import numpy as np
from skimage import measure

def normalize(a, axis=-1, order=2):
    '''
    Normalize any kinds of tensor data along a specific axis
    :param a: source tensor data.
    :param axis: data on this axis will be normalized.
    :param order: Norm order, L0, L1 or L2.
    :return:
    '''
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1

    if len(a.shape) == 1:
        return a / l2
    else:
        return a / np.expand_dims(l2, axis)

def read_mapping_csv(file, from_label, to_label):
    with open(file, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        mapping_dict = dict()
        for row in reader:
            mapping_dict[row[from_label]] = row[to_label]
        return mapping_dict

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(np.array(contours, dtype=object), 1)
    for contour in contours:
        contour = contour.astype(float)
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons

def label_mapping_2D(img: np.ndarray, mapping_dict: Dict):
    '''To map the labels in img following the rule in mapping_dict.'''
    out_img = np.zeros_like(img)
    existing_labels = np.unique(img)
    for label in existing_labels:
        out_img[img==label] = mapping_dict[label]
    return out_img

def R_from_pitch_yaw_roll(pitch, yaw, roll):
    '''
    Retrieve the camera rotation from pitch, yaw, roll angles.
    Camera orientation. R:=[v1, v2, v3], the three column vectors respectively denote the left, up,
    forward vector relative to the world system.
    Hence, R = R_z(roll)Ry_(yaw)Rx_(pitch)
    '''
    if isinstance(pitch, (float, int)):
        pitch = np.array([pitch])
    if isinstance(yaw, (float, int)):
        yaw = np.array([yaw])
    if isinstance(roll, (float, int)):
        roll = np.array([roll])
    R = np.zeros((len(pitch), 3, 3))
    R[:, 0, 0] = np.cos(yaw) * np.cos(roll)
    R[:, 0, 1] = np.sin(pitch) * np.sin(yaw) * np.cos(roll) - np.cos(pitch) * np.sin(roll)
    R[:, 0, 2] = np.cos(pitch) * np.sin(yaw) * np.cos(roll) + np.sin(pitch) * np.sin(roll)
    R[:, 1, 0] = np.cos(yaw) * np.sin(roll)
    R[:, 1, 1] = np.sin(pitch) * np.sin(yaw) * np.sin(roll) + np.cos(pitch) * np.cos(roll)
    R[:, 1, 2] = np.cos(pitch) * np.sin(yaw) * np.sin(roll) - np.sin(pitch) * np.cos(roll)
    R[:, 2, 0] = - np.sin(yaw)
    R[:, 2, 1] = np.sin(pitch) * np.cos(yaw)
    R[:, 2, 2] = np.cos(pitch) * np.cos(yaw)
    return R

def get_box_corners(center, vectors, return_faces=False):
    '''
    Convert box center and vectors to the corner-form.
    Note x0<x1, y0<y1, z0<z1, then the 8 corners are concatenated by:
    [[x0, y0, z0], [x0, y0, z1], [x0, y1, z0], [x0, y1, z1],
     [x1, y0, z0], [x1, y0, z1], [x1, y1, z0], [x1, y1, z1]]
    :return: corner points and faces related to the box
    '''
    corner_pnts = [None] * 8
    corner_pnts[0] = tuple(center - vectors[0] - vectors[1] - vectors[2])
    corner_pnts[1] = tuple(center - vectors[0] - vectors[1] + vectors[2])
    corner_pnts[2] = tuple(center - vectors[0] + vectors[1] - vectors[2])
    corner_pnts[3] = tuple(center - vectors[0] + vectors[1] + vectors[2])

    corner_pnts[4] = tuple(center + vectors[0] - vectors[1] - vectors[2])
    corner_pnts[5] = tuple(center + vectors[0] - vectors[1] + vectors[2])
    corner_pnts[6] = tuple(center + vectors[0] + vectors[1] - vectors[2])
    corner_pnts[7] = tuple(center + vectors[0] + vectors[1] + vectors[2])

    if return_faces:
        faces = [(0, 1, 3, 2), (1, 5, 7, 3), (4, 6, 7, 5), (0, 2, 6, 4), (0, 4, 5, 1), (2, 3, 7, 6)]
        return corner_pnts, faces
    else:
        return corner_pnts

def project_points_to_2d(points, cam_K, cam_T):
    '''
    transform box corners to cam system
    :param points: N x 3 coordinates in world system
    :param cam_K: cam K matrix
    :param cam_T: 4x4 extrinsic matrix with open-gl setting. (http://www.songho.ca/opengl/gl_camera.html)
                  [[v1, v2, v3, T]
                   [0,  0,  0,  1,]]
                  where v1, v2, v3 corresponds to right, up, backward of a camera
    '''
    # transform to camera system
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    points_cam = np.linalg.inv(cam_T).dot(points_h.T)
    points_cam = points_cam[:3]

    # transform to opencv system
    points_cam[1] *= -1
    points_cam[2] *= -1

    # delete those points whose depth value is non-positive.
    invalid_ids = np.where(points_cam[2] <= 0)[0]
    points_cam[2, invalid_ids] = 0.0001

    # project to image plane
    points_cam_h = points_cam / points_cam[2][np.newaxis]
    pixels = (cam_K.dot(points_cam_h)).T

    return pixels
