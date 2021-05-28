from glob import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os

def split_image(img, x_split, flip, temp_path, show=False):
    """
    Split image horizontally at position x_split, crop both sides
    to a common width and flip one side.    
    
    :param img: input image
    :param x_split: position to split
    :param flip: either 'left' or 'right' 
    :type img: numpy.ndarray
    :type x_split: int
    :return: left and right image
    :rtype: tuple
    """
    if show: 
        window_name = 'Split Image into Stereo Pair'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    height, width, _ = img.shape
    widthL = x_split
    widthR = width - x_split

    # du images
    imgL = img.copy()
    imgR = img.copy()
    canvas = draw_stereo(imgL, imgR, os.path.join(temp_path,'01_sidebyside.png'))

    if show: 
        cv2.imshow(window_name, canvas)
        cv2.waitKey(0)

    # mask images
    maskL = np.ones_like(imgL)
    maskR = np.ones_like(imgR)
    maskL[:, x_split:, :] = 0
    maskR[:, :x_split, :] = 0
    imgL = np.multiply(maskL, imgL) 
    imgR = np.multiply(maskR, imgR) 
    canvas = draw_stereo(imgL, imgR, os.path.join(temp_path,'02_masked.png'))

    if show: 
        cv2.imshow(window_name, canvas)
        cv2.waitKey(0)

    # flip left or right image horizontally
    if flip=='left':
        imgL = cv2.flip(imgL, 1)
        maskL = cv2.flip(maskL, 1)
    elif flip=='right':
        imgR = cv2.flip(imgR, 1)
        maskR = cv2.flip(maskR, 1)

    # draw flipped image
    canvas = draw_stereo(imgL, imgR, os.path.join(temp_path,'03_flipped.png'))
    if show: 
        cv2.imshow(window_name, canvas)
        cv2.waitKey(0)

    if show: cv2.destroyWindow(window_name)
    return imgL, imgR, maskL, maskR

def getDownSampledImg(scale, img, verbose=0, show=False):
    """
    Down sampling image (for faster further processing)
    usage of cv2.resize :
    https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d
    
    Image downsampling function
    
    :param scale: scale of downsampling \in (0,1]
    :param img: input image
    :param verbose: verbosity (0,1,2) for logging into standard output
    :param show: show intermediate results with cv2.imshow()
    :type scale: float
    :type img: numpy.ndarray
    :type show: bool
    :return: downsampled image
    :rtype: numpy.ndarray
    """
    width  = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    if verbose >=2: print('Resized Dimensions : ', resized.shape)

    if show:
        cv2.imshow("Resized image", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return resized

def computeSVD(E):
    ''' Computes the Singular value decomposition of the essential matrix E

    :param numpy.ndarray E: Essential matrix
    :return: SVD composition
    '''

    SD, U, VT = cv2.SVDecomp(E)
    S = np.identity(3) * SD
    return (U, S, VT)

def getRotTrans(E):
    ''' Computes the rotation and Translation matrix based on the Essential Matrix

    :param numpy.ndarray E: Essential Matrix
    :return: Rotation and Translation
    '''

    R1, R2, t = cv2.decomposeEssentialMat(E)

    # Disentangle Rotation matrix R1
    rot_vec1, _ = cv2.Rodrigues(R1)
    angle1 = np.linalg.norm(rot_vec1) 
    rot_vec1 = rot_vec1/angle1
    sign1 = np.sign(rot_vec1[rot_vec1.nonzero()]).prod() # flip sign
    rot_vec1, angle1 = rot_vec1 * sign1, angle1 * sign1
    print(f'Rotation vector 1:{rot_vec1.flatten().round(2)} with angle {(angle1 * 180 / np.pi).round(2)}°')

    # Disentangle Rotation matrix R2
    rot_vec2, _ = cv2.Rodrigues(R2)
    angle2 = np.linalg.norm(rot_vec2) 
    rot_vec2 = rot_vec2/angle2

    sign2 = np.sign(rot_vec2[rot_vec2.nonzero()]).prod() # flip sign
    rot_vec2, angle2 = rot_vec2 * sign2, angle2 * sign2
    print(f'Rotation vector 2:{rot_vec2.flatten().round(2)} with angle {(angle2 * 180 / np.pi).round(2)}°')
    
    # Disentangle Translation vector
    disp_vec = t / np.abs(t).max()
    print(f'Displacement vector {disp_vec.flatten().round(2)}')

    return (R1, t)

#TODO: hardcoded intrinsics?;
#At the moment hard coded intrinsics
def get_intrinsics():
    K = np.array([[743.999, 0.0000, 480.0000],
                  [0.0000, 743.999, 272.5000],
                  [0.0000, 0.0000, 1.0000]])
    return K

def manual_K(width_px, height_px, focal_length_mm, sensor_width_mm):

    aspect_ratio = height_px / width_px
    sensor_height_mm = aspect_ratio * sensor_width_mm

    K = np.zeros((3,3))

    K[0,0] = focal_length_mm * width_px / sensor_width_mm
    K[1,1] = focal_length_mm * height_px / sensor_height_mm

    K[0,2] = width_px / 2
    K[1,2] = height_px / 2
    K[2,2] = 1.0
    return K

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def manual_E(rot_axis, angle, translation):
    R, _ = cv2.Rodrigues(rot_axis * angle)
    return np.matmul(skew(translation), R)

def manual_F(K, E):
    # K transforms from 2D-H to pixel coordinates, e.g. [0,0,1] => [cx,cy]
    # Fundamental works on pixel coordinates directly: F = K*E*inv(K) 
    # => F * K = K * E
    # => K^T * F^T = (K*E)^T
    return np.linalg.solve(K.T, np.matmul(K,E).T).T

def catadioptric_EF(tilt_deg, pivot, K):
    '''
    Compute essential and fundamental matrices for a simplified catadioptric
    stereo with a mirror tilted towards the camera. The rotation is aroung
    the y axis and the pivot point lies in the xz-plane.
    '''
    tilt_rad = tilt_deg * np.pi / 180
    axis = np.array([0,1,0])

    px, pz = pivot[0], pivot[2]
    offset_x = np.cos(tilt_rad) * np.sin(tilt_rad) * (pz + np.tan(tilt_rad)/px)
    offset_z = np.sin(tilt_rad) * np.sin(tilt_rad) * (pz + np.tan(tilt_rad)/px)
    
    offset = np.array([offset_x, 0, offset_z])
    print(2*offset)

    E_man = manual_E(axis, 2*tilt_rad, 2*offset)
    F_man = manual_F(K, E_man)
    return E_man, F_man

def make_temp_dir(parent='temp'):
    time_now = time.strftime('%Y-%m-%d--%H-%M-%S')
    temp_path = os.path.join(parent, time_now)
    Path(temp_path).mkdir(parents=True, exist_ok=True)
    return temp_path


def draw_mirror_line(x_split, path, real_output_path):
    ''' Draws the mirror into a given image at the mirror_possition, revertes the mirrored side of the image

    :param int mirror_position:
    :param numpy.ndarray img: immage on which the mirror should be drawn in
    :param output_path: path where files should be stored in
    :return: None
    '''

    cap = cv2.VideoCapture(path)
    ret, img = cap.read()

    imgL, imgR = split_image(img, x_split, 'left', show=False)

    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # saves images
    cv2.imwrite(real_output_path + '/imgL.png', imgL)
    cv2.imwrite(real_output_path + '/imgR.png', imgR)
    cv2.imwrite(real_output_path + '/select_border.png', img)

def draw_stereo(canvL, canvR, path):
    '''
    Draws two gray or BGR images into a matplotlib figure with subplots
    and stores the result into a file specified by path.
    :param numpy.ndarray canvL: left canvas to draw
    :param numpy.ndarray canvR: right canvas to draw
    :param path: path where file should be stored
    :return: None
    '''
    sep_width = 2

    # possibly add third axis to gray images
    canvL = canvL.reshape(canvL.shape[0], canvL.shape[1], -1)
    canvR = canvR.reshape(canvR.shape[0], canvR.shape[1], -1)

    # get dimensions of shapes
    heightL, widthL, nchannelsL = canvL.shape
    heightR, widthR, nchannelsR = canvR.shape

    # check that canvases are compatible
    if heightL != heightR or nchannelsL != nchannelsR:
        raise ValueError(f'Height and number of color channels do not match: {heightL} vs {heightR} and {nchannelsL} vs {nchannelsR}')
    else:
        height = heightL
        nchannels = nchannelsL


    line = np.zeros((height, sep_width, nchannels), dtype=canvL.dtype)
    canvas = np.concatenate([canvL, line, canvR], axis=1)
    cv2.imwrite(path, canvas)

    return canvas

def draw_lines(img, lines, pts):
    ''' 
    Draw epilines and their corresponding keypoints onto the 
    image and reaturn the canvas.
    :param numpy.ndarray img: input image
    :param numpy.ndarray lines: epilines
    :param numpy.ndarray pts: keypoints
    :return: None
    '''

    canv = img.copy()
    if(len(canv.shape) == 2): # make color
        canv = cv2.cvtColor(canv, cv2.COLOR_GRAY2BGR)

    r, c, _ = canv.shape
    for r, pt in zip(lines, pts):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1] ])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        canv = cv2.line(canv, (x0, y0), (x1, y1), color, 1)
        canv = cv2.circle(canv, tuple(pt), 5, color, -1)
    return canv

def draw_epilines(imgL, imgR, ptsL, ptsR, F, original='right'):
    #if original == 'right':
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    linesL = cv2.computeCorrespondEpilines(ptsR.reshape(-1, 1, 2), 2, F)
    linesL = linesL.reshape(-1, 3)
    canvasL = draw_lines(imgL, linesL, ptsL)


        
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    linesR = cv2.computeCorrespondEpilines(ptsL.reshape(-1, 1, 2), 1, F)
    linesR = linesR.reshape(-1, 3)
    canvasR = draw_lines(imgR, linesR, ptsR)


    return canvasL, canvasR
