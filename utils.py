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

    # singular values decomposition to get Rotation and Translation from left to right camera.
    U, S, VT = computeSVD(E)
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    # rotation matrix
    R = np.matmul(np.matmul(U, W.T), VT)

    print('')
    print(f'Rotation from one camera to the other:\n{R.round(2)}\n')
    
    # get translation in cross matrix, then vector
    tx = np.matmul(U, np.matmul(W, np.matmul(S, U.T)))
    mask = [[2, 1], [0, 2], [1, 0]]
    t = tx[[2, 0, 1], [1, 2, 0]].reshape(-1, 1)
    print(f'Translation from one camera to the other:\n{t.round(2)}\n')
    

    # Disentangle Rotation matrix
    rot_vec, _ = cv2.Rodrigues(R)
    angle = np.linalg.norm(rot_vec) 
    rot_vec = rot_vec/angle
    print(f'=> Rotation vector {rot_vec.flatten().round(2)} with angle {angle.round(2)}'),
    
    # Disentangle Translation vector
    distance = np.linalg.norm(t)
    disp_vec = t / distance
    print(f'=> Displacement vector {disp_vec.flatten().round(2)} with distance {distance.round(2)}')

    return (R, t)

#TODO: hardcoded intrinsics?;
#At the moment hard coded intrinsics
def get_intrinsics():
    K = np.array([[743.999, 0.0000, 480.0000],
                  [0.0000, 743.999, 272.0000],
                  [0.0000, 0.0000, 1.0000]])
    return K

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
    linesL = cv2.computeCorrespondEpilines(ptsR.reshape(-1, 1, 2), 1, F)
    linesL = linesL.reshape(-1, 3)
    canvasL = draw_lines(imgL, linesL, ptsL)


        
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    linesR = cv2.computeCorrespondEpilines(ptsL.reshape(-1, 1, 2), 1, F)
    linesR = linesR.reshape(-1, 3)
    canvasR = draw_lines(imgR, linesR, ptsR)


    return canvasL, canvasR
