from pathlib import Path
import numpy as np
import cv2
import time
import os


def split_image(img, x_split, flip, show=False):
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
    if show: cv2.namedWindow('Image Split', cv2.WINDOW_NORMAL)

    height, width, _ = img.shape
    widthL = x_split
    widthR = width - x_split

    # split images
    imgL = img[:, :x_split, :]
    imgR = img[:, x_split:, :]

    if show: # draw splitted images
        line = np.zeros((height, 2, 3), dtype=np.uint8)
        canvas = np.concatenate([imgL, line, imgR], axis=1)
        cv2.imshow('Image Split', canvas)
        cv2.waitKey(0)

    # crop images from center
    new_width = min(widthL, widthR)
    imgL = imgL[:, -new_width:, :]
    imgR = imgR[:, :new_width, :]

    if show: # draw cropped images
        line = np.zeros((height, 2, 3), dtype=np.uint8)
        canvas = np.concatenate([imgL, line, imgR], axis=1)
        cv2.imshow('Image Split', canvas)
        cv2.waitKey(0)

    # flip left or right image horizontally
    if flip=='right':
        imgR = cv2.flip(imgR, 1)
    elif flip=='left':
        imgL = cv2.flip(imgL, 1)

    if show: # draw flipped image
        line = np.zeros((height, 2, 3), dtype=np.uint8)
        canvas = np.concatenate([imgL, line, imgR], axis=1)
        cv2.imshow('Image Split', canvas)
        cv2.waitKey(0)

    if show: cv2.destroyWindow('Image Split')
    return imgL, imgR

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
    # get translation in cross matrix, then vector
    tx = np.matmul(U, np.matmul(W, np.matmul(S, U.T)))
    mask = [[2, 1], [0, 2], [1, 0]]
    t = tx[[2, 0, 1], [1, 2, 0]].reshape(-1, 1)
    print('\n\nRatation from one camera to the other:\n', np.round(R, decimals=2),
          '\n\nTranslation from one camera to the other:\n', np.round(t, decimals=2))
    return (R, t)

#TODO: hardcoded intrinsics?;
#At the moment hard coded intrinsics
def get_intrinsics():
    K = np.array([[1333.3334, 0.0000, 480.0000],
                  [0.0000, 1333.3334, 270.0000],
                  [0.0000, 0.0000, 1.0000]])
    return K

def make_temp_dir(parent='temp'):
    time_now = time.strftime('%Y-%m-%d--%H-%M-%S')
    temp_path = os.path.join(parent, time_now)
    Path(temp_path).mkdir(parents=True, exist_ok=True)
    Path(temp_path+'/disparity_img').mkdir(parents=True, exist_ok=True)
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