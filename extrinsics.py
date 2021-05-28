import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('agg')
from utils import *


def calculate_E_F(imgL, imgR, K, temp_path):
    '''
    Compute SIFT features on imgL and imgR
    and determine the Essential and Fundamental Matrices.

    :param np.ndarray imgL: BGR image of left view
    :param np.ndarray imgR: BGR image of right view
    :param np.ndarray K: intrinsics matrix K
    :return: tuple of Essential&Fundamental Matrix and SIFT Key-Point
    '''

    # transform left and right frames to grayscale
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    print('SIFT_detector is called: ')
    sift = cv2.SIFT_create(contrastThreshold=0.02)

    # find the keypoints and descriptors with SIFT # TODO: mask
    kp1, des1 = sift.detectAndCompute(imgL, None)
    kp2, des2 = sift.detectAndCompute(imgR, None)

    key_img = imgL.copy()

    # plots SIFT keypoints and descriptor
    canvL = cv2.drawKeypoints(imgL, kp1, key_img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    canvR = cv2.drawKeypoints(imgR, kp2, key_img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    draw_stereo(canvL, canvR, os.path.join(temp_path, '04_sift_features.png'))

    # This matcher trains cv::flann::Index on a train descriptor collection and calls its nearest search methods
    # to find the best matches. So, this matcher may be faster when matching a large train collection than the brute
    # force matcher.
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append([m])
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    # method can be changed to RANSAC if wanted.
    F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_7POINT)
    print('\nFundamental Matrix: ')
    print((F / np.abs(F).max()).round(2))
    # (https://stackoverflow.com/questions/59014376/what-do-i-do-with-the-fundamental-matrix)

    # We select only inlier points, needed for rectification
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # Essential from Fundamental
    # (https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga0c86f6478f36d5be6e450751bbf4fec0)
    # method can be changed to RANSAC if wanted.
    E, mask2 = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K, method=cv2.FM_7POINT)
    print('\nEssential vMatrix: ')
    print((E / np.abs(E).max()).round(2))

    # cv2.drawMatchesKnn expects list of lists as matches.
    canvMatches = cv2.drawMatchesKnn(imgL, kp1, imgR, kp2, good, flags=2, outImg=None)

    point = kp1[1].pt
    for x in kp1:
        if x.pt[0] < point[0]:
            point = x.pt

    cv2.imwrite(os.path.join(temp_path, '05_sift_matches.png'), canvMatches)

    return E, F, pts1, pts2


def rectification(imgL, imgR, pts1, pts2, F):
    ''' 
    Recftify left and right images, given some keypoints and the
    Fundamental matrix
    
    :param np.ndarray imgL: BGR or gray image of left view
    :param np.ndarray imgR: BGR or gray image of right view
    :param np.ndarray imgR: keypoints in left image
    :param np.ndarray imgR: keypoint in right image
    '''
    
    # TODO: why *UN*calibrated?
    heightL, widthL = imgL.shape[0], imgL.shape[1]
    info, HL, HR = cv2.stereoRectifyUncalibrated(pts1, pts2, F, (widthL, heightL))
    rectL = cv2.warpPerspective(imgL, HL, (imgL.shape[1], imgL.shape[0]))
    rectR = cv2.warpPerspective(imgR, HR, (imgR.shape[1], imgR.shape[0]))
    return rectL, rectR
