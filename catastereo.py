import numpy as np
import cv2
from matplotlib import pyplot as plt
from operator import itemgetter

# read BGR image
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html
img = cv2.imread('blender_capture.png')
height, width, channels = img.shape
print("img.shape = " + str(img.shape))


###############################################################################
# Select the image border manually 
###############################################################################

# create mouse callback with dictionary to pass values by reference
# https://answers.opencv.org/question/11536/passing-and-returning-values-from-mouse-handler-function/?answer=11537
params = {'BORDER_OFFSET':-1}
def select_border(event, x, y, flags, params):
    if (event == cv2.EVENT_LBUTTONUP) and (params['BORDER_OFFSET'] == -1):
        params['BORDER_OFFSET'] = x

# open dynamic window with mouse callback
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_mouse_handling/py_mouse_handling.html
cv2.namedWindow('select border', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('select border', select_border, params)
cv2.setWindowProperty('select border', cv2.WINDOW_FULLSCREEN, 1)
cv2.imshow('select border', img)

while(1): # wait for selection or ESC key    
    if params['BORDER_OFFSET'] >= 0: 
        break
    if cv2.waitKey(20) & 0xFF == 27: 
        break
    

###############################################################################
# Split into left and right image 
###############################################################################
print('Offset = ' + str(params['BORDER_OFFSET']))
x = params['BORDER_OFFSET']

# draw black line at x
img[:,x,:] = 0

# flip right image horizontally
img[:,x+1:,:] = cv2.flip(img[:,x+1:,:], 1) 

# determine width of stereo images 
new_width = x # min(x, width-x) 

# draw line of right image with new width 
img[:,x+1+new_width, :] = 0

# crop images and make grayscale
imgR = cv2.cvtColor(img[:,:x,:], cv2.COLOR_BGR2GRAY)
imgL = cv2.cvtColor(img[:,x+1:x+1+new_width,:], cv2.COLOR_BGR2GRAY)

cv2.imshow('select border', img)
#cv2.imshow('imgL', imgL)
#cv2.imshow('imgR', imgR)

###############################################################################
# Intrinsics K
###############################################################################

K = np.array([[1333.3334,    0.0000, 480.0000],
              [0.0000, 1333.3334, 270.0000],
              [0.0000,    0.0000,   1.0000]])

###############################################################################
# SIFT Features
###############################################################################
print('SIFT_detector is called: ')

# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(imgL, None)
kp2, des2 = sift.detectAndCompute(imgR, None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

########### (https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html)
def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

# (https://stackoverflow.com/questions/59014376/what-do-i-do-with-the-fundamental-matrix)

# We select only inlier points
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

# Essential from Fundamental
# (https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga0c86f6478f36d5be6e450751bbf4fec0)
E, mask2 = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K)
print(E)
print(mask2)

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(imgL, imgR, lines1, pts1, pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(imgR, imgL, lines2, pts2, pts1)
plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img3)
plt.show()
###########


# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(imgR, kp1, imgR, kp2, good, flags=2, outImg=None)

point = kp1[1].pt
for x in kp1:
     if (x.pt[0] < point[0]):
        point = x.pt

plt.imshow(img3), plt.show()

print('SIFT_detector has ended: ')


###############################################################################
# Camera Calibration
###############################################################################




###############################################################################
# Image Rectification
###############################################################################




###############################################################################
# Disparity Computation 
###############################################################################
stereo = cv2.StereoSGBM_create(numDisparities=50, blockSize=8, speckleRange=50, speckleWindowSize=30, uniquenessRatio=9)
disparity = stereo.compute(imgL,imgR)
print('disparity.shape = ' + str(disparity.shape))

#cv2.namedWindow('disparity map', cv2.WINDOW_NORMAL)
#cv2.setWindowProperty('disparity map', cv2.WINDOW_FULLSCREEN, 1)
cv2.imshow('disparity map', disparity / disparity.max())

cv2.waitKey(2 * 60 * 1000) & 0xFF # continue after keypress or after 2min = 2 * 60 * 1000ms
cv2.destroyAllWindows() # close all windows