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
imgL = cv2.cvtColor(img[:,:x,:], cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(img[:,x+1:x+1+new_width,:], cv2.COLOR_BGR2GRAY)

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

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(imgL, kp1, imgR, kp2, good, flags=2, outImg=None)

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
disparity = stereo.compute(imgR,imgL)
print('disparity.shape = ' + str(disparity.shape))

#cv2.namedWindow('disparity map', cv2.WINDOW_NORMAL)
#cv2.setWindowProperty('disparity map', cv2.WINDOW_FULLSCREEN, 1)
cv2.imshow('disparity map', disparity / disparity.max())

cv2.waitKey(2 * 60 * 1000) & 0xFF # continue after keypress or after 2min = 2 * 60 * 1000ms
cv2.destroyAllWindows() # close all windows