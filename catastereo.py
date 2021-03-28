import numpy as np
import cv2

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
# Camera Calibration
###############################################################################




###############################################################################
# Image Rectification
###############################################################################




###############################################################################
# Disparity Computation 
###############################################################################
stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR) 
print('disparity.shape = ' + str(disparity.shape))

#cv2.namedWindow('disparity map', cv2.WINDOW_NORMAL)
#cv2.setWindowProperty('disparity map', cv2.WINDOW_FULLSCREEN, 1)
cv2.imshow('disparity map', disparity / disparity.max())

cv2.waitKey(2 * 60 * 1000) & 0xFF # continue after keypress or after 2min = 2 * 60 * 1000ms
cv2.destroyAllWindows() # close all windows