import numpy as np
import cv2

# simple I/O
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html
img = cv2.imread('blender_capture.png')
height, width, channels = img.shape
print(img.shape)


BORDER_OFFSET = -1
disparity = None


def select_border(event, x, y, flags, BORDER_OFFSET):
    if (event == cv2.EVENT_LBUTTONUP) and (BORDER_OFFSET < 0):
        BORDER_OFFSET = x
        print('Offset = ' + str(BORDER_OFFSET))
        img[:,x:x+1,:] = 0 # draw black line
        #cv2.line(temp, line_start, line_end, line_color, line_thinkness)

        img[:,x+1:,:] = cv2.flip(img[:,x+1:,:], 1)
        new_width = x # min(x, width-x) 
        imgL = cv2.cvtColor(img[:,:x,:], cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(img[:,x+1:x+1+new_width,:], cv2.COLOR_BGR2GRAY)

        print(imgL.shape)
        print(imgR.shape)

        stereo = cv2.StereoBM(numDisparities=16, blockSize=15)
        print("stereo.compute() crashes:")
        #disparity = stereo.compute(imgL,imgR)
        print('this code is not reached')



cv2.namedWindow('select border', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('select border', cv2.WINDOW_FULLSCREEN, 1)
cv2.setMouseCallback('select border', select_border, BORDER_OFFSET)

while(1):
    cv2.imshow('select border', img)

    if BORDER_OFFSET >= 0:
        cv2.namedWindow('disparity map', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('disparity map', cv2.WINDOW_FULLSCREEN, 1)
        cv2.imshow('disparity map', disparity)

    if cv2.waitKey(20) & 0xFF == 27: ## if ESC is pressed
        break
    
print(BORDER_OFFSET)
#cv2.waitKey(60000) & 0xFF # continue after keypress or after 1min = 60'000ms
cv2.destroyAllWindows() # close all windows

print(img[0][0])