import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import time
import matplotlib; matplotlib.use('agg')
import sys
import os
from utils import * #contains all the functions for brevity
from mirror_detection import manual_mirror_detection, automatic_mirror_detection
from depth_estimation import calculate_E_F, rectification, calculate_disparity
from calibration import calibrateChessboard
# segmentation
# intrinsics
# extrinsics
# stereo
from parser import make_parser
# from operator import itemgetter

args = make_parser().parse_args()


opticalflow_path = 'data/blender/animation_2_0.mkv'
calibration_path = 'data/real/calibration/*.JPG'

input_path = 'data/real/calibration/IMG_2373.JPG'

output_path = 'disparity.png'
temp_path = make_temp_dir('temp')

# variables
output_step = 30



# main code

#TODO: adjust this code depending on input parameter

if args.mirror:
    mirror_position = automatic_mirror_detection(opticalflow_path, 100)
else:
    mirror_position = manual_mirror_detection(input_path)


draw_mirror_line(mirror_position, input_path, temp_path)

# compute intrinsics matrix K from Chessboard
# TODO: use Rs and ts from mirrored view to compute extrinsics?
if args.inrinsic:
    K,_,_ = calibrateChessboard(
        filepattern = calibration_path,
        chess_size=(9,6),
        tile_size=0.014, # <= 14 mm
        split_position=mirror_position,
        partition='left',
        flip=False,
        verbose=2,
        show=False # or what ever you want #TODO: when to show?
        )
else:
    K = get_intrinsics()


#capture: loads first frame, captures one frame after another in a file, somewhat of an iterator
# Grabs, decodes and returns the next video frame.
# Parameters: [out]	image	the video frame is returned here. If no frames has been grabbed the image will be empty.
# Returns: false if no frames has been grabbed
cap = cv2.VideoCapture(input_path)
ret, frame = cap.read() #frame: first frame that comes out, #ret: no clue? TODO: find out. See comment above
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #TODO: is gray, but nobody knows why. find out if relevant


#TODO: find out what pts1 and pts2 do
#maybe matches?
# pts1 and pts2 -> good points (matches) of left and right image. Needed for rectification.
E, F, pts1, pts2 = calculate_E_F(mirror_position, frame, temp_path, K)

# Prints Rotation and Translation from left to right camera
_ , _ = getRotTrans(E)


#closes all windows
cv2.destroyAllWindows()
plt.close('all')


fig = plt.figure()
iterator = 0



print('\n\n\nStart of img reading of mov clip and building of disparity map.\n')
while(cap.isOpened()):
    ret, frame = cap.read()

    if not ret: # only continue if sucessful
        break

    imgR, imgL = split_image(frame, mirror_position, 'left')
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    rectR , rectL = rectification(imgR, imgL, pts1, pts2, F)
    disparity = calculate_disparity(rectR, rectL)
    
    fig.canvas.draw() # to be able to save the figure
    plt.imshow(disparity)

    # redraw the canvas
    # https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
    fig.canvas.draw()

    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    # display image with opencv or any operation you like
    cv2.imshow('output',img)
    cap.set(1, iterator)

    cv2.imwrite(temp_path + '/disparity_img/{0}.png'.format(iterator), img)

    print('Number of frame: ', iterator, ' iteration step: ', output_step)
    iterator = iterator+output_step

    # if the `q` key was pressed, break from the loop
    key = cv2.waitKey(1) & 0xFF #necessary on 64-bit machines
    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
print('script has ended.')

