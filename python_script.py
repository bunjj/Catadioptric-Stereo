import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import time
import matplotlib; matplotlib.use('agg')
import sys
import os
import argparse
from helper import * #contains all the functions for brevity
# from operator import itemgetter

text = "This is a program to compute the disparity map and depth estimation of an mirror image."
parser = argparse.ArgumentParser(description= text)

#add argument for source file
parser.add_argument("-s", "--source", help = "input file to apply depth estimation on")
#add argument for target file
parser.add_argument("-t", "--target", help = "target path to store files in")
#add argument for intrinsic calibration
#TODO: adjust implementation for intrinsics! respectively add it
parser.add_argument("-i", "--inrinsic",
                    help = "compute the intrinsic parameters from chessboard images,"
                           "otherwise they will be loaded form a NP file",
                    action = "store_true")
#add parameter for mirror detection
#can be adjusted to take a path to a video file as well
#TODO: hand in path for mirror detection
parser.add_argument("-m", "--mirror",
                    help = "mirror calibration with optical flow,"
                           "if parameter not set, mirror detection must be done manually",
                    action = "store_true")
#TODO:adjust implementation such that this is possible
parser.add_argument("-l", "--load",
                    help = "loads the previous mirror offset from a tmp file",
                    action = "store_true")

args = parser.parse_args()

if not args.mirror: 
    mirror_detection_case = 1 # manual mirror detection
else:
    mirror_detection_case = 2 # automatic mirror detection

# variables
output_step = 30
automatic_mirror_detection_grid_size = 100

if args.source:
    path = args.source
else:
    path = 'animation/current.avi' #relative path suffices usually

if args.target:
    output_path = args.target
else:
    output_path = '../img/'# path where disparity images are saved


# functions




# main code
#capture: loads first frame, captures one frame after another in a file, somewhat of an iterator

# Grabs, decodes and returns the next video frame.
# Parameters: [out]	image	the video frame is returned here. If no frames has been grabbed the image will be empty.
# Returns: false if no frames has been grabbed
cap = cv2.VideoCapture(path)
ret, frame = cap.read() #frame: first frame that comes out, #ret: no clue? TODO: find out. See comment above
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #TODO: is gray, but nobody knows why. find out if relevant

time_now = str(time.asctime( time.localtime(time.time()) )) #create output folder depending on time and date
real_output_path = (output_path+time_now) #TODO: rename path names, not clean done. Is used in different helper functions
os.mkdir(real_output_path)
os.mkdir(real_output_path+'/disparity_img')

#TODO: adjust this code depending on input parameter
try:
    if mirror_detection_case == 1:
        mirror_position = manual_mirror_detection(path)
    elif mirror_detection_case == 2:
        mirror_position = automatic_mirror_detection(path, automatic_mirror_detection_grid_size)
except:
    print('There is no mirror detection activated!')



draw_mirror_line(mirror_position, path, real_output_path)
K = get_intrinsics() #TODO: add intrinsic computation
#TODO: find out what pts1 and pts2 do
#maybe matches?
# pts1 and pts2 -> good points (matches) of left and right image. Needed for rectification.
E, F, pts1, pts2 = calculate_E_F(mirror_position, frame, real_output_path, K)

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

    if ret:
        imgR, imgL = get_right_and_left_image(mirror_position, frame)
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

        cv2.imwrite(output_path+'/'+time_now+'/disparity_img/{0}.png'.format(iterator), img)

        print('Number of frame: ', iterator, ' iteration step: ', output_step)
        iterator = iterator+output_step
    
    else:
        break

    key = cv2.waitKey(1) & 0xFF #TODO: find out what this does, might be the reason why one has to press q


    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print('script has ended.')

