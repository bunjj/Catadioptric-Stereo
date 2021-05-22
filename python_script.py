from FrameIterator import FrameIterator
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('agg')

from intrinsics import calibrateChessboard
from segmentation import manual_split, lk_segmentation
from extrinsics import calculate_E_F, rectification
from utils import * # contains utility functions
from parser import make_parser

###############################################################################
# Setup Parameters and Parse Arguments
###############################################################################

args = make_parser().parse_args()

opticalflow_path = 'data/blender/optical_flow.avi'
intrinsics_path = 'data/real/calibration/*.JPG'
temp_path = make_temp_dir('temp')

input_path = 'data/blender/blender_chessboard.png'
output_path = temp_path + '/depth.png'

# parameters for intrinsics calibration
intrinsics_params = dict(        
    chess_size=(9,6),
    tile_size=0.014, # <= 14 mm
    partition='left',
    flip=False,
    verbose=1,
    show=False)

# parameters for lukas kanade calibration
lk_segmentation_params = dict(
    grid_size=100,
    verbose=1,
    show=False)

###############################################################################
# Intrinsics Calibration
###############################################################################

if args.intrinsic:
    
    # get mirror segmentation once for all intrinsics images
    mirror_seg_intr = manual_split(FrameIterator(intrinsics_path).first())

    # compute intrinsics matrix K from Chessboard 
    K = calibrateChessboard(
        filepattern=intrinsics_path,
        split_position=mirror_seg_intr,
        **intrinsics_params)

else: K=None

###############################################################################
# Image Segmentation: Detect the mirror 
###############################################################################

if args.mirror:

    # compute split with Lukas-Kanade optical flow
    mirror_segmentation = lk_segmentation(
        path=opticalflow_path,
        **lk_segmentation_params
        )

else: mirror_segmentation=None


###############################################################################
#  Depth Estimation on input image
###############################################################################

# load input image
img = FrameIterator(input_path).first()
print(f'img.shape={img.shape}')


# fill missing values with defaults

#TODO: wouldn't it make sense to add this above where we add the parameters from the parser? probably not that important though
if K is None:
    K = get_intrinsics()
if mirror_segmentation is None:
    mirror_segmentation = manual_split(img, verbose=1)
print('here')


# split and flip image according to mirror position into stereo pair
imgL, imgR = split_image(img, mirror_segmentation, flip='left', show=False)

# calculate essential and fundamental matrices as well as the SIFT keypoints
E, F, pts1, pts2 = calculate_E_F(imgL, imgR, K, temp_path)

# compute rectified stereo pair
rectR , rectL = rectification(imgL, imgR, pts1, pts2, F)

# compute disparity using semi-global matching
stereo = cv2.StereoSGBM_create(minDisparity=-20, numDisparities=50, blockSize=18, speckleRange=50,
                                speckleWindowSize=30, uniquenessRatio=9)
disparity = stereo.compute(rectL, rectR)

# get the translation and estimate horizontal focal length
_, t = getRotTrans(E)
distance = np.linalg.norm(t)
x_focal = K[0,0]

# computedepth
depth = distance * x_focal / disparity

###############################################################################
#  Plot Estimated Depth Map
###############################################################################

plt.figure(figsize=(15, 15))
plt.imshow(disparity)
plt.colorbar()
plt.savefig(output_path)


###############################################################################
#  Code for Estimation on Multiple Frames
###############################################################################

# print('\n\n\nStart of img reading of mov clip and building of disparity map.\n')
# while(cap.isOpened()):
#     ret, frame = cap.read()

#     if not ret: # only continue if sucessful
#         break

#     imgR, imgL = split_image(frame, mirror_position, 'left')
#     imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
#     imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

#     rectR , rectL = rectification(imgR, imgL, pts1, pts2, F)
#     disparity = calculate_disparity(rectR, rectL)
    
#     fig.canvas.draw() # to be able to save the figure
#     plt.imshow(disparity)

#     # redraw the canvas
#     # https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
#     fig.canvas.draw()

#     # convert canvas to image
#     img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
#     img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

#     # img is rgb, convert to opencv's default bgr
#     img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

#     # display image with opencv or any operation you like
#     cv2.imshow('output',img)
#     cap.set(1, iterator)

#     cv2.imwrite(temp_path + '/disparity_img/{0}.png'.format(iterator), img)

#     print('Number of frame: ', iterator, ' iteration step: ', output_step)
#     iterator = iterator+output_step

#     # if the `q` key was pressed, break from the loop
#     key = cv2.waitKey(1) & 0xFF #necessary on 64-bit machines
#     if key == ord("q"):
#         break


# cap.release()
# cv2.destroyAllWindows()
# print('script has ended.')

