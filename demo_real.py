from FrameIterator import FrameIterator
import cv2
import numpy as np
from matplotlib import pyplot as plt

from intrinsics import calibrateChessboard, stereoCalibrateChessboard
from segmentation import manual_split, lk_segmentation
from extrinsics import calculate_E_F, rectification
from utils import * # contains utility functions
from parser import make_parser
from os import path

###############################################################################
# Setup Parameters and Parse Arguments
###############################################################################

args = make_parser().parse_args()

opticalflow_path ='data/real/opticalflow/MVI_2399.MP4'
intrinsics_path = 'data/real/calibration/*.JPG'
temp_path = make_temp_dir('temp')

input_path = 'data/real/catadioptric_f-6-mm.jpg'
output_path = os.path.join(temp_path,'disparity.png')

# parameters for intrinsics calibration
intrinsics_params = dict(        
    chess_size=(9,6),
    tile_size=0.14, # <= 14 mm
    mirror='left',
    #flip=False,
    scale=0.8,
    verbose=1,
    show=True)

# parameters for lukas kanade calibration
lk_segmentation_params = dict(
    grid_size=100,
    iterater_max=100,
    verbose=1,
    show=True)

###############################################################################
# Intrinsics Calibration
###############################################################################

if args.intrinsic:
    
    # get mirror segmentation once for all intrinsics images
    mirror_seg_intr = manual_split(FrameIterator(intrinsics_path).first())

    # compute intrinsics matrix K from Chessboard 
    intr, extr = stereoCalibrateChessboard(
        filepattern=intrinsics_path,
        split_position=mirror_seg_intr,
        **intrinsics_params)
    K = intr['KL']

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
img = getDownSampledImg(0.5, img)
print(f'img.shape={img.shape}')
cv2.imwrite(path.join(temp_path,'00_input.png'), img)


# fill missing values with defaults

#TODO: wouldn't it make sense to add this above where we add the parameters from the parser? probably not that important though
if K is None:
    width, height = img.shape[1], img.shape[0]
    K = manual_K(width, height, focal_length_mm=6, sensor_width_mm=76)
if mirror_segmentation is None:
    mirror_segmentation = manual_split(img, verbose=1)

height, width, _ = img.shape
# split and flip image according to mirror position into stereo pair
imgL, imgR, maskL, maskR = split_image(img, mirror_segmentation, flip='right', temp_path=temp_path, show=True)
# calculate essential and fundamental matrices as well as the SIFT keypoints
E, F, pts1, pts2 = calculate_E_F(imgL, imgR, K, temp_path)

window_name = 'Disparity Computation'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
canvL, canvR = draw_epilines(imgL, imgR, pts1, pts2, F)
canv = draw_stereo(canvL, canvR, path.join(temp_path,'06_epilines_unrect.png'))
cv2.imshow(window_name, canv)
cv2.waitKey(0)

# compute rectified stereo pair
canvL, canvR, = rectification(canvL, canvR, pts1, pts2, F)
canv = draw_stereo(canvL, canvR, path.join(temp_path,'07_epilines_rect.png'))
cv2.imshow(window_name, canv)
cv2.waitKey(0)

rectL, rectR = rectification(imgL, imgR, pts1, pts2, F)
canv = draw_stereo(rectL, rectR, path.join(temp_path,'08_rectification.png'))
cv2.imshow(window_name, canv)
cv2.waitKey(0)

maskL, maskR = rectification(maskL, maskR, pts1, pts2, F)
canv = draw_stereo(maskL * 255, maskR * 255, path.join(temp_path,'08_rectification_masked.png'))
cv2.imshow(window_name, canv)
cv2.waitKey(0)

rectL = getDownSampledImg(0.5, rectL)
rectR = getDownSampledImg(0.5, rectR)
maskL = getDownSampledImg(0.5, maskL)
maskR = getDownSampledImg(0.5, maskR)

# compute disparity using semi-global block matching
stereo = cv2.StereoSGBM_create(minDisparity=-5, numDisparities=48, blockSize=16, speckleRange=0,
    speckleWindowSize=0, uniquenessRatio=10)
disparity = stereo.compute(rectL, rectR)

# mask disparity
mask = np.logical_and(maskL, maskR).astype(np.uint8)
disparity[mask[:,:,0]==0] = disparity.min()


###############################################################################
#  Plot Estimated Disparity Map
###############################################################################

im = plt.imshow(disparity)
plt.colorbar(im,fraction=0.046, pad=0.04)
plt.show()
