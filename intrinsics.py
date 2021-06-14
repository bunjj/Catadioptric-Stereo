import cv2
import numpy as np
from utils import *
from FrameIterator import FrameIterator


def calibrateChessboard(
    filepattern,
    chess_size,
    tile_size,
    split_position=None,
    partition=None,
    flip=False,
    scale=1.0,
    verbose=0,
    show=False,
    ):
    """
    Compute the camera intrinsics K using from multiple views containing the same
    chessboard. The computation can be done on a possibly flipped partition
    of the view, which is specifically usefull for the catadioptric setup.
    
    Take care: if chessboard has 10x7 white+black squares then width is 9, height 6
    Square size in meters
    
    returning values of this function :
    https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
    
    function proposition from this blog article :
    https://aliyasineser.medium.com/opencv-camera-calibration-e9a48bdd1844
    
    :param filepattern: a pattern to specify filepath for calibration
    :param chess_size: (width, height) of the chessboard
    :param tile_size: edge size of one chessboard tile in meters
    :param split_position: location to perform horizontal split
    :param partition: partition to work on (None, 'left', 'right')
    :param flip: flip partition in case it is mirrored
    :param scale: scale to downsample view for performance \in (0,1]
    :param verbose: verbosity (0,1,2) for logging into standard output
    :param show: show intermediate results with cv2.imshow()
    :type filepattern: string
    :type chess_size: tuple
    :type tile_size: float
    :type split_position: int
    :type partition: string
    :type flip: bool
    :type scale: float
    :type verbose: int
    :type show: bool
    :return: intrinsic matirx K, rotation matrices Rs and translation vectors ts
    :rtype: tuple
    """

    if verbose >= 1: print('Calibrate Camera with Chessboard:')

    width, height = chess_size

    frame_iter = FrameIterator(filepattern, verbose-1)

    if show: 
        window_name = 'Detected Chessboard Pattern'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # prepare key points in object coordinates, i.e. chessboard corners
    points_obj = np.zeros((height*width, 3), np.float32)
    points_obj[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    points_obj = points_obj * tile_size


    # for every view, store list of points in object and camera coordinates
    list_points_obj = []  # 3D point in object coordinates
    list_points_pix = []  # 2D points pixel coordinates
    for view in frame_iter:

        if verbose >= 1: print(f'processing \'{frame_iter.current_frame()}\'')

        # possibly partition image
        if split_position is not None and partition is None:
            view = view
        elif partition == 'left':
            view = view[:, :split_position, :]
        elif partition == 'right':
            view = view[:, split_position:, :]
        else:
            raise ValueError('Unkown partition specification: ' + partition)

        if flip: # possibly flip image
            view = cv2.flip(view, 1)


        # convert view to grayscale
        view_gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
                    
        if scale < 1.0: # possibly downsample view for performance
            view_gray = getDownSampledImg(scale, view_gray)


        # find the chess board corners pixel coordinates
        flags = None
        ret, points_pix = cv2.findChessboardCorners(view_gray, (width, height), flags)

        if not ret: # continue with next frame if unsuccessful
            continue 

        # potentially upsample
        points_pix = points_pix/scale

        # refine corner locations iteratively with termination criteria
        # https://docs.opencv.org/master/dd/d92/tutorial_corner_subpixels.html
        winSize = (11, 11)
        zeroZone = (-1, -1)
        term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        points_pix = cv2.cornerSubPix(view_gray, points_pix, winSize, zeroZone, term_criteria)
        
        # append the points from current view to list
        list_points_obj.append(points_obj)
        list_points_pix.append(points_pix)

        if show: # Draw corners and and display the corners
            view = cv2.drawChessboardCorners(view, (width, height), points_pix, ret)
            cv2.imshow(window_name, view)
            cv2.waitKey(0)

    if show: cv2.destroyWindow(window_name)

    n_valid_views = len(list_points_pix)
    if verbose >=1: print(f'number of valid views: {n_valid_views}')

    if n_valid_views == 0:
        raise RuntimeError('Could not detect any chessboard key points')

    # compute camera calibration
    K_init = None # inital intrinsics matrix 
    D_init = None # intial distortion params
    rmse, K_res, D_res, Rs, ts = cv2.calibrateCamera(
        list_points_obj,
        list_points_pix,
        view_gray.shape[::-1],
        K_init, D_init
    )

    if verbose >= 1:
        print(f'root mean squared error: {rmse}')
        print(f'intrinsics matrix K:\n{K_res}')

    return K_res #, Rs, ts)

def stereoCalibrateChessboard(
    filepattern,
    chess_size,
    tile_size,
    split_position=None,
    mirror='right',
    scale=1.0,
    verbose=0,
    show=False,
    ):
    """
    Compute the camera intrinsics and extrinsics from a catadioptric images containing
    the same chessboard.
    
    Take care: if chessboard has 10x7 white+black squares then width is 9, height 6
    Square size in meters
    
    returning values of this function :
    https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
    
    function proposition from this blog article :
    https://aliyasineser.medium.com/opencv-camera-calibration-e9a48bdd1844
    
    :param filepattern: a pattern to specify filepath for calibration
    :param chess_size: (width, height) of the chessboard
    :param tile_size: edge size of one chessboard tile in meters
    :param split_position: location to perform horizontal split
    :param mirror: partition which is mirrored ('left', 'right')
    :param scale: scale to downsample view for performance \in (0,1]
    :param verbose: verbosity (0,1,2) for logging into standard output
    :param show: show intermediate results with cv2.imshow()
    :type filepattern: string
    :type chess_size: tuple
    :type tile_size: float
    :type split_position: int
    :type mirror: string
    :type scale: float
    :type verbose: int
    :type show: bool
    :return: intrinsic matirx K, rotation matrices Rs and translation vectors ts
    :rtype: tuple
    """

    if verbose >= 1: print('Stereo Calibrate Camera with Chessboard:')

    width, height = chess_size

    frame_iter = FrameIterator(filepattern, verbose-1)

    if show: 
        window_name = 'Detected Chessboard Pattern'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # prepare key points in object coordinates, i.e. chessboard corners
    points_obj = np.zeros((height*width, 3), np.float32)
    points_obj[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    points_obj = points_obj * tile_size


    # for every view, store list of points in object and camera coordinates
    list_points_obj = []  # 3D point in object coordinates
    list_pointsL_pix = [] # 2D points pixel coordinates in left view
    list_pointsR_pix = [] # 2D points pixel coordinates in right view
    for view in frame_iter:

        if verbose >= 1: print(f'processing \'{frame_iter.current_frame()}\'')

        viewL, viewR, _, _ = split_image(view, split_position, mirror, None)

        # convert view to grayscale
        viewL_gray = cv2.cvtColor(viewL, cv2.COLOR_BGR2GRAY)
        viewR_gray = cv2.cvtColor(viewR, cv2.COLOR_BGR2GRAY)
                    
        if scale < 1.0: # possibly downsample view for performance
            viewL_gray = getDownSampledImg(scale, viewL_gray)
            viewR_gray = getDownSampledImg(scale, viewR_gray)


        # find the chess board corners pixel coordinates
        flags = None
        retL, pointsL_pix = cv2.findChessboardCorners(viewL_gray, (width, height), flags)
        retR, pointsR_pix = cv2.findChessboardCorners(viewR_gray, (width, height), flags)

        if not (retR and retL): # continue with next frame if unsuccessful
            continue 

        # potentially upsample
        pointsL_pix = pointsL_pix/scale
        pointsR_pix = pointsR_pix/scale

        # refine corner locations iteratively with termination criteria
        # https://docs.opencv.org/master/dd/d92/tutorial_corner_subpixels.html
        winSize = (11, 11)
        zeroZone = (-1, -1)
        term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
        pointsL_pix = cv2.cornerSubPix(viewL_gray, pointsL_pix, winSize, zeroZone, term_criteria)
        pointsR_pix = cv2.cornerSubPix(viewR_gray, pointsR_pix, winSize, zeroZone, term_criteria)
        
        # append the points from current view to list
        list_points_obj.append(points_obj)
        list_pointsL_pix.append(pointsR_pix)
        list_pointsR_pix.append(pointsL_pix)

        if show: # Draw corners and and display the corners
            viewL = cv2.drawChessboardCorners(viewL, (width, height), pointsL_pix, retL)
            viewR = cv2.drawChessboardCorners(viewR, (width, height), pointsR_pix, retR)
            cv2.imshow(window_name, draw_stereo(viewL, viewR, 'temp/'+str(frame_iter.index)+'.png'))
            cv2.waitKey(0)

    if show: cv2.destroyWindow(window_name)

    n_valid_views = len(list_pointsL_pix)
    if verbose >=1: print(f'number of valid views: {n_valid_views}')

    if n_valid_views == 0:
        raise RuntimeError('Could not detect any chessboard key points')

    # compute camera calibration
    imgSize = viewL_gray.shape[::-1]

    # initillize with standard parameters
    K_init, D_init = cv2.initCameraMatrix2D(
        2*list_points_obj, 
        list_pointsL_pix + list_pointsR_pix, 
        imgSize, aspectRatio=1), None

    if verbose >= 1:
        print(f'initial intrinsics matrix K_init:\n{K_init}')

    # determine focal length from all available catadioprtric views
    rmse, K_init, D_init, Rs, ts = cv2.calibrateCamera(
        2*list_points_obj,
        list_pointsL_pix + list_pointsR_pix, 
        imgSize,
        K_init, D_init,
        flags=(
            + cv2.CALIB_USE_INTRINSIC_GUESS 
            + cv2.CALIB_FIX_PRINCIPAL_POINT 
            + cv2.CALIB_FIX_ASPECT_RATIO
            + cv2.CALIB_RATIONAL_MODEL
        )
    )

    if verbose >= 1:
        print(f'root mean squared error mono calibration: {rmse}')
        print(f'intrinsics matrix K_init:\n{K_init}')

    KL_init, KR_init = K_init, K_init # inital intrinsic matrix params
    DL_init, DR_init = D_init, D_init # intial distortion params

    rmse, KL, DL, KR, DR, R, T, E, F = cv2.stereoCalibrate(
        list_points_obj,
        list_pointsL_pix,
        list_pointsR_pix,
        KL_init, DL_init,
        KR_init, DR_init,
        imgSize,
        flags=(
            + cv2.CALIB_USE_INTRINSIC_GUESS 
            #+ cv2.CALIB_FIX_INTRINSIC
            + cv2.CALIB_FIX_PRINCIPAL_POINT 
            + cv2.CALIB_FIX_ASPECT_RATIO
            #+ cv2.CALIB_ZERO_TANGENT_DIST
            + cv2.CALIB_SAME_FOCAL_LENGTH
            + cv2.CALIB_RATIONAL_MODEL
            #+ cv2.CALIB_FIX_K3
            #+ cv2.CALIB_FIX_K4
            #+ cv2.CALIB_FIX_K5
        )
    )
    
    if verbose >= 1:
        print(f'root mean squared error: {rmse}')
        print(f'intrinsics matrix K:\n{KL}')
        getRotTrans(E)


    intr = dict(KL=KL, DL=DL, KR=KR, DR=DR)
    extr = dict(R=R, T=T, E=F, F=F)
    return intr, extr