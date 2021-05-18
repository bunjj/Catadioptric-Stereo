import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('agg')


#TODO: rewrite function, give more reasonable names
#TODO: finish doc
def automatic_mirror_detection(source_path, gridgrid_size):
    ''' Computes automatic mirror detection for an input scene by applying optical flow

    Args:
        path: path to the video sequence form which the mirror should be extracted from
        gridgrid_size:

    Returns:

    '''
    cap = cv2.VideoCapture(source_path) #video loaded, romeve that
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
    # https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=4,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and define point in it on a line (y=const. along x direction)
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    frame_size = old_gray.shape  # frame size x and y are changed

    grid_size = gridgrid_size

    p0 = np.zeros((grid_size, 1, 2), np.float32)
    for point in range(grid_size):
        p0[point][0][0] = (frame_size[1] / grid_size) * (point + 1)
        p0[point][0][1] = (frame_size[0] / 2)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    diff_list = list()

    #TODO: can be made dynamic
    iterater_max = 10 # number of frames for optical flow for mirror detection
    for iteration in range(iterater_max):
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points which exist in frame and frame-1 and saves them in diff_list. Also deletes points which
        # which was good in older frames but bad at the current frame.
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            diff = p1[:, :, 0] - p0[:, :, 0]
            diff_list.append(diff)

            for k in range(len(diff_list)):
                a = (diff_list[k])[st == 1]
                index_value = a.shape[0]
                a = a.reshape((index_value, 1))
                diff_list[k] = a

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)


    # TODO: Threshold can be made dynamically
    # Assigns 1 and -1 if the point was moving right or left from frame to frame. Afterwards it checks if
    # the point was over the last frames range(iterater_max) moving right or left with a threshold of 0.7.
    # Then it looks for the latest 1 and first -1 in the list. So this code works just if the frame moves towards
    # the mirror.
    point_movement = list()
    test = 1
    for point in range(diff_list[0].shape[0]):
        sum = 0
        for frame_number in range(iterater_max):
            sum = sum + np.sign((diff_list[frame_number])[point])

        if sum >= 0.7 * iterater_max:
            point_movement.append(1)
            latest_positive_value_x_coordinate = p0[point][0][0]
        elif sum <= -0.7 * iterater_max:
            point_movement.append(-1)
            while (test):
                first_negative_value_x_coordinate = p0[point][0][0]
                test = 0
        else:
            point_movement.append(0)

    mirror_position_x = int((latest_positive_value_x_coordinate + first_negative_value_x_coordinate) / 2)

    print('automatic mirror detection leads to mirror_position_x: ', mirror_position_x)

    return mirror_position_x


#TODO: what is the input needed for?
def manual_mirror_detection(source_path, ):
    ''' Method to manually determine the mirror position by mouse input

    Args:
        source_path: Path to the input sequence

    Returns:

    '''
    cap = cv2.VideoCapture(source_path)
    ret, frame = cap.read()
    _img = frame

    params = {'BORDER_OFFSET': -1}

    def select_border(event, x, y, flags, params):
        try:
            if (event == cv2.EVENT_LBUTTONUP) and (params['BORDER_OFFSET'] == -1):
                params['BORDER_OFFSET'] = x
            else:
                print('nothing')
        except:
            print('Error occured in manual_mirror_detection.')

    cv2.namedWindow('select border', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('select border', select_border, params)
    cv2.setWindowProperty('select border', cv2.WINDOW_FULLSCREEN, 1)
    cv2.imshow('select border', _img)

    while (1):  # wait for selection or ESC key
        if params['BORDER_OFFSET'] >= 0:
            break
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    print('Offset = ' + str(params['BORDER_OFFSET']))
    x = params['BORDER_OFFSET'] # x = mirror position
    return x


def draw_mirror_line(mirror_position, path, real_output_path):
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()


    # draw black line at x
    x = mirror_position
    frame[:, x, :] = 0
    img = frame

    # flip right image horizontally
    img[:, x + 1:, :] = cv2.flip(img[:, x + 1:, :], 1)

    # determine width of stereo images
    new_width = x  # min(x, width-x)

    # draw line of right image with new width
    img[:, x + 1 + new_width, :] = 0

    # crop images and make grayscale
    # why it's necessary to crop the image ??
    imgR = cv2.cvtColor(img[:, :x, :], cv2.COLOR_BGR2GRAY)
    imgL = cv2.cvtColor(img[:, x + 1:x + 1 + new_width, :], cv2.COLOR_BGR2GRAY)

    # saves images
    cv2.imwrite(real_output_path + '/imgL.png', imgL)
    cv2.imwrite(real_output_path + '/imgR.png', imgR)
    cv2.imwrite(real_output_path + '/select_border.png', img)


def get_right_and_left_image(mirror_position, img):
    # split original image in right and left image (gray) by given mirror position
    x = mirror_position

    # flip right image horizontally
    img[:, x + 1:, :] = cv2.flip(img[:, x + 1:, :], 1)
    imgR = cv2.cvtColor(img[:, :x, :], cv2.COLOR_BGR2GRAY)
    imgL = cv2.cvtColor(img[:, x + 1:x + 1 + x, :], cv2.COLOR_BGR2GRAY)
    return imgR, imgL

#TODO: hardcoded intrinsics?;
#At the moment hard coded intrinsics
def get_intrinsics():
    K = np.array([[1333.3334, 0.0000, 480.0000],
                  [0.0000, 1333.3334, 270.0000],
                  [0.0000, 0.0000, 1.0000]])
    return K


def calculate_E_F(mirror_position, img, real_output_path, K):
    imgR, imgL = get_right_and_left_image(mirror_position, img)

    # Initiate SIFT detector
    print('SIFT_detector is called: ')
    sift = cv2.SIFT_create(contrastThreshold=0.02)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(imgL, None)
    kp2, des2 = sift.detectAndCompute(imgR, None)

    key_img = imgL.copy()

    # plots SIFT keypoints and descriptor
    plt.figure(figsize=(15, 15))
    plt.subplot(121), plt.imshow(
        cv2.drawKeypoints(imgL, kp1, key_img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
    plt.subplot(122), plt.imshow(
        cv2.drawKeypoints(imgR, kp2, key_img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
    plt.savefig(real_output_path + '/sift_features.png')

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
    print(F)
    # (https://stackoverflow.com/questions/59014376/what-do-i-do-with-the-fundamental-matrix)

    # We select only inlier points, needed for rectification
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # Essential from Fundamental
    # (https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga0c86f6478f36d5be6e450751bbf4fec0)
    # method can be changed to RANSAC if wanted.
    E, mask2 = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K, method=cv2.FM_7POINT)
    print('\nEssential vMatrix: ')
    print(E)

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(imgL, kp1, imgR, kp2, good, flags=2, outImg=None)

    point = kp1[1].pt
    for x in kp1:
        if x.pt[0] < point[0]:
            point = x.pt

    plt.figure(figsize=(15, 15))
    plt.imshow(img3)
    plt.savefig(real_output_path + '/sift_matches.png')

    return E, F, pts1, pts2


def rectification(imgR, imgL, pts1, pts2, F):
    # rectification of left and right image
    info, HL, HR = cv2.stereoRectifyUncalibrated(pts1, pts2, F, imgL.shape)
    rectL = cv2.warpPerspective(imgL, HL, (imgL.shape[1], imgL.shape[0]))
    rectR = cv2.warpPerspective(imgR, HR, (imgR.shape[1], imgR.shape[0]))
    return rectR, rectL


def computeSVD(E):
    SD, U, VT = cv2.SVDecomp(E)
    S = np.identity(3) * SD
    return (U, S, VT)


def getRotTrans(E):
    # singular values decomposition to get Rotation and Translation from left to right camera.
    U, S, VT = computeSVD(E)
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    # rotation matrix
    R = np.matmul(np.matmul(U, W.T), VT)
    # get translation in cross matrix, then vector
    tx = np.matmul(U, np.matmul(W, np.matmul(S, U.T)))
    mask = [[2, 1], [0, 2], [1, 0]]
    t = tx[[2, 0, 1], [1, 2, 0]].reshape(-1, 1)
    print('\n\nRatation from one camera to the other:\n', np.round(R, decimals=2),
          '\n\nTranslation from one camera to the other:\n', np.round(t, decimals=2))
    return (R, t)


def calculate_disparity(rectR, rectL):
    stereo = cv2.StereoSGBM_create(minDisparity=-20, numDisparities=50, blockSize=18, speckleRange=50,
                                   speckleWindowSize=30, uniquenessRatio=9)
    disparity = stereo.compute(rectL, rectR)

    return disparity