import numpy as np
import cv2
from utils import *


def manual_split(img, verbose=0):
    ''' 
    Simple function which takes a mouse click as input for the manual mirror detection

    :param numpy.ndarray img: Image on which the mirror detection should be executed
    :param verbose: verbosity (0,1) for logging into standard output
    :return: offset of the mirror
    '''
    
    # dictionary to pass values by reference
    params = {'SELECTED_SPLIT': -1}

    # call back for mouse click
    def select_border(event, x, y, flags, params):
        try:
            if (event == cv2.EVENT_LBUTTONUP) and (params['SELECTED_SPLIT'] == -1):
                params['SELECTED_SPLIT'] = x
        except:
            print('Error occured in manual_mirror_detection.')

    # show image in window with callback
    window_name = 'Select Split'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, select_border, params)
    #cv2.setWindowProperty(window_name, cv2.WINDOW_FULLSCREEN, 1)
    cv2.imshow(window_name, img)

    while (1):  # wait for selection or ESC key
        if params['SELECTED_SPLIT'] >= 0:
            break
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyWindow(window_name)

    x_split = params['SELECTED_SPLIT']
    if verbose >= 1:  print(f'selected split: {x_split}')
    return x_split


#TODO: rewrite function, give more reasonable names
#TODO: finish doc
def lk_segmentation(path, grid_size, iterater_max=100, verbose=0, show=False):
    ''' Computes automatic mirror detection for an input scene by applying 
    Lukas-Kanade optical flow

    :param string path: path to Video sequence to compute the optical flow on
    :param int grid_size: TODO: add explanation
    :return: offset of the mirror
    '''

    cap = cv2.VideoCapture(path) #video loaded, romeve that
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
    # https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html

    #TODO: isn't this defined in the python script? wouldn't it make sense to give this over as a paremter?

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

    p0 = np.zeros((grid_size, 1, 2), np.float32)
    for point in range(grid_size):
        p0[point][0][0] = (frame_size[1] / grid_size) * (point + 1)
        p0[point][0][1] = (frame_size[0] / 2)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    diff_list = list()

    if show:
        window_name = 'Lukas-Kanade Mirror Detection'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    #TODO: can be made dynamic
    #iterater_max = number of frames for optical flow for mirror detection
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

        if show:
            img = cv2.add(frame, mask)
            cv2.imshow(window_name, img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    

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

    if ('latest_positive_value_x_coordinate' not in locals() or
        'first_negative_value_x_coordinate' not in locals()):
        raise ValueError('Did not find a valid optical flow to separate image.')
    
    mirror_position_x = int((latest_positive_value_x_coordinate + first_negative_value_x_coordinate) / 2)

    if verbose >= 1: print(f'automatically detected mirror position: {mirror_position_x}')

    

    if show:

        sep_width = 2
        height, _, nchannels = img.shape

        canvL = img[:,:mirror_position_x, :]
        canvR = img[:,mirror_position_x:, :]
        line = np.zeros((height, sep_width, nchannels), dtype=canvL.dtype)
        canvas = np.concatenate([canvL, line, canvR], axis=1)
        
        cv2.imshow(window_name, canvas)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

    return mirror_position_x





