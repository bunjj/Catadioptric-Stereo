import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('agg')
from utils import *


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
        except:
            print('Error occured in manual_mirror_detection.')

    cv2.namedWindow('select border', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('select border', select_border, params)
    #cv2.setWindowProperty('select border', cv2.WINDOW_FULLSCREEN, 1)
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



def draw_mirror_line(x_split, path, real_output_path):
    cap = cv2.VideoCapture(path)
    ret, img = cap.read()

    imgL, imgR = split_image(img, x_split, 'left', show=False)

    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    print(imgR.shape, imgL.shape)

    # saves images
    cv2.imwrite(real_output_path + '/imgL.png', imgL)
    cv2.imwrite(real_output_path + '/imgR.png', imgR)
    cv2.imwrite(real_output_path + '/select_border.png', img)




