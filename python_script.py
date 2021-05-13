import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import time
import matplotlib; matplotlib.use('agg')
# from operator import itemgetter

def mirror_detection(path):
    cap = cv2.VideoCapture(path) 
    #https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
    #https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 50,
                        qualityLevel = 0.01,
                        minDistance = 7,
                        blockSize = 5 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 4,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))


    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)


    grid_size = 10
    x = np.zeros((grid_size**2,1,2),np.float32)
    for xx in range(grid_size):
        for yy in range(grid_size):
            x[(xx*grid_size)+yy][0][0] = (800/grid_size)*(yy+1)
            x[(xx*grid_size)+yy][0][1] = (500/grid_size)*(xx+1)

    grid_size_2 = 100
    x_2 = np.zeros((grid_size_2**2,1,2), np.float32)
    for xx in range(grid_size_2):
            x_2[(xx)][0][0] = (800/grid_size_2)*(xx+1)
            x_2[(xx)][0][1] = (382)

    # p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # p0 = np.array([[[50.,382.]],[[100.,382.]],[[150.,382.]],[[200.,382.]],[[250.,382.]],[[300.,382.]],[[350.,382.]],[[400.,382.]],
    #             [[450.,382.]],[[500.,382.]],[[550.,382.]],[[600.,382.]],[[650.,382.]],[[700.,382.]],[[750.,382.]]],np.float32)
    p0 = x_2

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    # diff = np.zeros([15,1])
    diff_list = list()
    good = np.zeros([15,1])

    #while(1):
    iterater_max = 20
    for iteration in range(iterater_max):
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
            diff = p1[:,:,0]-p0[:,:,0]
            diff_list.append(diff)
            
            for xx in range(len(diff_list)):
                a = (diff_list[xx])[st==1]
                index_value = a.shape[0]
                a = a.reshape((index_value,1))
                diff_list[xx] = a
        
            
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)
                
        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
            
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
        
    good_or_not = list()
    test=1
    for first_index in range(diff_list[0].shape[0]):
        sum = 0
        for index in range(iterater_max):
            sum = sum+np.sign((diff_list[index])[first_index])
            
        if sum >= 0.7*iterater_max:
            good_or_not.append(1)
            latest_positive_value_x_coordinate = p0[first_index][0][0]
        elif sum <= -0.7*iterater_max:
            good_or_not.append(-1)
            while(test):
                latest_negative_value_x_coordinate = p0[first_index][0][0]
                test=0
        else:
            good_or_not.append(0)
            
    mirror_position_x = int((latest_positive_value_x_coordinate + latest_negative_value_x_coordinate)/2)
        
        
    print('good_or_not: ',np.array(good_or_not))
    print('mirror_position_x: ', mirror_position_x)
    print('latest_positive_value_x_coordinate: ', latest_positive_value_x_coordinate)
    print('latest_negative_value_x_coordinate: ', latest_negative_value_x_coordinate)
    
    return mirror_position_x

def draw_mirror_line(mirror_position, path):
    cap = cv2.VideoCapture(path)
    # draw black line at x
    ret, frame = cap.read()

    x = mirror_position
    frame[:,x,:] = 0
    img = frame

    # flip right image horizontally
    img[:,x+1:,:] = cv2.flip(img[:,x+1:,:], 1) 

    # determine width of stereo images 
    new_width = x # min(x, width-x) 

    # draw line of right image with new width 
    img[:,x+1+new_width, :] = 0

    # crop images and make grayscale
    # why it's necessary to crop the image ??
    imgR = cv2.cvtColor(img[:,:x,:], cv2.COLOR_BGR2GRAY)
    imgL = cv2.cvtColor(img[:,x+1:x+1+new_width,:], cv2.COLOR_BGR2GRAY)

    cv2.imshow('select border', img)
    #cv2.imshow('imgL', imgL)
    #cv2.imshow('imgR', imgR)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def get_right_and_left_image(mirror_position, img):
    x = mirror_position
    new_width = x
    # flip right image horizontally
    img[:,x+1:,:] = cv2.flip(img[:,x+1:,:], 1) 
    imgR = cv2.cvtColor(img[:,:x,:], cv2.COLOR_BGR2GRAY)
    imgL = cv2.cvtColor(img[:,x+1:x+1+new_width,:], cv2.COLOR_BGR2GRAY)
    return imgR, imgL   

def get_intrinsics():

    K = np.array([[1333.3334,    0.0000, 480.0000],
              [0.0000, 1333.3334, 270.0000],
              [0.0000,    0.0000,   1.0000]])
    return K

def calculate_E_F(mirror_position, img):
    imgR, imgL = get_right_and_left_image(mirror_position, img)
    print('SIFT_detector is called: ')
    # Initiate SIFT detector
    sift = cv2.SIFT_create(contrastThreshold = 0.02)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(imgL, None)
    kp2, des2 = sift.detectAndCompute(imgR, None)

    key_img = imgL.copy()

    # plt.figure(figsize=(15,15))
    # plt.subplot(121), plt.imshow(cv2.drawKeypoints(imgL,kp1,key_img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
    # plt.subplot(122), plt.imshow(cv2.drawKeypoints(imgR,kp2,key_img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
    # plt.show()

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    ########### (https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    # F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.RANSAC)
    F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_7POINT)
    print('Fundamental Matrix')
    print(F)
    # (https://stackoverflow.com/questions/59014376/what-do-i-do-with-the-fundamental-matrix)

    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # Essential from Fundamental
    # (https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga0c86f6478f36d5be6e450751bbf4fec0)
    #E, mask2 = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K, method=cv2.RANSAC)
    E, mask2 = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K, method=cv2.FM_7POINT)
    print('Essential Matrix')
    print(E)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(imgL, kp1, imgR, kp2, good, flags=2, outImg=None)

    point = kp1[1].pt
    for x in kp1:
        if (x.pt[0] < point[0]):
            point = x.pt
    # plt.figure(figsize=(15,15))
    # plt.imshow(img3), plt.show()

    return E, F, pts1, pts2

def rectification(imgR, imgL, pts1, pts2, F):
    info,HL,HR = cv2.stereoRectifyUncalibrated(pts1,pts2,F,imgL.shape)
    rectL = cv2.warpPerspective(imgL,HL,(imgL.shape[1], imgL.shape[0]))
    rectR = cv2.warpPerspective(imgR,HR,(imgR.shape[1], imgR.shape[0]))
    return rectR, rectL

def computeSVD(E):
    SD,U,VT = cv2.SVDecomp(E)
    S=np.identity(3)*SD
    return (U,S,VT)

def getRotTrans(E):
    # singular values decomposition
    U,S,VT = computeSVD(E)
    W = np.array([
        [0,-1,0],
        [1,0,0],
        [0,0,1]
    ])
    # rotation matrix
    R = np.matmul(np.matmul(U,W.T),VT)
    # get translation in cross matrix, then vector
    tx = np.matmul(U,np.matmul(W,np.matmul(S,U.T)))
    mask = [[2,1],[0,2],[1,0]]
    t = tx[[2,0,1],[1,2,0]].reshape(-1,1)
    # print
    print(np.round(R,decimals=2),np.round(t,decimals=2))
    return (R,t)

def calculate_disparity(rectR , rectL):
    stereo = cv2.StereoSGBM_create(minDisparity = -20,numDisparities=50, blockSize=18, speckleRange=50, speckleWindowSize=30, uniquenessRatio=9)
    
    # stereo = cv2.StereoBM_create(48,19)
    disparity = stereo.compute(rectL , rectR)

    return disparity

path = '/Users/dominikbornand/Desktop/ETHZ/FS21/3D_Vision/Catadioptric-Stereo/animation/movie_small.avi'
cap = cv2.VideoCapture(path)
ret, frame = cap.read()
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print(frame.shape)

# mirror_position = mirror_detection(path)
mirror_position = 408
draw_mirror_line(mirror_position, path)
K = get_intrinsics()
E, F, pts1, pts2 = calculate_E_F(mirror_position, frame)


size = (frame.shape[0], frame.shape[1])
print(frame.shape)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter(filename = 'outpy.avi', fourcc = fourcc, fps = 5, frameSize = (540, 399))
cv2.destroyAllWindows()
plt.close('all')

fig = plt.figure()
iterator = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret:
        imgR, imgL = get_right_and_left_image(mirror_position, frame)
        rectR , rectL = rectification(imgR, imgL, pts1, pts2, F)
        disparity = calculate_disparity(rectR, rectL)
        
        fig.canvas.draw()
        plt.imshow(disparity)

        # redraw the canvas
        fig.canvas.draw()

        # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        # display image with opencv or any operation you like
        cv2.imshow('output',img)
        cap.set(1, iterator)
        # out.write(np.uint8(img))
        print(iterator)
        iterator = iterator+5

        # cv2.imshow("plot",img)
    
    else:
        break

    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()