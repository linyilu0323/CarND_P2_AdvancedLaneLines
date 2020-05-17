import numpy as np
import cv2
import glob
import pickle


# define internal grid size
point_size = [6,9]

# load images for camera calibration
cal_images = glob.glob('camera_cal/calibration*.jpg')

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((point_size[0]*point_size[1],3), np.float32)
objp[:,:2] = np.mgrid[0:point_size[1], 0:point_size[0]].T.reshape(-1,2)

# Initialize arrays to store object points and image points
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Step through the list and search for chessboard corners
for idx, fname in enumerate(cal_images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (point_size[1],point_size[0]), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

# Use extracted objpoints and imgpoints to run camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# dump the results to pickle
pickle.dump([mtx, dist], open( "camera_cal_result.p", "wb" ))
