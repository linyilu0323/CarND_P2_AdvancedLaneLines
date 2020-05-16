import matplotlib.pyplot as plt
import cv2
import helper as h
import numpy as np
import pickle

# load camera calibration
[mtx, dist] = pickle.load(open( "camera_cal_result.p", "rb" ))

# select an image to work with
#image = cv2.imread('test_images/test1.jpg')
image = cv2.imread('test_images/debug_test2.jpg')
#image = cv2.imread('test_images/straight_lines2.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Un-distort the original image
img_undist = cv2.undistort(image_rgb, mtx, dist, None, mtx)

# Color and gradient thresholding
img_bin = h.cg_thd(image)

# Apply perspective transform to the binary lane image
ysize = image.shape[0]
xsize = image.shape[1]
x_offset = 250
src_points = np.array([[(200, ysize),(0.43*xsize, 0.65*ysize), (0.57*xsize, 0.65*ysize), \
(xsize-100, ysize)]], dtype=np.float32)
#src_points = np.array([[190,720],[585,455],[705,455],[1130,720]],dtype=np.float32)
dst_points = np.array([[(x_offset, ysize), (x_offset, 0), (xsize-x_offset, 0), \
(xsize-x_offset, ysize)]], dtype=np.float32)
img_top = h.birds_eye(img_bin, src_points, dst_points)

# Run sliding windows to identify lane line pixels
left_x, left_y, right_x, right_y = h.lane_finder(img_top)

# Determine pixel -> real world meter correlations, note that this is in "birds_eye view" space
mx = 3/80 # this is the length of dashed lane line
my = 3.7/465 # this is the distance between lane lines

# Run polyfit and determine curvature of the lane and the distance between vehicle center and lane center
left_fit, right_fit, curvrad_m, lane_center = h.fitpoly2_curvrad_lnct(left_y, left_x, \
right_y, right_x, my, mx, img_top.shape[0])
cam_veh_offset_m = (img_top.shape[1]/2 - lane_center) * mx

# Overlay the detected lane back to original image and annotate information
Minv = cv2.getPerspectiveTransform(dst_points, src_points)
img_lane = h.warped_lane_polygon(img_top, Minv, left_fit, right_fit)
img_lanepx = h.warped_lanepix(img_top, Minv, left_x, left_y, right_x, right_y)
img_comb = h.overlay_annotate_img(img_undist, img_lane, img_lanepx, curvrad_m, cam_veh_offset_m)


# Plot the result
f1, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f1.tight_layout()

ax1.imshow(img_undist)
#ax1.plot(src_points[0,:,0],src_points[0,:,1],'r-',linewidth=5)
ax1.set_title('Undistorted Image', fontsize=40)

ax2.imshow(img_comb)
#ax2.scatter(left_x, left_y, c='r')
#ax2.scatter(right_x, right_y, c='b')
ax2.set_title('Detected Car Lane', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


f2, ax = plt.subplots(figsize=(12,9))
ax.imshow(img_comb)
f2.tight_layout()
