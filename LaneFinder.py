import numpy as np
import helper as h
import cv2
import pickle


# load camera calibration
[mtx, dist] = pickle.load(open( "camera_cal_result.p", "rb" ))

# below function requires RGB image input
def process_image(image):

    # Un-distort the original image
    img_undist = cv2.undistort(image, mtx, dist, None, mtx)

    # Color and gradient thresholding
    img_bin = h.combined_thd(img_undist)

    # Apply perspective transform to the binary lane image
    ysize = image.shape[0]
    xsize = image.shape[1]
    x_offset = 250
    src_points = np.array([[(200, ysize),(0.43*xsize, 0.65*ysize), (0.57*xsize, 0.65*ysize), \
    (xsize-100, ysize)]], dtype=np.float32)
    dst_points = np.array([[(250, ysize), (x_offset, 0), (xsize-x_offset, 0), \
    (xsize-250, ysize)]], dtype=np.float32)
    img_top = h.birds_eye(img_bin, src_points, dst_points)

    # Detect the lane pixels
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

    return img_comb
