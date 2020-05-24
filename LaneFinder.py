import numpy as np
import helper as h
import cv2
import pickle


# load camera calibration
[mtx, dist] = pickle.load(open( "camera_cal_result.p", "rb" ))
left_lane_obj = h.Lane()
right_lane_obj = h.Lane()


# below function requires RGB image input
def process_image(image):

    # Un-distort the original image
    img_undist = cv2.undistort(image, mtx, dist, None, mtx)

    # Color and gradient thresholding
    img_bin = h.combined_thd(img_undist)

    # Apply perspective transform to the binary lane image
    ht = image.shape[0]
    wd = image.shape[1]
    src_points = np.array([[(200, ht),(0.43*wd, 0.65*ht), (0.57*wd, 0.65*ht), (wd-100, ht)]], dtype=np.float32)
    dst_points = np.array([[(250, ht), (250, 0), (wd-250, 0), (wd-250, ht)]], dtype=np.float32)
    img_top = h.birds_eye(img_bin, src_points, dst_points)

    # Detect the lane pixels
    if (left_lane_obj.best_fit is not None) and (right_lane_obj.best_fit is not None):
        left_x, left_y, right_x, right_y = h.lane_finder_series(img_top, left_lane_obj.best_fit, right_lane_obj.best_fit)
    else:
        left_x, left_y, right_x, right_y = h.lane_finder(img_top)

    # Determine pixel -> real world meter correlations, note that this is in "birds_eye view" space
    my = 30/720 # this is the length of dashed lane line
    mx = 3.7/730 # this is the distance between lane lines
    #left_lane_obj.lane_img_size(img_top.shape)
    #right_lane_obj.lane_img_size(img_top.shape)

    # Run polyfit and do sanity check
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)
    left_lane_obj.fit_sanity_check(left_fit)
    right_lane_obj.fit_sanity_check(right_fit)

    # Calculate curvature and offset distance
    curvrad_m_left = h.get_real_curvrad(left_lane_obj.current_fit, my, mx, img_top.shape[0])
    curvrad_m_right = h.get_real_curvrad(right_lane_obj.current_fit, my, mx, img_top.shape[0])
    curvrad_m = (curvrad_m_left + curvrad_m_right)/2
    lane_center = h.get_lane_center(left_lane_obj.current_fit, right_lane_obj.current_fit, img_top.shape[0])
    cam_veh_offset_m = (img_top.shape[1]/2 - lane_center) * mx

    # Overlay the detected lane back to original image and annotate information
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    img_lane = h.warped_lane_polygon(img_top, Minv, left_lane_obj.current_fit, right_lane_obj.current_fit)
    img_lanepx = h.warped_lanepix(img_top, Minv, left_x, left_y, right_x, right_y)
    img_out = h.overlay_annotate_img(img_undist, img_lane, img_lanepx, curvrad_m, cam_veh_offset_m)
    #img_out = h.diagnostic_img(img_out, img_top, left_x, left_y, left_lane_obj, right_x, right_y, right_lane_obj)
    return img_out
