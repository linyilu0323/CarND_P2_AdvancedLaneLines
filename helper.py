import numpy as np
import cv2


def color_thd(img, threshold, colorspace, ch):
    """convert rgb image to requested colorspace and run thresholding in requested channel"""
    img_colorspace = cv2.cvtColor(img, colorspace)
    img_channel = img_colorspace[:,:,ch]
    img_bin = np.zeros_like(img_channel)
    img_bin[(img_channel >= threshold[0]) & (img_channel <= threshold[1])] = 1
    return img_bin

def sobelx_thd(img, threshold):
    """run sobel-x operator on given rgb image and thresholding"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    gray = hls[:,:,2]"""
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1])] = 1
    return sx_binary

def combined_thd(img):
    """run combined thresholding on given rgb image"""

    s_binary = color_thd(img, (170, 255), cv2.COLOR_RGB2HLS, 2)
    l_binary = color_thd(img, (120, 200), cv2.COLOR_RGB2HLS, 1)
    sx_binary = sobelx_thd(img, (20, 255))

    imgout_binary = np.zeros_like(img[:,:,0])
    imgout_binary[(s_binary==1) & (l_binary==1) | (sx_binary==1)] = 1
    #imgout_binary[(s_binary==1) | (sx_binary==1)] = 1
    return imgout_binary

def birds_eye(img, src, dst):
    """run perspective transformation to create bird's eye view"""
    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

def lane_finder(img):
    """run sliding window algorithm to fine lane pixel coordinates"""
    # Take a histogram of the bottom half of the image to find starting point
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    nwindows = 9 # number of sliding windows
    margin = 100 # width of the windows +/- margin
    minpix = 50 # minimum number of pixels found to recenter window

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(img.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_idx = []
    right_lane_idx = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current-margin
        win_xleft_high = leftx_current+margin
        win_xright_low = rightx_current-margin
        win_xright_high = rightx_current+margin

        # Identify the nonzero pixels in x and y within the window
        good_left_idx = np.nonzero((nonzerox>=win_xleft_low) & (nonzerox<=win_xleft_high) & \
        (nonzeroy>=win_y_low) & (nonzeroy<=win_y_high))[0]
        good_right_idx = np.nonzero((nonzerox>=win_xright_low) & (nonzerox<=win_xright_high) & \
        (nonzeroy>=win_y_low) & (nonzeroy<=win_y_high))[0]

        # Append these indices to the lists
        left_lane_idx.append(good_left_idx)
        right_lane_idx.append(good_right_idx)

        # Update window position if necessary
        if len(good_left_idx) >= minpix:
            leftx_current = np.int(np.average(nonzerox[good_left_idx]))
        if len(good_right_idx) >= minpix:
            rightx_current = np.int(np.average(nonzerox[good_right_idx]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_idx = np.concatenate(left_lane_idx)
        right_lane_idx = np.concatenate(right_lane_idx)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_idx]
    lefty = nonzeroy[left_lane_idx]
    rightx = nonzerox[right_lane_idx]
    righty = nonzeroy[right_lane_idx]

    return leftx, lefty, rightx, righty

def fitpoly2_curvrad_lnct(xl, yl, xr, yr, mx, my, y_eval):
    """run second order polynomial fit to the lane pixels identified and then calculate curvature and lane center"""
    p_left = np.polyfit(xl,yl,2)
    curvrad1_m = get_real_curvrad(p_left, mx, my, y_eval)
    p_right = np.polyfit(xr,yr,2)
    curvrad2_m = get_real_curvrad(p_left, mx, my, y_eval)
    curverad_m = (curvrad1_m + curvrad2_m)/2
    lane_center = get_lane_center(p_left, p_right, y_eval)
    return p_left, p_right, curverad_m, lane_center

def get_real_curvrad(p_fit, mx, my, y_eval):
    """get real world curvature radius (in meters)"""
    curvrad_m = ((1 + (2*(mx/my**2)*p_fit[0]*y_eval + (mx/my)*p_fit[1])**2)**1.5)\
    /np.absolute(2*(mx/my**2)*p_fit[0])
    return curvrad_m

def get_lane_center(p_left, p_right, y_eval):
    """calculate lane center position"""
    left_lane_pos_eval = p_left[0] * y_eval**2 + p_left[1] * y_eval + p_left[2]
    right_lane_pos_eval = p_right[0] * y_eval**2 + p_right[1] * y_eval + p_right[2]
    lane_center = (left_lane_pos_eval + right_lane_pos_eval) / 2
    return lane_center

def warped_lane_polygon(img_top, Minv, p_left, p_right):
    """make lane line plots and warp it back to original image perspective"""

    img_empty = np.zeros_like(img_top).astype(np.uint8)
    color_lane_img = np.dstack((img_empty, img_empty, img_empty))
    # generate points from curve fit
    ploty = np.linspace(0, img_top.shape[0]-1, img_top.shape[0])
    plotx_left = p_left[0] * ploty**2 + p_left[1] * ploty + p_left[2]
    plotx_right = p_right[0] * ploty**2 + p_right[1] * ploty + p_right[2]

    # fill a polygon with given points in green color
    pts_left = np.array([np.transpose(np.vstack([plotx_left, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([plotx_right, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_lane_img, np.int_([pts]), (0,255, 0))

    # plot the points
    img_lane_warped = cv2.warpPerspective(color_lane_img, Minv, (img_top.shape[1], img_top.shape[0]))
    return img_lane_warped

def warped_lanepix(img_top, Minv, left_x, left_y, right_x, right_y):
    """make detected lane pixel plots and warp it back to original image perspective"""

    img_empty = np.zeros_like(img_top).astype(np.uint8)
    color_lanepix_img = np.dstack((img_empty, img_empty, img_empty))

    # plot left and right lane pixels in red (left) and blue (right)
    color_lanepix_img[left_y, left_x] = [255, 0, 0]
    color_lanepix_img[right_y, right_x] = [0, 0, 255]

    # plot the points
    img_lanepix_warped = cv2.warpPerspective(color_lanepix_img, Minv, (img_top.shape[1], img_top.shape[0]))
    return img_lanepix_warped


def overlay_annotate_img(raw_img, lane_poly, lane_pixel, curvrad_m, offset_m):
    """Overlay lane image to original image and annotate curvature radius and offset distance"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (255, 255, 255)
    thickness = 3

    left_or_right = ''
    if offset_m > 0:
        left_or_right = 'right'
    else:
        left_or_right = 'left'

    crvrad_txt = "Curvature Radius = " + "{:5.0f}".format(curvrad_m) + " m"
    offset_txt = "Vehicle is " + "{:5.3f}".format(np.absolute(offset_m)) + " m " + left_or_right + " of lane center"

    img_comb = cv2.addWeighted(raw_img, 0.8, lane_pixel, 1, 0)
    img_comb = cv2.addWeighted(lane_poly, 0.3, img_comb, 1, 0)
    img_comb = cv2.putText(img_comb, crvrad_txt, (50, 50), font, fontScale, fontColor, thickness)
    img_comb = cv2.putText(img_comb, offset_txt, (50, 100), font, fontScale, fontColor, thickness)
    return img_comb
