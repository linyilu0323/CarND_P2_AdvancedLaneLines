# **Project 2: Advanced Lane Finding**

---


[//]: # "Image References"

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./examples/thresholding_output.png "Combined Thresholding"
[image3]: ./examples/perspective_output.png "Perspective Transformed"
[image4]: ./examples/sliding_window_output.png "Sliding Window Search"
[image5]: ./examples/example_output.png "Output"
[image6]: ./examples/vid_diagnostic.png "Example Diagnostic Video"



**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

*Here I will consider the rubric points individually and describe how I addressed each point in my implementation.*

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The piece of code for the camera calibration step is contained in `CameraCalibration.py`. It is a stand-alone script to run in advance, so that the camera calibration information can be acquired and saved as an input to the image pipeline.

To use `cv2.calibrateCamera` function, we need to first start with getting the "object points" and "image points":

- The **object points** are the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.
- The **image points** are the (x, y) coordinates of the chessboard corners on the camera image. They can be aquired by running `cv2.findChessboardCorners` function.

Object points and image points are extracted from images taken from different angle, and concantanated into an array named `objpoints` and `imgpoints`. I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.

To save the calibration results, the camera matrix and distortion coefficient are exported to `camera_cal_result.p` for easier future use.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The first step of the image pipeline is to un-distort the camera image. Here, I load the camera matrix and distortion coefficient that are previously saved at `camera_cal_result.p`, and then use `cv2.undistort` function to process the image.

Below is an example of the oringal vs. undistorted image, the differences are very subtle, but if we focus on the near the edges, it will become more noticeable.

![alt text][image1]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

After un-distort the image, I then feed the image through a helper function (`helper.py`) called `combined_thd` to run a series of color and gradient thresholding.

I experimented with different color spaces and combinations with gradient, and ended up with doing color thresholding in S component and L component in HLS colorspace - this gives me good enough detection for white and yellow car lanes. In addition to that, I also used the x-direction sobel thresholding to better pick up some lane pixels when color is not as effective.

Below is the result of the "combined thresholding":

![alt text][image2]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transoform is done with a helper function (`helper.py`) called  `birds_eye`. It takes three inputs: the image to be transformed `img_bin`, the source points `src_points`, and the destination points `dst_points`.

To use the function for different sizes of images, I formulated the source and destination points as a function of image shape:

```python
ht = image.shape[0]
wd = image.shape[1]
src_points = np.array([[(200, ht),(0.43*wd, 0.65*ht), (0.57*wd, 0.65*ht), (wd-100, ht)]], dtype=np.float32)
dst_points = np.array([[(250, ht), (250, 0), (wd-250, 0), (wd-250, ht)]], dtype=np.float32)
```

Taking the example of the provided image, which has a height of 720 and width of 1280, the resultant points mapping are as follows:

|   Source   | Destination |
| :--------: | :---------: |
|  200, 720  |  250, 720   |
| 550.4, 468 |   250, 0    |
| 729.6, 468 |   1030, 0   |
| 1180, 720  |  1030, 720  |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image3]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After color and gradient thresholding, and the perspective transform. Depending if we have prior knowledge to the approximate location of car lane lines, the binary image is fed into two helper functions (`helper.py`) called `lane_finder_series` and `lane_finder`. The switching between two functions are as follows:

```python
# Detect the lane pixels
if (left_lane_obj.best_fit is not None) and (right_lane_obj.best_fit is not None):
    left_x, left_y, right_x, right_y = h.lane_finder_series(img_top, left_lane_obj.best_fit, right_lane_obj.best_fit)
else:
    left_x, left_y, right_x, right_y = h.lane_finder(img_top)
```

- Without any knowledge to where the car lane might be, we will use the traditional "sliding window" method, to determine the lane pixels. This method will first determine the starting point by calculating the histogram of the bottom half of the image. Then, the algorithm will create "windows" from bottom of the image to the top of image to search for lane pixels, and after each window, the x-position of window will be realigned based on the algorithmtic average of x-values in the current window.
- With prior knowledge of approximate location, for example, if a good fit is detected on the previous frame of the video, the abovementioned process can be accelerated by defining the search region using the curve fit from previous frame, plus and minus a width. This is very similar to the "sliding window", but the "windows" are being ultra-finely discretized, and there is no re-alignment required.

The result from this step will look like image below, with red dots showing the detected left lane pixels, and blue dots showing the right lane pixels.


![alt text][image4]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Before we go into this step, a correlation coefficient between pixel point and real-world meters is determined. Here I assume the distance of car lanes shown on the field of view is 30 meters, and the distance between car lanes is 3.7 meters. Thus, by counting how many pixels these "real-world meters" are equivalent to, we can get:

```python
my = 30/720 # how far we can see in the image
mx = 3.7/730 # distance between left and right lanes
```

The curvature can then be calculated by using the tutorial provided here in [this link](https://en.wikipedia.org/wiki/Curvature).

The only watch-out here is to bring the "magnifying coefficient" between pixel and real-world meters into the equation, with a bit of math, the final equation is formulated as:

```python
curvrad_m = ((1 + (2*(my/mx**2)*fit[0]*y_eval + (my/mx)*fit[1])**2)**1.5)\
    /np.absolute(2*(my/mx**2)*fit[0])
```

I have created a helper function called `get_real_curvrad`, it requires 4 inputs: the fit coefficient matrix; magnifying coefficient on x and y directions respectively, and the height of image for evaluation of curvature.

After determined the curvature, the position of lane centers can be determined by calling helper function `get_lane_center`, given the left and right lane fit coefficients, and the height of image. Then the distance between vehicle and lane center can be calculated by subtracting half of the width of image, then multiplied by the "horizontal direction magnifying coefficient".

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

After the lanes are detected and fitted to a polynomial function, I then overlay the results to the original image and transform them back to the normal perspective. An example can be seen in image below:

![alt text][image5]

---

### Sanity Check

Before moving on to applying the image pipeline to the video, I added a sanity check portion so that the algorithm is robust to sudden failure of detections or mis-detections.

To store the result of lane detection from frames of images, I defined a `Lane()` class:

```python
class Lane():
    def __init__(self):
        # flag to indicate if there is line detected in this frame
        self.detected = False
        # the fit array from the average of past 5 frames
        self.best_fit = None
        # array of fit arrays for past few frames
        self.past_fits = [np.array([False])]
        # array of fit arrays for all frames
        self.current_fit = np.array([0, 0, 0])
        # difference in curvature between last and new fits
        self.diffs = np.array([0, 0, 0])
        # img_size
        self.img_size = np.array([0, 0])
        # failure count
        self.fail_count = 0
```

An instance of the `Lane()` class is initiated for the left and right lane resepectively to keep track of recent detections and to perform sanity checks.

The core function of sanity check is to compare the polynomial fit of the current frame to the average of past few frames: as we can imagine, we don't expect car lanes to have abrupt changes while we're driving on the road.

If the current frame has failed to pass sanity check, I will deem the fit result to be "low-confidence", and I'll use the "best fit" (average of past few frames) to replace the fit result of current frame.

If I have multiple frames continuously to fail the sanity check, I will deem the "best fit" to be out-dated and/or low-confidence, so I will reset the "best-fit" result to allow the algorithm to restart building the benchmark again.

----

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://www.youtube.com/watch?v=pAm8lQ_ZqXQ).

I also have a diagnostic version to show all the details behind the pipeline in [this video](https://www.youtube.com/watch?v=W8NO7m9BtxQ).

An example of a screenshot of diagnostic video is shown below:

![alt text][image6]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The project video output showed pretty good result. I think the problems or issues faced include:

- **Implementing the sanity check:** I feel this is one of the most important piece to implement in this project, this algorithm will greatly improve the robustness of the code when there is a few frames that are difficult to detect - rather than displaying crazily wrong results, we can now use historical result to temporarily make the patch.

- **Picking the appropriate tresholding method:** this part takes a little bit effort and I still don't think it's anywhere close to perfection - example can be seen by looking at the video outputs for [challenge video](https://youtu.be/r5pxGCW8Tow) and the [even harder challenge video](https://youtu.be/ecpx1w_4_0c). I think when environment light condition change, or the road surface color is un-even, these will cause the thresholding to be failing.
- **Working with shaper curves:** In the [even harder challenge video](https://youtu.be/ecpx1w_4_0c), I can see in addition to the challenges from color and gradient thresholding, another big challenge is the sharp curves. When the curve is too sharp (like in the harder challenge video 00:40- 00:45), the current algorithm is unable to detect any lane pixels.
