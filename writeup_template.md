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

[//]: # (Image References)

[img1]: ./output_images/calib_corners/calibration2.jpg
[img2]: ./output_images/calib_undistorted/calibration1.jpg
[img3]: ./camera_cal/calibration1.jpg
[img4]: ./output_images/test_images/test1.jpg
[img5]: ./test_images/test1.jpg
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Source Files 
The python source files are:
- `calibrate.py` - calibrate camera from images
- `camera.py` - calibrated camera interface 
- `test_camera.py` - runs `undistort` to calibration images
- `np_util.py` - numpy, cv, etc. utils
- `util.py` - python utils
- `lane_detect.py` - detect lane lines in input image
- `pipeline.py` - takes lane detection from lane_detect.py, run it through input video to produce annotated lane detection output video.


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file `calibrate.py`, where the `set_map_pts` function calls `cv2.findChessboardCorners` for each of the calibration chessboard image to get all chessboard corners in the image. These corner coordinates in the image (referred to as *image points*) will be added to the list `imgpts`.

When enough of these coordinates are obtained, they along with and their corresponding mapping of real world coordinates ([0,0,0], [1,0,0], ..., [8,5,0]) (referred to as *object points*) will be feed to `cv2.calibrateCamera` to return the camera calibration matrix and distortion coefficients that can be used in our pipeline to map images taken from the camera into non-distorted versions of it via `cv2.undistort` function.

The `set_map_pts` also calls `cv2.drawChessboardCorners` to draw out the corners that is detected by `cv2.findChessboardCorners`. Here is a sample output of corners drawn:

![][img1]
*Fig 1. Chessboard Corners Drawn*

Here is a different chessboard image after applying `cv2.undistort` function with the obtained camera calibration matrix and distortion coefficients: 

![][img2]
*Fig 2. Undistorted Chessboard Image*

Here is the original distorted image for comparison.

![][img3]
*Fig 3. Distorted Chessboard Input Image*

Calibration Outputs:
- `output_images/calib_undistored/` - Undistorted chessboard images, via `test_camera.py`
- `output_images/calib_corners/` - Chessboard images with corners drawn
- `camera_cal.json` - camera calibration matrix and distortion coefficients saved


### Single Image Pipeline

#### 1. Provide an example of a distortion-corrected image.

![][img4]
*Fig 4. Sample undistorted lane image*

Here is the original distorted version:
![][img5]
*Fig 5. Sample distorted lane image*


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

