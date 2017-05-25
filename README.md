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
[img6]: ./output_images/test_images-unwarped/straight_lines1.jpg
[img7]: ./output_images/straight_lines1-unwarp-drawn.jpg
[img8]: ./output_images/straight_lines1-warped-drawn.jpg
[img9]: ./output_images/test_images-pipeline/straight_lines1.jpg
[img10]: ./output_images/straight_lines1-warped.jpg
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### [Template](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) for this writeup report.

### Note on inclusion of line numbers in code
Line numbers will *not* be mentioned in this report if function name is referenced since it is hard to kept line numbers in sync with code changes; where as function names are a bit more stable and is easy to search.


---
### Source Files 
The python source files are:

- `lib/calibrate.py` - calibrate camera from images and save calibration matrix and distortion coefficients to `camera_cal.json`
- `lib/camera.py` - calibrated camera interface 
- `lib/np_util.py` - utils
- `lib/lane_detection.py` - lane detection
- `lib/line_fit.py` - line fit and detection pipeline
- `test_camera.py` - write images from running `undistort` from `camera.py` to calibration images
- `run.py` - process video by running `process_image` from `line_fit.py` for each image in video if video is provided, otherwise run `process_image` and detection function outputs on test images.

The image paths are:
- `camera_cal/` - chessboard images to calibrate camera.
- `output_images/calib_corners/` - corners drawn on chessboard images using calibrated camera matrix
- `output_images/test_images/` - undistorted images using calibrated camera matrix on test images
- `output_images/test_images-unwarped/` - unwarped detection output on test images
- `output_images/test_images-warped/` - warped detection output on test images
- `output_images/test_images-pipeline/` - pipeline output on test images

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


### Single Image Pipeline

#### 1. Provide an example of a distortion-corrected image.

![][img4]
*Fig 4. Sample undistorted lane image*

Here is the original distorted version:
![][img5]
*Fig 5. Sample distorted lane image*


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image. The core function is in `filtered_dict` in `lane_detection.py`. The default function parameters are the values used for the filters.

- The yellow lines are identified using a combination of H and S channels of HSV color space, and positive and negative Sobel edges.

- The white lines are identified using a combination of V and S channels of HSC color space, and positive and negative Sobel edges.

![][img6]
*Fig 6. Sample unwarped lane detection*

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for perspective transform is in `roi_warper()` on top of `lane_detection.py`. It computes source and destination points from input image as copied below.

```python
    ht, wd = img.shape[:2]
    top = ht * .63
    btm = ht * .97
    qtr = wd/4
    src = np.float32([(220,btm),(595,top),(685,top),(1060,btm)])
    dst = np.float32([(qtr,ht),(qtr,0),(3*qtr,0),(3*qtr,ht)]) 
```

This resulted in the following source and destination points:

| x,y Source   | x,y Destination | 
|:-------------:|:-------------:| 
| 220, 698      | 320, 720      | 
| 595, 454      | 320, 0      |
| 685, 454      | 960, 0      |
| 1060, 698     | 960, 720    |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![][img7]
*Fig 7. Sample unwarped image with source points drawn*

![][img8]
*Fig 8. Sample warped image with dest points drawn*

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The lane line pixels are found via `findlinepixs_fast` and `findlinepixs` methods near the bottom of `lane_detection.py`. `findlinepixs` uses the sliding windows method to find where the mean of pixels are within the windows. Once found, `findlinepixs_fast` can be used to detect the pixels without the slower sliding windows method.

The curve fitting is done in `fit_curve` method in `line_fit.py`. The loop in lines 64 to 67 produces the formula Ay^2 +By + C.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature radius and center of individual lane lines are calculated in `set_center_and_curve_radius` method in `line_fit.py`. Line 41 sets the curve radius. It implements the formula (1 + (2Ay+B)^2 )^1.5 / abs(2A). 

The center of lane line is set on the next line  did this in lines # through # in my code in `my_other_file.py`

Lines 166 and 167 gathers these values for both lanes lines and determines the radius of the vehicle and center offset, copied here:
```python
radm = (line[L].curve_rad + line[R].curve_rad)/2
offcenter = (img_wd/2 - (line[L].center + line[R].center)/2) * xmppx
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

`output_image` function in `lane_detection.py` plots the detected and fitted lane lines.

![][img9]
*Fig 8. Sample pipeline output image with lane lines identified*

---

### Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to the project video result](./project-out.mp4)
Here's a [link to the challenge video result](./challenge-out.mp4)

---

### Reproduction

Commands to reproduce the outputs:
```sh
python lib/calibrate.py # calibrate camera from images and save calibration matrix and distortion coefficients to `camera_cal.json`
python test_camera.py # write images in output_images from running `undistort` from `camera.py` to calibration images
python run.py # produces images in output_images from test_images folder
python run.py video_in.mp4 video-out.mp4 # produces project and challenge video outputs
```

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I have used so much time in this project to get the challenge video to work ok that I am actually too embarrassed to mention it. Much of it had to do with manual fine tuning of parameters and debugging. Had I knew of the time involved, I may have chosen to attempt this project with a deep learning approach. Reason being I do not believe this manual approach can get us to a level 5 autonomy. Thus although I gained knowledge in this project, I feel like the time may be better worth if spent on a deep learning approach.

