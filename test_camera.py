import cv2, os, glob
from lib.camera import undistort
from lib.calibrate import cal_imgs_dir, cal_imgs_glob, output_dir


## Use the calibrated camera's undistort function to run test images

## Create 'calib_undistorted/' in output_images/ path if not exists
output_path = output_dir+'calib_undistorted/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

## Run camera_cal/ images and save in above folder
images = [(cv2.imread(imgpath), imgpath) for imgpath in cal_imgs_glob]
for img, imgpath in images:
    undist = undistort(img)
    name = imgpath.split(cal_imgs_dir)[1]
    cv2.imwrite(output_path+name, undist)

## Run test_images/ images and save in above folder
test_imgs_glob = glob.glob('test_images/*.jpg')
images = [(cv2.imread(imgpath), imgpath) for imgpath in test_imgs_glob]
for img, imgpath in images:
    undist = undistort(img)
    name = imgpath.split('test_images/')[1]
    cv2.imwrite(output_path+name, undist)
