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

## Create 'test_images/' in output_images/ path if not exists
test_imgs_dir = 'test_images/'
output_path = output_dir+test_imgs_dir
if not os.path.exists(output_path):
    os.makedirs(output_path)

## Run test_images/ images and save in above folder
imgs_glob = glob.glob(test_imgs_dir+'*.jpg')
images = [(cv2.imread(imgpath), imgpath) for imgpath in imgs_glob]
for img, imgpath in images:
    undist = undistort(img)
    name = imgpath.split(test_imgs_dir)[1]
    cv2.imwrite(output_path+name, undist)
