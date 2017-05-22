import numpy as np
import cv2, glob, os, json
import matplotlib.pyplot as plt

# number of x and y corners in calibration images
xcrnrCnt = 9
ycrnrCnt = 6

cal_imgs_dir = 'camera_cal/'
cal_imgs_glob = glob.glob(cal_imgs_dir+'calibration*.jpg')
output_dir = './output_images/'
output_path = output_dir+'calib_corners/'
                                        
# prepare object points
# transforms list of [0,0,0]'s into [0,0,0], [1,0,0], ..., [8,5,0]
objpt = np.zeros((xcrnrCnt*ycrnrCnt, 3), np.float32)
objpt[:,:2] = np.mgrid[0:xcrnrCnt, 0:ycrnrCnt].T.reshape(-1, 2) 

# Arrays to store object points and image points from all the images.
objpts = [] # 3d points in real world space
imgpts = [] # 2d points in image plane.

def set_map_pts(cal_imgs_glob, drawCornersOnImages=True):
    ''' Sets objpts and imgpts for each image in cal_imgs_glob.
    '''
    images = [(cv2.imread(imgpath), imgpath) for imgpath in cal_imgs_glob]
    img_size = None
    for img, imgpath in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (xcrnrCnt,ycrnrCnt), None)
    
        if found:
            objpts.append(objpt)
            imgpts.append(corners)
            if not img_size:  
                # Assumes all images of same size
                img_size = (img.shape[1], img.shape[0])

            if drawCornersOnImages:
                cv2.drawChessboardCorners(img, (xcrnrCnt,ycrnrCnt), corners, found)
                name = imgpath.split(cal_imgs_dir)[1]
                cv2.imwrite(output_path+name, img)
    return images, img_size
        
if __name__ == "__main__": 
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    corners_imgs, img_size = set_map_pts(cal_imgs_glob)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, img_size, None,None)
    camera_cal = {
        'mtx': mtx.tolist(),
        'dist': dist.tolist(),
        'rvecs': np.array(rvecs).tolist(),
        'tvecs': np.array(tvecs).tolist()
    }
    with open('camera_cal.json', 'w') as f:
        json.dump(camera_cal, f, indent=2, sort_keys=True)
