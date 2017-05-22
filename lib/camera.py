import numpy as np
import cv2, json
from types import SimpleNamespace as SNS


with open('camera_cal.json', 'r') as f:
    global cam
    camera = json.load(f)
    cam = SNS(
        mtx =np.array(camera['mtx']),
        dist=np.array(camera['dist']),
    )

def undistort(img):
    return cv2.undistort(img, cam.mtx, cam.dist, None, cam.mtx)
