import numpy as np
import json
import cv2
from util import Obj


with open('camera_cal.json', 'r') as f:
    global cam
    cam = json.load(f)
    cam['mtx'] = np.array(cam['mtx'])
    cam['dist'] = np.array(cam['dist'])
    cam = Obj(cam)

def undistort(img):
    return cv2.undistort(img, cam.mtx, cam.dist, None, cam.mtx)
