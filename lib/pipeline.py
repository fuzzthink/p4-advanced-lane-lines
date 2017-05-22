import numpy as np
import cv2, os, sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from types import SimpleNamespace as SNS
from lib.lane_detection import *
from lib.camera import undistort

linNorm = np.linalg.norm

# Define conversions in x and y from pixels space to meters
ymppx = 30/720 # meters per pixel in y dimension
xmppx = 3.7/700 # meters per pixel in x dimension
img_wd, img_ht = (1280, 720)
poly_order = 2
ylinspace = np.linspace(0, img_ht-1, img_ht)
MA_len = 7

class Line:
    def __init__(self):
        self.fit = None 
        self.fits = [] 
        self.fit_MA = None

        self.coef = None
        self.coefs = []
        self.coef_MA = None
        self.fit_err = None

        self.curve_rad = None
        self.center = None
        self.MA_len = MA_len
        self.found = False

    def set_center_and_curve_rad(self, coef=None, poly_order=poly_order):
        if coef is None:
            coef = self.coef_MA
        self.curve_rad = ((ymppx**2 + xmppx**2*(2*coef[0]*img_ht + coef[1])**2)**1.5) / (2*xmppx**2 * coef[0])
        dist = -img_wd/2
        for i in range(poly_order+1):
            dist += coef[i]*img_ht**(poly_order-i)
        self.center = dist * xmppx

    def set_all(self, set_center_and_curve_rad=True):
        if len(self.coefs) >= self.MA_len: 
            self.fits.pop(0)
            self.coefs.pop(0)
        
        self.fits.append(self.fit)
        self.fit_MA = np.average(np.array(self.fits), axis=0)
        self.coefs.append(self.coef)
        self.coef_MA = np.average(np.array(self.coefs), axis=0)
        if set_center_and_curve_rad:
            self.set_center_and_curve_rad()

    def fit_curve(self, xCoords, yCoords, poly_order=poly_order):
        self.found = True
        self.coef, residuals,_,_,_ = np.polyfit(yCoords, xCoords, poly_order, full=True)
        self.fit_err = residuals/len(xCoords)**1.2
        
        self.fit = 0 # will be a list
        for i in range(poly_order):
            self.fit += self.coef[i] * ylinspace**(poly_order-i)
        self.fit += self.coef[poly_order]
        if self.fit_MA == None: # only once per L/R instantiation
            self.set_all(False)
        self.set_center_and_curve_rad(self.coef)

    def fit_diff(self):
        ''' Returns norm of difference of current fit x positions vs. its MA
        '''
        return linNorm(self.fit - self.fit_MA)


_max = SNS(
    fitdiff= 1700, # poor left lane fit if < 1700 on project vid @38sec
    fit_err= 80,
    centr_dif= 800*xmppx, # lane center difference in meters
    curvature= 1500,
    fail= 2,
)
_min_centr_dif = 500*xmppx

def bad_curvature(a, b):
    return False if max(abs(a), abs(b)) > _max.curvature else a*b < 0

def bad_fit(fit1, fit2, radius1, radius2):
    fit1 = np.array(fit1)
    fit2 = np.array(fit2)
    fit_diff = fit1-fit2
    min_fits = np.minimum(np.absolute(fit1), np.absolute(fit2)) 
    return linNorm(fit_diff[:2]/min_fits[:2]) * 2000/(abs(radius1)+abs(radius2)) > 10

def fitMsg(fit_diff, fit_err):
    if fit_diff > _max.fitdiff and fit_err > _max.fit_err:
        return 'Bad - Fit diff vs. MA: %5d, Fit error too big: %5d' % (fit_diff, fit_err)
    elif fit_diff > _max.fitdiff:
        return 'Bad - Fit diff vs. MA: %5d' % fit_diff
    elif fit_err > _max.fit_err:
        return 'Bad - Fit error too large: %5d' % fit_err
    else:
        return 'OK - Fit diff vs. MA: %5d' % fit_diff

line = [Line(), Line()]
fail_detect = [2, 2]

def pipeline(_img):
    img = undistort(_img)
    detect = LaneDetect(img)
    main_img = np.zeros_like(img).astype(np.uint8)
    coef = [None, None]
    fitmsg = ['', '']
    detectmsg = ''

    if fail_detect[L] >= _max.fail or fail_detect[R] >= _max.fail:
        detect.findlinepixs()
        detectmsg = 'slow'
    else:
        for side in LR:
            coef[side] = line[side].coef_MA
        detect.findlinepixs_fast(coef)
        detectmsg = 'fast'

    for side in LR:
        pxs = detect.pxCoords
        if detect.found(side):
            line[side].fit_curve(pxs.x[side], pxs.y[side])
            line[side].found = True
        else:
            fail_detect[side] += 1
            line[side].found = False

    badCurve = bad_curvature(line[L].curve_rad, line[R].curve_rad)
    badFit = bad_fit(line[L].coef, line[R].coef, line[L].curve_rad, line[R].curve_rad)
    badCenter = not _min_centr_dif <= abs(line[L].center - line[R].center) <= _max.centr_dif

    for side in LR:
        fit_diff = line[side].fit_diff()
        fit_err = line[side].fit_err

        if badCurve or badFit or badCenter:
            if fit_diff > _max.fitdiff or fit_err > _max.fit_err:
                if detect.found(side): # +1 already if not found
                    fail_detect[side] += 1
            fitmsg[side] = fitMsg(fit_diff, fit_err)
        else:
            if fit_err > _max.fit_err:
                if detect.found(side): # +1 already if not found
                    fail_detect[side] += 1
            elif line[side].found and fit_diff < _max.fitdiff:
                fail_detect[side] = 0
                line[side].set_all()
            fitmsg[side] = fitMsg(fit_diff, fit_err)

        coef[side] = line[side].coef_MA
        pts = np.array(np.vstack((line[side].fit_MA, ylinspace)).T, dtype=np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(detect.dbg_wins[2], [pts], False, (0,255,0), thickness=10)
        
        pts = np.array(np.vstack((line[side].fit, ylinspace)).T, dtype=np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(detect.dbg_wins[2], [pts], False, (255,0,0), thickness=2)
        line[side].set_center_and_curve_rad()

    radm = (line[L].curve_rad + line[R].curve_rad)/2
    offcenter = xmppx*(line[L].center + line[R].center)/2

    out = detect.plot_curve(coef)
    return draw.with_debug_wins(out, detect.dbg_wins, offcenter, radm, fitmsg, detectmsg)
