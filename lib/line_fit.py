import numpy as np
import cv2

from types import SimpleNamespace as SNS
from lib.lane_detection import *
from lib.camera import undistort


linNorm = np.linalg.norm

ymppx = 30/720  # meters per px in y dimension (lane ~30 meters long, 720=ht)
xmppx = 3.7/700 # meters per px in x dimension (lane ~3.7m wide, ~700px btw lanes) 
img_wd, img_ht = (1280, 720)
poly_order = 2
_0_to_ht = np.linspace(0, img_ht-1, img_ht) # ary of [0,1,..,ht-1]
MA_len = 7 # moving avg period

class Line:
    ''' Line fit for one lane line
    '''
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

    def set_center_and_curve_radius(self, coef=None):
        if coef is None:
            coef = self.coef_MA
        y = img_ht-1
        ## curve radius formula: (1 + (2Ay+B)^2 )^1.5 / abs(2A)
        self.curve_rad = ((1 + (2*coef[0]*y*ymppx + coef[1])**2)**1.5) / np.absolute(2*coef[0])
        self.center = coef[0]*(y**2) + coef[1]*y + coef[2]

    def set_all(self, set_center_and_curve_radius=True):
        if len(self.coefs) >= self.MA_len:
            self.fits.pop(0)
            self.coefs.pop(0)
        
        self.fits.append(self.fit)
        self.fit_MA = np.average(np.array(self.fits), axis=0)
        self.coefs.append(self.coef)
        self.coef_MA = np.average(np.array(self.coefs), axis=0)
        if set_center_and_curve_radius:
            self.set_center_and_curve_radius()

    def fit_curve(self, xCoords, yCoords):
        self.found = True
        coef, residuals,_,_,_ = np.polyfit(yCoords, xCoords, poly_order, full=True)
        self.fit_err = residuals/len(xCoords)**1.2
        self.fit = 0 # self.fit will be a list of len of coef[i] after += below
                     # 0 is the init vals for list
        
        ## loop produces formula: Ay^2 +By + C
        for i in range(poly_order):
            self.fit += coef[i] * _0_to_ht**(poly_order-i)
            ## coef is highest power first, so poly_order-i to match coef order
        self.fit += coef[poly_order] # one more += for _0_to_ht**0
        self.coef = coef

        if self.fit_MA == None:
            self.set_all(False)
        self.set_center_and_curve_radius(self.coef)

    def fit_vs_MA(self):
        ''' Returns norm of difference of current fit x positions vs. its MA
        '''
        return linNorm(self.fit - self.fit_MA)


_max = SNS(
    fitdiff= 1700, # poor left lane fit if < 1700 on project vid @38sec
    fit_err= 80,
    fail= 2,
)

def bad_fit_diff(lf_coef, rt_coef, lf_radius, rt_radius, min_radm=1000):
    ''' Returns if the difference between coefs of left and right fits are more 
        than an order of magnitude than expected.
    min_radm: Min expected radius in meters. Adjust to actual min of road radius.
    ''' 
    lf_coef = np.array(lf_coef)
    rt_coef = np.array(rt_coef)
    fitdiff = lf_coef-rt_coef
    min_of_2 = np.minimum(np.absolute(lf_coef), np.absolute(rt_coef)) 
    expected = (min_radm * 2)/(abs(lf_radius)+abs(rt_radius))
    return linNorm(fitdiff[:2]/min_of_2[:2]) * expected > 10

def fitMsg(fit_vs_MA, fit_err):
    if fit_vs_MA > _max.fitdiff and fit_err > _max.fit_err:
        return 'Bad - Fit diff vs. MA: %5d, Fit error too big: %5d' % (fit_vs_MA, fit_err)
    elif fit_vs_MA > _max.fitdiff:
        return 'Bad - Fit diff vs. MA: %5d' % fit_vs_MA
    elif fit_err > _max.fit_err:
        return 'Bad - Fit error too large: %5d' % fit_err
    else:
        return 'OK - Fit diff vs. MA: %5d' % fit_vs_MA

line = [Line(), Line()]
fail_detect = [2, 2]

def process_image(_img):
    ''' Detection pipeline for image in video using L/R lines and LaneDetect obj.
    '''
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
        if detect.found(side):
            line[side].fit_curve(detect.pxs.x[side], detect.pxs.y[side])
            line[side].found = True
        else:
            fail_detect[side] += 1
            line[side].found = False

    badFit = bad_fit_diff(line[L].coef, line[R].coef, line[L].curve_rad, line[R].curve_rad)

    for side in LR:
        fit_vs_MA = line[side].fit_vs_MA()
        fit_err = line[side].fit_err

        if badFit:
            if fit_vs_MA > _max.fitdiff or fit_err > _max.fit_err:
                if detect.found(side): # +1 already above if not found, so +1 if found here
                    fail_detect[side] += 1
        else:
            if fit_err > _max.fit_err:
                if detect.found(side): # +1 already above if not found, so +1 if found here
                    fail_detect[side] += 1
            elif line[side].found and fit_vs_MA < _max.fitdiff:
                fail_detect[side] = 0
                line[side].set_all()

        fitmsg[side] = fitMsg(fit_vs_MA, fit_err)
        coef[side] = line[side].coef_MA

        for fit,thick in [(line[side].fit_MA, 10), (line[side].fit, 3)]:
            pts = np.array(np.vstack((fit, _0_to_ht)).T, dtype=np.int32).reshape((-1,1,2))
            cv2.polylines(detect.side_wins[2], [pts], False, (0,255,0), thickness=thick)
        
        line[side].set_center_and_curve_radius()

    ## calculate road radius and off center of vehicle from the 2 line values
    radm = (line[L].curve_rad + line[R].curve_rad)/2
    offcenter = (img_wd/2 - (line[L].center + line[R].center)/2) * xmppx

    out = detect.output_image(coef)
    return draw.with_debug_wins(out, detect.side_wins, offcenter, radm, fitmsg, detectmsg)
