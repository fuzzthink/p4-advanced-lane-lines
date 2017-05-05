import numpy as np
import cv2, os, sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from lane_detect import *
from camera import undistort
from util import Obj

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


font = cv2.FONT_HERSHEY_SIMPLEX
linetype = cv2.LINE_AA 

def sideWins(detect):
    ''' Returns stacked side wins image
    ''' 
    xratio = 0.32
    yratio = 0.34
    vis1 = cv2.resize(detect.vis1, None, None, xratio, yratio, cv2.INTER_AREA)
    vis2 = cv2.resize(detect.vis2, None, None, xratio, yratio, cv2.INTER_AREA)
    vis3 = cv2.resize(detect.vis3, None, None, xratio, yratio, cv2.INTER_AREA)
    win_ht, win_wd = vis1.shape[:2]
    txt_row_ht = win_ht//5
    text1 = np.zeros((txt_row_ht, win_wd,3)).astype(np.uint8)
    text2 = np.zeros((txt_row_ht, win_wd,3)).astype(np.uint8)
    text3 = np.zeros((txt_row_ht, win_wd,3)).astype(np.uint8)
    for i in range(1,3):
        text1[:,:,i] = 255
        text2[:,:,i] = 255
        text3[:,:,i] = 255
    txtpos = (10, txt_row_ht-15)
    size = 1 
    fontwd = 2
    cv2.putText(text1,'Edge', txtpos, font,size,(0,0,0), fontwd, linetype)
    cv2.putText(text2,'Yellow',txtpos, font,size,(0,0,0), fontwd,linetype)
    cv2.putText(text3,'coef(R=cur, G=avg used)',txtpos,font,size,(0,0,0),fontwd,linetype)
    return np.vstack((text1,vis1, text2,vis2, text3,vis3))

def toFeet(n):
    return n * 3.28084

def getColor(msg):
    return (0,255,255) if msg.startswith('OK') else (255,0,0)

def addBtmText(img, out_ht, offcent, radm, fitmsg, detectmsg):
    ''' Returns stacked main image and bottom texts
    ''' 
    txt_row_ht = 25
    absRadm = abs(radm)
    txtimg = np.zeros((out_ht-img_ht, img_wd, 3)).astype(np.uint8)
    radtxt = 'Curve Radius: %dm/%dft to the %s' % \
     (absRadm, toFeet(absRadm), 'Right' if radm < 0 else 'Left')
    postxt = 'Car off center to the %s by %3.1fm/%4.1fft' % \
     ('left' if offcent > 0 else 'right', offcent, toFeet(offcent))
    toptxt = radtxt+'. '+postxt
    lfit = 'Lf Fit: %s' % fitmsg[L]
    rfit = 'Rt Fit: %s' % fitmsg[R]
    ldetect = 'Lf Detect: %s' % detectmsg
    rdetect = 'Rt Detect: %s' % detectmsg
    x,y = 10,20
    size = .5
    ht = int(size * 45)
    wd = 1
    cv2.putText(txtimg, toptxt,(x,y), font, size, (255,255,255), wd,linetype)
    cv2.putText(txtimg, lfit,   (x,y+ht*1),font,size,getColor(fitmsg[L]),wd,linetype)
    cv2.putText(txtimg, rfit,   (x,y+ht*2),font,size,getColor(fitmsg[R]),wd,linetype)
    cv2.putText(txtimg, ldetect,(x,y+ht*3),font,size,(255,255,255),wd,linetype)
    cv2.putText(txtimg, rdetect,(x,y+ht*4),font,size,(255,255,255),wd,linetype)
    return np.vstack((img, txtimg))

_max = Obj({
    'fitdiff': 1700, # poor left lane fit if < 1700 on project vid @38sec
    'fit_err': 80,
    'centdiff': 800*xmppx, # lane center difference in meters
    'curvature': 1500,
    'fail': 2,
})
_min = Obj({
    'centdiff': 500*xmppx,
})

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
    badCenter = not _min.centdiff <= abs(line[L].center - line[R].center) <= _max.centdiff

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
        cv2.polylines(detect.vis3, [pts], False, (0,255,0), thickness=10)
        
        pts = np.array(np.vstack((line[side].fit, ylinspace)).T, dtype=np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(detect.vis3, [pts], False, (255,0,0), thickness=2)
        line[side].set_center_and_curve_rad()

    radm = (line[L].curve_rad + line[R].curve_rad)/2
    offcenter = xmppx*(line[L].center + line[R].center)/2

    lineimg = detect.plot_curve(coef)
    sidewin = sideWins(detect)
    mainwin = addBtmText(lineimg, sidewin.shape[0], offcenter, radm, fitmsg, detectmsg)
    return np.hstack((mainwin, sidewin))


#-------------------------------------------------------------------------------------------

if __name__ == "__main__":
    vidpath = sys.argv[1] if len(sys.argv) > 1 else 'project_video.mp4'
    output = 'video-out.mp4'
    clip = VideoFileClip(vidpath)
    clip = clip.fl_image(pipeline)
    clip.write_videofile(output, audio=False)
