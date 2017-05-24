import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from types import SimpleNamespace as SNS
from lib.np_util import scale_255
from lib import draw

L, R = 0, 1
LR = (L, R)
Lo, Hi = 0, 1

def roi_warper(img, lane=None, debug=False):
    ''' Returns birdseyes view warped image from undistored first person view img.
    Modifies: lane.src, lane.dst
    lane: LaneDetect() object
    '''
    ht, wd = img.shape[:2]
    top = ht * .63
    btm = ht * .97
    qtr = wd/4
    src = np.float32([(220,btm),(595,top),(685,top),(1060,btm)])
    dst = np.float32([(qtr,ht),(qtr,-100),(3*qtr,-100),(3*qtr,ht)]) 

    M = cv2.getPerspectiveTransform(src, dst) # Matrix for mapping src to dst
    warped = cv2.warpPerspective(img, M, (wd,ht), flags=cv2.INTER_NEAREST)
    if lane:
        lane.src = src
        lane.dst = dst 
    if debug:
        pts = np.int32(src)
        pts = pts.reshape((-1, 1, 2))
        annotated = np.copy(img)
        annotated = cv2.polylines(annotated, [pts], True, (255,0,0), thickness=5)
        warped = cv2.line(warped, (int(wd/4), ht), (int(wd/4), -100), (255,0,0), thickness=5)
        warped = cv2.line(warped, (int(3*wd/4), ht), (int(3*wd/4), -100), (255,0,0), thickness=5)
        return annotated, warped
    else:
        return warped

def filtered_dict(img, 
    low =SNS(h=15, vx=20, vx_yel=10, sx=10, s=75, v=175), #bad @4sec challenge if s=80,v=180
    high=SNS(h=35, vx=120,vx_yel=60, sx=100)):
    ''' 
    Returns dict of 7 filtered channels on img of:
    'yel': yellow filter via h channel
    'white': white filter via s channel
    'white2': white filter with narrower thresholds
    'posEdge': possitive sobelx via v channel
    'negEdge': negative sobelx via v channel
    'yelPos': possitive sobelx via s channel for yellow
    'yelNeg': negative sobelx via s channel for yellow
    (all h,s,v channels are of hsv space)
    '''
    d = {}  
    
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
    h_ch = hsv[:,:,0]
    s_ch = hsv[:,:,1]
    v_ch = hsv[:,:,2]

    # color thresholds
    d['yel'] = np.zeros_like(h_ch)
    d['yel'][(h_ch >= low.h) & (h_ch <= high.h) & (s_ch >= low.s)] = 1

    d['white'] = np.zeros_like(v_ch)
    d['white'][(v_ch >= low.v+s_ch+20)] = 1

    d['white2'] = np.copy(d['white'])
    d['white2'][v_ch >= low.v+s_ch] = 1

    # edge with sobel
    vx_pos = cv2.Sobel(v_ch, cv2.CV_64F, 1, 0, ksize=3)
    vx_pos[vx_pos <= 0] = 0
    vx_pos = scale_255(vx_pos)

    vx_neg = np.copy(vx_pos)
    vx_neg[vx_neg > 0] = 0
    vx_neg = np.absolute(vx_neg)
    vx_neg = scale_255(vx_neg)

    d['posEdge'] = np.zeros_like(vx_pos)
    d['posEdge'][(vx_pos >= low.vx) & (vx_pos <= high.vx)] = 1

    d['negEdge'] = np.zeros_like(vx_neg)
    d['negEdge'][(vx_neg >= low.vx) & (vx_neg <= high.vx)] = 1

    sobelx = cv2.Sobel(s_ch, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = np.absolute(sobelx)
    sobelx = scale_255(sobelx)

    d['yelPos'] = np.zeros_like(sobelx)
    d['yelPos'][(sobelx >= low.sx) & (sobelx <= high.sx) 
              & (vx_pos >= low.vx_yel) & (vx_pos <= high.vx_yel)] = 1

    d['yelNeg'] = np.zeros_like(sobelx)
    d['yelNeg'][(sobelx >= low.sx) & (sobelx <= high.sx) 
              & (vx_neg >= low.vx_yel) & (vx_neg <= high.vx_yel)] = 1           
    return d
        
def filteredPixels(filtereds):
    ''' Returns nonzero pixels object for filters in filtereds in x and y dims.
    filtereds: warped filters dict (see filtered_dict() for list of filters)
    '''
    o = SNS(x={}, y={})
    for fltr in filtereds:
        nonzero = filtereds[fltr].nonzero()
        o.x[fltr] = np.array(nonzero[1])
        o.y[fltr] = np.array(nonzero[0])
    return o

# printThresholds = True
def xyfound(roi, filtereds, filteredPxs, img, windows=True,
    minpct={'cnt':.011, 'white':.011, 'yel':.032},
    maxpct={'cnt':1.1, 'white2':.88, 'white2InWins':.22, 'white':.55, 'whiteInWins':.055},
    minpxs=None, maxpxs=None):
    ''' Returns tuple of (x-coords, y-coords, found, filters-used) of lane detection

    roi: ROI of dict of different filtered pixels
    filtereds: warped filters dict (see filtered_dict() for list of filters)
    filteredPxs: filteredPixels() result
    windows: full detection via windows or not
    minpct, maxpct: min/max pixels % of total image pixels thresholds dict
    minpxs, maxpxs: min/max pixels thresholds dict, overrides minpct/maxpct
    '''
    nPixels = img.shape[0]*img.shape[1]

    # calc min and max pxs from pct params
    if not minpxs:
        minpxs = {}
        for k,v in minpct.items():
            minpxs[k] = int(v*.01*nPixels)
    minpx = SNS(**minpxs)

    if not maxpxs:
        maxpxs = {}
        for k,v in maxpct.items():
            maxpxs[k] = int(v*.01*nPixels)
    maxpx = SNS(**maxpxs)

    # global printThresholds
    # if printThresholds:
    #     print('Min Thresholds used:', minpxs)
    #     print('Max Thresholds used:', maxpxs)
    # printThresholds = False

    pxCnt, mean, stdev = {}, {}, {}
    for fltr in filtereds: 
        _pxCnt = np.sum(roi[fltr])
        if _pxCnt < minpx.cnt or (windows and _pxCnt > maxpx.cnt):
            continue
        pxCnt[fltr] = _pxCnt
        mean[fltr]  = np.mean(filteredPxs.x[fltr][roi[fltr]])
        stdev[fltr] = np.std(filteredPxs.x[fltr][roi[fltr]])
    
    filters = pxCnt.keys()
    pxCnt = SNS(**pxCnt)

    if windows:
        used = [c for c in filters if stdev[c]<35]
    else:
        used = [c for c in filters]

    # remove poor filters
    if 'posEdge' in used:
        if 'negEdge' not in filters:
            used.remove('posEdge')

    if 'yelPos' in used:
        if 'yelNeg' not in filters:
            used.remove('yelPos')

    if ('white2' in used):
        if 'white' in used:
            used.remove('white')
        if pxCnt.white2 > maxpx.white2 or (windows and pxCnt.white2 > maxpx.white2InWins):
            used.remove('white2')

    if 'white' in used and (pxCnt.white < minpx.white or pxCnt.white > maxpx.white):
        used.remove('white')

    if windows and 'white' in used and pxCnt.white > maxpx.whiteInWins:
        used.remove('white')

    if 'yel' in used and pxCnt.yel < minpx.yel:
        used.remove('yel')
    
    # combine all used channels
    combine_x, combine_y = [],[]
    for fltr in used:
        combine_x.append(filteredPxs.x[fltr][roi[fltr]])
        combine_y.append(filteredPxs.y[fltr][roi[fltr]])
    if combine_x:
        return np.concatenate(combine_x), np.concatenate(combine_y), True, used
    else:
        return None, None, False, used

def pxsObj():
    ''' Returns object of:
    x:  array of L/R array of pixels in x coordinates
    y:  array of L/R array of pixels in y coordinates
    found: array of L/R line found boolean
    '''
    return SNS(
        x = [[], []], 
        y = [[], []], 
        found = [False, False]
    )

def pxObj():
    ''' Returns object of:
    x:  array of L/R pixels in x coordinates
    y:  array of L/R pixels in y coordinates
    found: array of L/R line found boolean
    '''
    return SNS(
        x = [None, None],
        y = [None, None],
        found = [False, False]
    )

def filteredBoundsPxObj(wb, filtereds, filteredPxs, img):
    ''' Returns pixObj within window boundaries
    wb: sliding window boundaries obj
    filtereds: warped filters dict (see filtered_dict() for list of filters)
    filteredPxs: filteredPixels() result
    '''
    px = pxObj()
    for side in LR:
        roi = {}
        for fltr in filtereds:
            roi[fltr] = (
                (filteredPxs.x[fltr] >  wb.x0[side]) & 
                (filteredPxs.x[fltr] <  wb.x1[side]) & 
                (filteredPxs.y[fltr] >  wb.y0) & 
                (filteredPxs.y[fltr] <= wb.y1)
            )
        px.x[side], px.y[side], px.found[side], used = xyfound(roi, filtereds, filteredPxs, img)
    return px, used

def noLaneFound(gap, x_mean, img_wd):
    ''' Returns True if gap between lanes is too little or too big
    gap: gap btw the lanes
    x_mean: mean of x pixels of L/R lanes
    '''
    minGap, maxGap = img_wd*.28, img_wd*.6
    minL, maxR = -55, img_wd+55
    return (gap < minGap) or (gap > maxGap) or (x_mean[L] < minL) or (x_mean[R] > maxR)


font = cv2.FONT_HERSHEY_SIMPLEX
fontsize = 1.8
linetype = cv2.LINE_AA 
txtwd = 3

class LaneDetect:
    def __init__(self, img):
        self.img = img
        self.img_ht = img.shape[0]
        self.img_wd = img.shape[1]

    def found(self, side):
        return self.pxsCoords.found[side]

    def warpedChannels(self, fdict, unwarp_wht_yel=False):
        ''' Returns warped fdict and white and yellow channels
        fdict: filtered channels of img dict
        '''
        warped = {fltr: roi_warper(img, lane=self) for fltr,img in fdict.items()}
        if unwarp_wht_yel:
            wht = np.dstack((fdict['posEdge'], fdict['negEdge'], fdict['white']))
            yel = np.dstack((fdict['yelPos'], fdict['yelNeg'], fdict['yel']))
        else:
            wht = np.dstack((warped['posEdge'], warped['negEdge'], warped['white']))
            yel = np.dstack((warped['yelPos'], warped['yelNeg'], warped['yel']))
        return warped, wht, yel

    def init_dbg_wins(self, img1, img2):
        self.dbg_wins = [np.copy(img1)*255]
        self.dbg_wins.append(np.copy(img2)*255)
        self.dbg_wins.append(np.zeros_like(self.dbg_wins[0]))

    def findlinepixs_fast(self, coef, margin=80):
        ''' Find lane lines with mean and stddev via 1 window for L and R sides.
        Sets self.pxsCoords to pxsObj() of result
        coef: Array of L,R fitted line coefficients
        '''
        filtereds, wht, yel = self.warpedChannels(filtered_dict(self.img))
        pxsCoords = pxsObj()
        pixels = filteredPixels(filtereds) 
        self.init_dbg_wins(wht, yel)

        for side in LR:
            roi={}
            for fltr in filtereds:
                roi[fltr] = (
                 (pixels.x[fltr] > (
                    coef[side][0] * pixels.y[fltr]**2 + 
                    coef[side][1] * pixels.y[fltr] + 
                    coef[side][2] - margin
                 )) & 
                 (pixels.x[fltr] < (
                    coef[side][0] * pixels.y[fltr]**2 + 
                    coef[side][1] * pixels.y[fltr] + 
                    coef[side][2] + margin
                 ))
                )

            pxsCoords.x[side], pxsCoords.y[side], pxsCoords.found[side], used = \
                xyfound(roi, filtereds, pixels, self.img, windows=False) 
                
            if pxsCoords.found[side]:
                self.dbg_wins[2][pxsCoords.y[side], pxsCoords.x[side], 2] = 255
        self.pxsCoords = pxsCoords

    def findlinepixs(self, nwindows=10):
        ''' Find lane lines with mean and stddev via sliding windows
        Sets self.pxsCoords to pxsObj() of result
        '''
        filtereds, wht, yel = self.warpedChannels(filtered_dict(self.img))
        self.init_dbg_wins(wht, yel)

        win_ht = self.img_ht//nwindows
        qtr_wd = self.img_wd//4
        _3qtrs = qtr_wd*3
        default_margin = 150

        margin = [qtr_wd, qtr_wd]
        x_mean = [qtr_wd, _3qtrs]
        prvx_mean = [qtr_wd, _3qtrs]
        prvx = [None, None] # save prvx_mean if it's good
        
        pixels = filteredPixels(filtereds) 
        pxsCoords = pxsObj()
        lanes_gap = self.img_wd//2
        momentum = [0, 0]
        last_update = [0, 0]

        # Step through the windows one by one
        for i in range(nwindows):

            # window boundaries
            wb = SNS(
                x0 = [int(x_mean[side] - margin[side]) for side in LR],
                x1 = [int(x_mean[side] + margin[side]) for side in LR],
                y0 = self.img_ht - (i+1)*win_ht,
                y1 = self.img_ht - i*win_ht,
            )
            px, used = filteredBoundsPxObj(wb, filtereds, pixels, self.img)

            if noLaneFound(lanes_gap, x_mean, self.img_wd):
                break

            gap = min(lanes_gap, self.img_wd//2)
            if px.found[L] or px.found[R]:
                margin[L] = 50 if px.found[L] else 150
                margin[R] = 50 if px.found[R] else 150
                x_mean[L] = np.mean(px.x[L]) if px.found[L] else np.mean(px.x[R]) - gap
                x_mean[R] = np.mean(px.x[R]) if px.found[R] else np.mean(px.x[L]) + gap
                for side in LR:
                    x_mean[side] = np.int(x_mean[side])
                if px.found[L] or px.found[R]:
                    lanes_gap = (lanes_gap + prvx_mean[R] - prvx_mean[L])/2
            else:
                margin[L] = default_margin
                margin[R] = default_margin

                
            for side in LR:
                # Add good pixels to list
                if px.found[side]:
                    pxsCoords.x[side].append(px.x[side])
                    pxsCoords.y[side].append(px.y[side])
                    prvx_mean[side] = x_mean[side]
                
                # Draw the windows on the visualization image
                draw.rect(self.dbg_wins[0], 
                    ((wb.x0[side],wb.y0), (wb.x1[side],wb.y1)), (0,255,side*255))  
                draw.rect(self.dbg_wins[1], 
                    ((wb.x0[side],wb.y0), (wb.x1[side],wb.y1)), (0,255,side*255)) 
                self.dbg_wins[2][px.y[side], px.x[side], 2] = 255

                if px.found[side]:
                    txt = '%d'%x_mean[side]
                    txtpos = x_mean[side]-130 if len(txt)==3 else x_mean[side]-175
                    cv2.putText(self.dbg_wins[2], txt, (txtpos, wb.y0+win_ht-13), 
                        font, fontsize, (0,255,255), txtwd, linetype)
            
                # decide window centers based on previous windows 
                last_update[side] += 1
                if prvx[side]:
                    momentum[side] = np.int(momentum[side]) 
                    if px.found[side]:
                        momentum[side] += ((x_mean[side] - prvx[side])/(last_update[side]))//2
                if px.found[side]: 
                    prvx[side] = x_mean[side]
                    last_update[side] = 0
                x_mean[side] += momentum[side]

        for side in LR:
            if pxsCoords.x[side]:
                pxsCoords.found[side] = True
                pxsCoords.x[side] = np.concatenate(pxsCoords.x[side])
                pxsCoords.y[side] = np.concatenate(pxsCoords.y[side])
            else:
                pxsCoords.x[side] = None
                pxsCoords.y[side] = None
        self.pxsCoords = pxsCoords

    def plot_curve(self, coef, poly_order=2):
        ''' Returns fitted lane lines image 
        '''
        img = self.img
        _0toHt = np.linspace(0, self.img_ht-1, self.img_ht)
        fit = [None, None]
        for side in LR:
            if coef[side] != None:
                fit[side] = coef[side][poly_order]
                for i in range(poly_order):
                    fit[side] += coef[side][i] * _0toHt**(poly_order-i)
                fit[side] = fit[side]
                    
        if fit[L] != None and fit[R] != None:
            lane_img = np.zeros((self.img_ht, self.img_wd+300, 3))
            pts_x = np.hstack((fit[0], fit[1][::-1]))
            pts_y = np.hstack((_0toHt, _0toHt[::-1]))
            pts = np.vstack((pts_x, pts_y)).T
            cv2.fillPoly(lane_img, np.int32([pts]), (0,255, 0))

            # unwarp image
            Minv = cv2.getPerspectiveTransform(self.dst, self.src)
            lane_img = cv2.warpPerspective(lane_img, Minv, (self.img_wd, self.img_ht))
            plot_img = cv2.addWeighted(img, 1, np.uint8(lane_img), 0.3, 0)
            return plot_img
        else:
            return img
